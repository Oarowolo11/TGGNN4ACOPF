#!/usr/bin/env python3
"""
Hybrid Message-Passing Neural Network for AC Optimal Power Flow
N-1 Contingency Datakit Dataset Version

Adapts the full topology dataset to the N-1 contingency dataset.

Key differences from the full topology version:
  1. VARIABLE TOPOLOGY: Per scenario, generators or branches may be disconnected.
     The last feature column in gen_data (in_service) and edge_data (br_status)
     indicates the active status of each element (1 = active, 0 = disconnected).
  2. FIXED r,x: Branch resistance/reactance do NOT vary across scenarios
     (unlike the full topology dataset). We use samp_grid_*_features directly.
  3. VARIABLE GRAPH SIZE: HeteroData objects have variable numbers of generator
     nodes and physical edges per scenario, depending on which elements are active.
  4. PE ENCODINGS: Computed on-the-fly via a contingency cache keyed by the set
     of active AC lines and transformers. Since N-1 has relatively few unique
     branch topologies, the cache stays small and is built once before training.
  5. line_h5_indices, transformer_h5_indices, active_gen_indices are no longer
     constant — they are determined per scenario from the H5 status columns.
"""

import os
import gc
import h5py
import argparse
import wandb
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import scipy
from sklearn.metrics import mean_squared_error
import math
import networkx as nx
import seaborn as sns
import time
from torch.nn import Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import ChebConv, GraphConv, GCNConv, TAGConv, GATConv
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import torch_scatter
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from torch_scatter import scatter_softmax, scatter_sum
from torch.amp import autocast, GradScaler
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.attention import PerformerAttention
import shutil

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# SCRATCH-LOCAL COPY 
# ============================================================================

def move_to_scratch(h5_source_path: str) -> str:
    """Copy HDF5 file to local NVMe scratch on Cluster for faster I/O."""
    scratch_dir = os.environ.get('TMPDIR')
    if not scratch_dir:
        print("TMPDIR not found. Running from original path (Slow).")
        return h5_source_path
    file_name = os.path.basename(h5_source_path)
    scratch_path = os.path.join(scratch_dir, file_name)
    if not os.path.exists(scratch_path):
        print(f"Copying to Cluster NVMe: {scratch_path}")
        start = time.time()
        shutil.copy2(h5_source_path, scratch_path)
        print(f"Copy finished in {time.time() - start:.1f}s")
    return scratch_path


# ============================================================================
# HDF5 DATASET  (unchanged from full-topology version)
# H5 expected column layout:
#   bus_data  : (n_scenarios, n_buses,  4)  [Pd, Qd, Vm, Va]
#   edge_data : (n_scenarios, n_edges,  7)  [r, x, b_ch, ..., ..., ..., br_status]
#   gen_data  : (n_scenarios, n_gens,   6)  [Pg, Qg, cp0, cp1, cp2, in_service]
#   edge_index: (2, n_edges)               fixed topology indexing
# ============================================================================

class H5PowerGridDataset(Dataset):
    """Loads the entire HDF5 file into RAM for ultra-fast per-sample access."""

    def __init__(self, h5_file_path: str, indices=None, transform=None):
        self.transform = transform

        print(f"Loading H5 Dataset into RAM: {h5_file_path}")
        st = time.time()

        with h5py.File(h5_file_path, 'r') as f:
            self.n_total = int(f.attrs['n_scenarios'])
            self.bus_data   = torch.from_numpy(f['bus_data'][:])
            self.edge_data  = torch.from_numpy(f['edge_data'][:])
            self.gen_data   = torch.from_numpy(f['gen_data'][:])
            self.edge_index = torch.tensor(f['edge_index'][:], dtype=torch.long)

        print(f"H5 Load Complete in {time.time()-st:.2f}s.")

        self.indices = list(indices) if indices is not None else list(range(self.n_total))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        scenario_idx = self.indices[idx]
        sample = {
            'bus_data':     self.bus_data[scenario_idx],
            'edge_data':    self.edge_data[scenario_idx],
            'gen_data':     self.gen_data[scenario_idx],
            'edge_index':   self.edge_index,
            'scenario_idx': scenario_idx,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ============================================================================
# POSITIONAL ENCODING FUNCTIONS  (shared with full-topology version)
# ============================================================================

def compute_gandb(edge_inputs: np.ndarray):
    """Compute branch conductance g and susceptance b from r and x (columns 4:6)."""
    line_r = edge_inputs[:, 4:5]
    line_x = edge_inputs[:, 5:6]
    line_g = line_r  / (line_r**2 + line_x**2)
    line_b = -line_x / (line_r**2 + line_x**2)
    return line_g, line_b


def get_B_matrix(N: int, edges: list, edge_weights: torch.Tensor) -> torch.Tensor:
    """Build symmetric susceptance adjacency matrix of shape (N, N)."""
    B_matrix = torch.zeros((N, N), dtype=torch.float64)
    sources, destinations = zip(*edges)
    B_matrix[sources, destinations] = edge_weights.squeeze()
    B_matrix[destinations, sources] = edge_weights.squeeze()
    return B_matrix


def adjacency_to_laplacian(B_adj: np.ndarray) -> np.ndarray:
    """Convert susceptance adjacency matrix to its graph Laplacian."""
    assert B_adj.shape[0] == B_adj.shape[1], "Input must be square"
    B_laplacian = B_adj.copy()
    np.fill_diagonal(B_laplacian, -B_adj.sum(axis=1))
    return B_laplacian


def effective_resistance_matrix(b_mat: np.ndarray) -> np.ndarray:
    """
    Compute effective resistance matrix from a susceptance Laplacian.
    Handles singularity by removing the reference (last) node.
    """
    L = b_mat
    n = L.shape[0]
    keep = np.arange(n - 1)
    L_reduced = L[np.ix_(keep, keep)]
    L_reduced_inv = np.linalg.inv(L_reduced)
    L_plus = np.zeros((n, n))
    L_plus[np.ix_(keep, keep)] = L_reduced_inv
    I    = np.eye(n)
    ones = np.ones((n, n)) / n
    L_plus = (I - ones) @ L_plus @ (I - ones)
    diag = np.diag(L_plus)
    R = diag[:, None] + diag[None, :] - 2 * L_plus
    return R


def compute_row_statistics_vectorized(resistance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute per-bus statistics over the effective resistance row (diagonal excluded).
    Returns shape (N, 5): [mean, median, std, max, min].
    """
    N = resistance_matrix.shape[0]
    mask = ~np.eye(N, dtype=bool)
    stats_matrix = np.zeros((N, 5))
    for i in range(N):
        row = resistance_matrix[i, mask[i]]
        stats_matrix[i, 0] = np.mean(row)
        stats_matrix[i, 1] = np.median(row)
        stats_matrix[i, 2] = np.std(row)
        stats_matrix[i, 3] = np.max(row)
        stats_matrix[i, 4] = np.min(row)
    return stats_matrix


# ============================================================================
# N-1–SPECIFIC PE HELPERS
# ============================================================================

def build_edge_inputs_pe(
    ac_feats:        np.ndarray,   # (n_ac_lines,  ≥9 features)
    xfmr_feats:      np.ndarray,   # (n_trafos,    ≥11 features)
    active_ac_idx:   np.ndarray,   # indices into ac_feats that are active
    active_xfmr_idx: np.ndarray,   # indices into xfmr_feats that are active
) -> np.ndarray:
    """
    Rearrange active AC-line and transformer features into the unified
    11-column edge_inputs format used by compute_gandb().

    Column layout (mirrors hybridMPNN_for_datakit.py):
      0-8  : branch parameters (from OPFData feature layout)
      9    : 1.0 for AC lines, 0.0 for transformers
      10   : transformer tap/shift (0.0 for AC lines)
    """
    n_ac   = len(active_ac_idx)
    n_xfmr = len(active_xfmr_idx)
    total  = n_ac + n_xfmr
    ei     = np.zeros((total, 11))

    if n_ac > 0:
        ei[:n_ac, :9]   = ac_feats[active_ac_idx]
        ei[:n_ac, 9:10] = 1.0              # is-AC-line flag

    if n_xfmr > 0:
        xf = xfmr_feats[active_xfmr_idx]  # raw transformer feature rows
        # Rearrange transformer columns to match the unified layout
        ei[n_ac:, :2]  = xf[:, :2]        # r, x
        ei[n_ac:, 2:4] = xf[:, 9:]        # tap magnitude, tap angle
        ei[n_ac:, 4:9] = xf[:, 2:7]       # remaining branch params
        ei[n_ac:, 9:]  = xf[:, 7:9]       # transformer-specific flags

    return ei


def compute_bus_pe_for_topology(
    ac_feats:        np.ndarray,
    xfmr_feats:      np.ndarray,
    active_ac_idx:   np.ndarray,
    active_xfmr_idx: np.ndarray,
    ac_senders:      np.ndarray,   # (n_ac_lines,) from-bus for each AC line
    ac_receivers:    np.ndarray,   # (n_ac_lines,) to-bus
    xfmr_senders:    np.ndarray,   # (n_trafos,)   from-bus for each transformer
    xfmr_receivers:  np.ndarray,   # (n_trafos,)   to-bus
    system_size:     int,
) -> torch.Tensor:
    """
    Compute bus positional encoding (effective-resistance statistics) for a
    specific N-1 topology identified by active_ac_idx and active_xfmr_idx.

    Returns a float32 tensor of shape (system_size, 5).
    """
    ei = build_edge_inputs_pe(ac_feats, xfmr_feats, active_ac_idx, active_xfmr_idx)

    # Collect sender/receiver lists for active branches only
    active_s = np.concatenate([
        ac_senders[active_ac_idx].flatten(),
        xfmr_senders[active_xfmr_idx].flatten()
    ])
    active_r = np.concatenate([
        ac_receivers[active_ac_idx].flatten(),
        xfmr_receivers[active_xfmr_idx].flatten()
    ])
    branch_list_local = list(zip(active_s, active_r))

    _, edge_b  = compute_gandb(ei)
    B_weighted = get_B_matrix(system_size, branch_list_local, torch.tensor(edge_b))
    B_lap      = adjacency_to_laplacian(np.array(B_weighted))
    e_R        = effective_resistance_matrix(B_lap)
    raw_PE     = compute_row_statistics_vectorized(e_R)

    return torch.tensor(raw_PE, dtype=torch.float)


def compute_pe_cache(
    h5_file_path:           str,
    line_h5_indices:        np.ndarray,   # which H5 edge rows are AC lines
    transformer_h5_indices: np.ndarray,   # which H5 edge rows are transformers
    samp_grid_ac_line_features:      np.ndarray,   # constant AC line features
    samp_grid_transformer_features:  np.ndarray,   # constant transformer features
    grid_ac_line_senders:   np.ndarray,
    grid_ac_line_receivers: np.ndarray,
    grid_transformer_senders:   np.ndarray,
    grid_transformer_receivers: np.ndarray,
    system_size:  int,
    n_scenarios:  int,
) -> Tuple[dict, list]:
    """
    Pre-compute bus PE for every unique branch contingency found in the dataset.

    Returns:
        contingency_pe_cache : dict  {key → Tensor(system_size, 5)}
        scenario_keys        : list  [key_for_scenario_0, key_for_scenario_1, ...]
    """
    n_ac   = len(line_h5_indices)
    n_xfmr = len(transformer_h5_indices)

    # ── Read only the branch-status column from H5 to save memory ──────────
    print("Reading branch statuses from H5 for PE cache computation...")
    with h5py.File(h5_file_path, 'r') as f:
        # Shape: (n_scenarios, n_edges) — only the last feature column
        all_edge_status = f['edge_data'][:n_scenarios, :, -1]   # float32 / int

    # Separate status for AC lines and transformers
    ac_statuses   = all_edge_status[:, line_h5_indices]        # (n_scenarios, n_ac)
    xfmr_statuses = all_edge_status[:, transformer_h5_indices] # (n_scenarios, n_xfmr)
    del all_edge_status
    gc.collect()

    # ── Full-topology PE (used for generator contingencies) ─────────────────
    full_active_ac   = np.arange(n_ac,   dtype=int)
    full_active_xfmr = np.arange(n_xfmr, dtype=int)
    full_key         = (tuple(full_active_ac.tolist()), tuple(full_active_xfmr.tolist()))

    print("Computing full-topology PE...")
    full_topology_pe = compute_bus_pe_for_topology(
        samp_grid_ac_line_features, samp_grid_transformer_features,
        full_active_ac, full_active_xfmr,
        grid_ac_line_senders,  grid_ac_line_receivers,
        grid_transformer_senders, grid_transformer_receivers,
        system_size,
    )
    contingency_pe_cache = {full_key: full_topology_pe}

    # ── Iterate through all scenarios and cache unique topologies ────────────
    scenario_keys = []
    for i in tqdm(range(n_scenarios), desc="Computing N-1 PE cache"):
        active_ac   = tuple(np.where(ac_statuses[i]   != 0)[0].tolist())
        active_xfmr = tuple(np.where(xfmr_statuses[i] != 0)[0].tolist())
        key = (active_ac, active_xfmr)
        scenario_keys.append(key)

        if key not in contingency_pe_cache:
            active_ac_idx   = np.array(active_ac,   dtype=int)
            active_xfmr_idx = np.array(active_xfmr, dtype=int)

            # If all branches are active, this is a generator contingency
            if len(active_ac_idx) == n_ac and len(active_xfmr_idx) == n_xfmr:
                contingency_pe_cache[key] = full_topology_pe
            else:
                contingency_pe_cache[key] = compute_bus_pe_for_topology(
                    samp_grid_ac_line_features, samp_grid_transformer_features,
                    active_ac_idx, active_xfmr_idx,
                    grid_ac_line_senders,  grid_ac_line_receivers,
                    grid_transformer_senders, grid_transformer_receivers,
                    system_size,
                )

    print(f"Unique contingency topologies found: {len(contingency_pe_cache)}")
    return contingency_pe_cache, scenario_keys


# ============================================================================
# N-1 HETERODATA CONVERTER
# Replaces H5ToHeteroDataConverter from the full-topology script.
# ============================================================================

class N1H5ToHeteroDataConverter:
    """
    Callable transform that converts a raw H5 sample dict into a HeteroData
    object for the N-1 contingency dataset.

    Args:
        samp_grid_bus              : np.ndarray (n_buses, n_bus_feat)       constant
        samp_grid_generator        : np.ndarray (n_opf_gens, n_gen_feat)   constant base features
        samp_grid_shunt            : np.ndarray (n_shunts, n_shunt_feat)   constant
        samp_grid_ac_line_features : np.ndarray (n_ac_lines, n_line_feat)  constant (r,x fixed in N-1)
        samp_grid_transformer_features : np.ndarray (n_trafos, n_trafo_feat) constant
        generator_indices          : array-like  bus indices for each OPFData generator
        load_indices               : array-like  bus indices for each load
        shunt_indices              : array-like  bus indices for each shunt
        ac_line_senders            : array-like  (n_ac_lines,) from-bus
        ac_line_receivers          : array-like  (n_ac_lines,) to-bus
        transformer_senders        : array-like  (n_trafos,)   from-bus
        transformer_receivers      : array-like  (n_trafos,)   to-bus
        line_h5_indices            : array-like  H5 edge rows corresponding to AC lines
        transformer_h5_indices     : array-like  H5 edge rows corresponding to transformers
        datakit_active_gen_index   : array-like  H5 gen rows that map to OPFData generators
        scenario_pe_list           : list of Tensor (n_buses, 5) — one per scenario
    """

    def __init__(
        self,
        samp_grid_bus:                    np.ndarray,
        samp_grid_generator:              np.ndarray,
        samp_grid_shunt:                  np.ndarray,
        samp_grid_ac_line_features:       np.ndarray,
        samp_grid_transformer_features:   np.ndarray,
        generator_indices,
        load_indices,
        shunt_indices,
        ac_line_senders,
        ac_line_receivers,
        transformer_senders,
        transformer_receivers,
        line_h5_indices,
        transformer_h5_indices,
        datakit_active_gen_index,
        scenario_pe_list,
    ):
        # ── Constant node features ───────────────────────────────────────────
        self.bus_features_const   = torch.tensor(samp_grid_bus.astype(np.float32),   dtype=torch.float)
        self.shunt_features_const = torch.tensor(samp_grid_shunt.astype(np.float32), dtype=torch.float)

        # ── Base features that may be partially updated per scenario ─────────
        # For N-1: r,x are FIXED so these arrays are used AS-IS (no cols 4:6 update)
        # Gencost coefficients (cols 8:) may still vary per scenario
        self._base_gen_features     = samp_grid_generator.astype(np.float32).copy()
        self._base_ac_line_features = samp_grid_ac_line_features.astype(np.float32).copy()
        self._base_trafo_features   = samp_grid_transformer_features.astype(np.float32).copy()

        # ── Topology index arrays (all stored as numpy for fast slicing) ─────
        self.base_gen_indices   = np.array(generator_indices).flatten()  # (n_opf_gens,) bus indices
        self.load_indices_np    = np.array(load_indices).flatten()
        self.shunt_indices_np   = np.array(shunt_indices).flatten()

        # Full AC-line and transformer sender/receiver arrays (pre-converted to tensors)
        self.all_ac_senders    = torch.tensor(np.array(ac_line_senders).flatten(),      dtype=torch.long)
        self.all_ac_receivers  = torch.tensor(np.array(ac_line_receivers).flatten(),    dtype=torch.long)
        self.all_tr_senders    = torch.tensor(np.array(transformer_senders).flatten(),  dtype=torch.long)
        self.all_tr_receivers  = torch.tensor(np.array(transformer_receivers).flatten(), dtype=torch.long)

        # ── Constant pseudo-edge indices (load and shunt edges never change) ─
        load_indices_t   = torch.tensor(self.load_indices_np,  dtype=torch.long)
        shunt_indices_t  = torch.tensor(self.shunt_indices_np, dtype=torch.long)

        load_node_ids  = torch.arange(len(self.load_indices_np),  dtype=torch.long)
        shunt_node_ids = torch.arange(len(self.shunt_indices_np), dtype=torch.long)

        self.load_edge_index  = torch.stack([load_node_ids,  load_indices_t],  dim=0)
        self.shunt_edge_index = torch.stack([shunt_node_ids, shunt_indices_t], dim=0)
        self.load_edge_attr   = torch.ones((len(self.load_indices_np),  3))
        self.shunt_edge_attr  = torch.ones((len(self.shunt_indices_np), 3))

        # PE lookup tensors for load and shunt (constant, use full set of bus PEs)
        self.load_indices_t  = load_indices_t
        self.shunt_indices_t = shunt_indices_t

        # ── H5 edge-row → AC-line / transformer mapping ──────────────────────
        self.line_h5_idx  = np.array(line_h5_indices)
        self.trafo_h5_idx = np.array(transformer_h5_indices)

        # ── H5 gen-row → OPFData generator mapping ───────────────────────────
        # base_datakit_gen_idx[k] is the H5 gen row for OPFData generator k
        self.base_datakit_gen_idx = np.array(datakit_active_gen_index)

        # ── PE: indexed by scenario_idx ──────────────────────────────────────
        # scenario_pe_list[i] is a Tensor(system_size, 5) for scenario i
        self.scenario_pe_list = scenario_pe_list

    # ── Main conversion ───────────────────────────────────────────────────────
    def __call__(self, h5_sample: dict) -> HeteroData:
        """
        Build a fully-populated HeteroData from one N-1 H5 sample.

        Variable-topology handling:
          - Active AC lines    : rows where edge_data[line_h5_idx, -1] != 0
          - Active transformers: rows where edge_data[trafo_h5_idx, -1] != 0
          - Active generators  : rows where gen_data[datakit_gen_idx, -1] != 0
          Generator and physical-edge features / edge_index are filtered accordingly.
        """
        bus_data     = h5_sample['bus_data']
        edge_data    = h5_sample['edge_data']
        gen_data     = h5_sample['gen_data']
        scenario_idx = int(h5_sample['scenario_idx'])

        # Convert to numpy for fast slicing
        bus_np  = bus_data.numpy()  if isinstance(bus_data,  torch.Tensor) else bus_data
        edge_np = edge_data.numpy() if isinstance(edge_data, torch.Tensor) else edge_data
        gen_np  = gen_data.numpy()  if isinstance(gen_data,  torch.Tensor) else gen_data

        # ── 1. Determine active branches for this scenario ───────────────────
        # line_h5_idx maps H5 edge rows → position in AC-line list
        ac_status   = edge_np[self.line_h5_idx,  -1]   # (n_ac_lines,)  status flags
        xfmr_status = edge_np[self.trafo_h5_idx, -1]   # (n_trafos,)    status flags

        # Local indices within the AC-line / transformer arrays that are active
        active_ac_local_idx   = np.where(ac_status   != 0)[0]  # subset of [0..n_ac-1]
        active_xfmr_local_idx = np.where(xfmr_status != 0)[0]  # subset of [0..n_xfmr-1]

        # ── 2. Determine active generators for this scenario ─────────────────
        # base_datakit_gen_idx maps OPFData generator k → its row in H5 gen_data
        gen_status           = gen_np[self.base_datakit_gen_idx, -1]  # (n_opf_gens,)
        active_gen_local_mask = gen_status != 0                        # bool mask
        active_gen_local_idx  = np.where(active_gen_local_mask)[0]    # subset of [0..n_opf_gens-1]

        # Corresponding H5 gen rows and bus connections for active generators
        active_h5_gen_idx    = self.base_datakit_gen_idx[active_gen_local_idx]
        active_gen_bus_idx   = self.base_gen_indices[active_gen_local_idx]  # bus indices

        # ── 3. PE lookup (pre-computed, indexed by scenario_idx) ─────────────
        bus_pe = self.scenario_pe_list[scenario_idx]   # Tensor(n_buses, 5)

        # ── 4. Load features (variable demand per scenario) ──────────────────
        # bus_data columns: [Pd, Qd, Vm, Va] — take load at load_indices
        load_features = torch.tensor(
            bus_np[self.load_indices_np, :2].copy(), dtype=torch.float
        )   # (n_loads, 2)

        # ── 5. Solution bus (Vm, Va → Va_rad, Vm) ────────────────────────────
        sol_bus_np = bus_np[:, 2:4].copy()           # [Vm, Va] from H5
        sol_bus_np = sol_bus_np[:, ::-1].copy()      # → [Va, Vm]
        sol_bus_np[:, 0] = np.radians(sol_bus_np[:, 0])  # Va: degrees → radians
        bus_solutions = torch.tensor(sol_bus_np, dtype=torch.float)  # (n_buses, 2)

        # ── 6. Solution generator (active gens only, per-unit) ───────────────
        sol_gen_np = gen_np[active_h5_gen_idx, :2] / 100.0  # [Pg, Qg] in p.u.
        generator_solutions = torch.tensor(sol_gen_np.copy(), dtype=torch.float)

        # ── 7. Generator input features (active gens only) ───────────────────
        gen_features_np = self._base_gen_features[active_gen_local_idx].copy()
        # Update gencost coefficients from H5 gen_data (cols 2:5 = cp0, cp1, cp2)
        # If gencosts are truly fixed in your N-1 dataset, this line is a no-op
        # but harmless. Comment it out to save a small amount of per-sample work.
        gencosts = gen_np[active_h5_gen_idx, 2:5]
        gen_features_np[:, 8:] = gencosts
        generator_features = torch.tensor(gen_features_np, dtype=torch.float)

        # ── 8. AC line features (active lines only, r,x are FIXED) ──────────
        # No r,x update from H5 — use the constant OPFData features directly
        ac_line_features = torch.tensor(
            self._base_ac_line_features[active_ac_local_idx].copy(), dtype=torch.float
        )

        # ── 9. Transformer features (active transformers only, r,x FIXED) ────
        trafo_features = torch.tensor(
            self._base_trafo_features[active_xfmr_local_idx].copy(), dtype=torch.float
        )

        # ── 10. Build edge_index tensors for active physical edges ────────────
        ac_senders_active   = self.all_ac_senders[active_ac_local_idx]
        ac_receivers_active = self.all_ac_receivers[active_ac_local_idx]
        tr_senders_active   = self.all_tr_senders[active_xfmr_local_idx]
        tr_receivers_active = self.all_tr_receivers[active_xfmr_local_idx]

        # ── 11. Generator pseudo-edge indices (active gens only) ─────────────
        n_active_gens   = len(active_gen_local_idx)
        gen_node_ids    = torch.arange(n_active_gens, dtype=torch.long)  # local gen IDs
        gen_bus_ids     = torch.tensor(active_gen_bus_idx, dtype=torch.long)

        # ── 12. Assemble HeteroData ───────────────────────────────────────────
        data = HeteroData()

        # Node features
        data['bus'].x       = self.bus_features_const   # constant (no bus outages)
        data['generator'].x = generator_features        # variable size per scenario
        data['load'].x      = load_features             # variable demand per scenario
        data['shunt'].x     = self.shunt_features_const # constant

        # Positional encodings
        data['bus'].pe       = bus_pe
        data['generator'].pe = bus_pe[gen_bus_ids]           # PE at generator buses
        data['load'].pe      = bus_pe[self.load_indices_t]
        data['shunt'].pe     = bus_pe[self.shunt_indices_t]

        # Store per-generator bus assignment so constraint evaluation can scatter
        # predictions onto the correct buses without re-deriving the active set.
        # Shape: (n_active_gens,) — PyG batches this into a flat concatenated tensor,
        # which we then split by batch['generator'].batch just like predictions.
        data['generator'].bus_idx = gen_bus_ids

        # Solution targets
        data['bus'].y       = bus_solutions
        data['generator'].y = generator_solutions

        # Physical edges: AC lines (active only)
        data['bus', 'ac_line', 'bus'].edge_index = torch.stack(
            [ac_senders_active, ac_receivers_active], dim=0
        )
        data['bus', 'ac_line', 'bus'].edge_attr = ac_line_features

        # Physical edges: transformers (active only)
        data['bus', 'transformer', 'bus'].edge_index = torch.stack(
            [tr_senders_active, tr_receivers_active], dim=0
        )
        data['bus', 'transformer', 'bus'].edge_attr = trafo_features

        # Pseudo-edges: generator → bus (active gens only)
        data['generator', 'connects_to', 'bus'].edge_index = torch.stack(
            [gen_node_ids, gen_bus_ids], dim=0
        )
        data['generator', 'connects_to', 'bus'].edge_attr = torch.ones((n_active_gens, 3))

        # Pseudo-edges: load → bus (constant)
        data['load', 'connects_to', 'bus'].edge_index = self.load_edge_index
        data['load', 'connects_to', 'bus'].edge_attr  = self.load_edge_attr

        # Pseudo-edges: shunt → bus (constant)
        data['shunt', 'connects_to', 'bus'].edge_index = self.shunt_edge_index
        data['shunt', 'connects_to', 'bus'].edge_attr  = self.shunt_edge_attr

        return data


# ============================================================================
# DATALOADER FACTORY  (N-1 version)
# ============================================================================

def create_n1_h5_dataloaders(
    h5_file_path:                    str,
    samp_grid_bus:                   np.ndarray,
    samp_grid_generator:             np.ndarray,
    samp_grid_shunt:                 np.ndarray,
    samp_grid_ac_line_features:      np.ndarray,
    samp_grid_transformer_features:  np.ndarray,
    generator_indices,
    load_indices,
    shunt_indices,
    ac_line_senders,
    ac_line_receivers,
    transformer_senders,
    transformer_receivers,
    line_h5_indices,
    transformer_h5_indices,
    datakit_active_gen_index,
    scenario_pe_list,            # list of Tensor(n_buses, 5), length = n_total_scenarios
    batch_size:   int = 256,
    num_workers:  int = 16,
    seed:         int = 42,
    data_len:     int = 300000,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build train / val / test PyG DataLoaders for the N-1 dataset.

    The converter handles variable-topology graphs transparently; PyG's
    built-in HeteroData batching stacks variable-size node/edge tensors.

    Returns:
        train_loader, val_loader, test_loader,
        train_indices, val_indices, test_indices
    """
    train_indices, val_indices, test_indices = train_val_test_split(data_len, seed=seed)

    print(f"Dataset split — train: {len(train_indices)}, "
          f"val: {len(val_indices)}, test: {len(test_indices)}")

    converter = N1H5ToHeteroDataConverter(
        samp_grid_bus=samp_grid_bus,
        samp_grid_generator=samp_grid_generator,
        samp_grid_shunt=samp_grid_shunt,
        samp_grid_ac_line_features=samp_grid_ac_line_features,
        samp_grid_transformer_features=samp_grid_transformer_features,
        generator_indices=generator_indices,
        load_indices=load_indices,
        shunt_indices=shunt_indices,
        ac_line_senders=ac_line_senders,
        ac_line_receivers=ac_line_receivers,
        transformer_senders=transformer_senders,
        transformer_receivers=transformer_receivers,
        line_h5_indices=line_h5_indices,
        transformer_h5_indices=transformer_h5_indices,
        datakit_active_gen_index=datakit_active_gen_index,
        scenario_pe_list=scenario_pe_list,
    )

    train_ds = H5PowerGridDataset(h5_file_path, indices=train_indices, transform=converter)
    val_ds   = H5PowerGridDataset(h5_file_path, indices=val_indices,   transform=converter)
    test_ds  = H5PowerGridDataset(h5_file_path, indices=test_indices,  transform=converter)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, prefetch_factor=2,
        persistent_workers=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers // 4, persistent_workers=True,
        prefetch_factor=2, pin_memory=True,
    )

    train_loader= None
    val_loader= None
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers // 4, persistent_workers=True,
        prefetch_factor=2, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_indices, val_indices, test_indices



# ============================================================================
# DATA LOADING  (unchanged from full-topology version)
# ============================================================================

def load_grid_data(system_size: int, data_dir: str):
    """Load OPFData formatted power system data for the given bus count."""
    filepath = os.path.join(data_dir, f'{system_size}bus_combined_dataset.npz')
    print(f"Loading OPFData from: {filepath}")

    data = np.load(filepath, mmap_mode='r')

    print("Available keys in OPFData dataset:")
    for key in data.files:
        print(f"  {key}: shape {data[key].shape}")

    grid_bus                   = data['grid_bus'][0].copy()
    grid_generator             = data['grid_generator'][0].copy()
    grid_load                  = data['grid_load'][0].copy()
    grid_shunt                 = data['grid_shunt'][0].copy()
    grid_ac_line_features      = data['grid_ac_line_features'][0].copy()
    grid_transformer_features  = data['grid_transformer_features'][0].copy()
    grid_ac_line_receivers     = data['grid_ac_line_receivers'].copy()
    grid_ac_line_senders       = data['grid_ac_line_senders'].copy()
    grid_transformer_senders   = data['grid_transformer_senders'].copy()
    grid_transformer_receivers = data['grid_transformer_receivers'].copy()

    solution_bus       = data['solution_bus'][0].copy()
    solution_generator = data['solution_generator'][0].copy()
    solution_objective = data['metadata_objective'].copy().reshape(-1, 1)

    branch_list      = list(zip(grid_ac_line_senders[0],      grid_ac_line_receivers[0]))
    transformer_list = list(zip(grid_transformer_senders[0],  grid_transformer_receivers[0]))
    branch_list.extend(transformer_list)

    generator_indices = data['grid_generator_link_receivers'][0].copy()
    load_indices      = data['grid_load_link_receivers'][0].copy()
    shunt_indices     = data['grid_shunt_link_receivers'][0].copy()
    grid_transformer_senders   = grid_transformer_senders[0].copy()
    grid_transformer_receivers = grid_transformer_receivers[0].copy()
    grid_ac_line_senders       = grid_ac_line_senders[0].copy()
    grid_ac_line_receivers     = grid_ac_line_receivers[0].copy()

    del data
    gc.collect()

    return (grid_bus, grid_generator, grid_load, grid_shunt, grid_ac_line_features,
            grid_ac_line_senders, grid_ac_line_receivers, grid_transformer_features,
            grid_transformer_senders, grid_transformer_receivers, solution_bus,
            solution_generator, solution_objective, branch_list, generator_indices,
            load_indices, shunt_indices)


# ============================================================================
# DATA PREPARATION  (N-1 version — simpler than full topology)
# ============================================================================

def prepare_datasets_n1(
    datakit_edge_index,        # (2, n_edges) full H5 edge index
    grid_bus,
    grid_generator,
    grid_shunt,
    grid_ac_line_features,     # (n_ac_lines, n_feat)  constant in N-1
    grid_transformer_features, # (n_trafos,   n_feat)  constant in N-1
    grid_ac_line_senders,
    grid_ac_line_receivers,
    grid_transformer_senders,
    grid_transformer_receivers,
    branch_list,
    generator_indices,
    load_indices,
    shunt_indices,
    full_topo_gen_in_service , # # np.ndarray (n_h5_gens,) float
    full_topo_branch_active
):
    """
    Prepare topology metadata for the N-1 dataset.

    Compared to the full-topology prepare_datasets:
      • We do NOT filter edges by status: in N-1 every branch in H5 may be
        active in at least one scenario (the full-topology scenario), so
        no branch should be treated as permanently disconnected.
      • We do NOT update r,x from H5: in N-1 these are fixed.
      • We find which H5 gen rows correspond to OPFData generators by checking
        the in_service flag in the reference scenario (first scenario / scenario 0).

    Returns all the arrays needed by N1H5ToHeteroDataConverter and PE cache.
    """
    # ── Find which generators are "active" in the reference topology ─────────
    # A generator belongs to the full topology if active in AT LEAST ONE scenario.

    datakit_active_gen_index = np.where(full_topo_gen_in_service == 1.0)[0]
    print(f"OPFData-aligned generators found in H5: {len(datakit_active_gen_index)}")


    # ── Match H5 edge rows to AC lines vs transformers ────────────────────────
    # Step 1: remove always-off branches from the H5 edge index.
    # Some H5 datasets contain branches that are never active in any scenario
    # (always switched off).  These have no counterpart in OPFData and must be
    # excluded before matching, otherwise the counts are inflated.
    # full_topo_branch_active is the OR-union of branch status across all
    # scenarios — exactly the same logic used for generators above.
    active_h5_edge_indices = np.where(full_topo_branch_active)[0]  # (n_ever_active,)
    datakit_edge_list      = datakit_edge_index[:, active_h5_edge_indices]  # (2, n_ever_active)
 
    print(f"H5 edges ever active: {len(active_h5_edge_indices)} "
          f"(dropped {datakit_edge_index.shape[1] - len(active_h5_edge_indices)} always-off branches)")

    # ── Match H5 edge rows to AC lines vs transformers ────────────────────────
    # We use the full edge_index (no status filtering) because individual scenarios
    # may disable any branch.  The OPFData transformer_branches define the boundary.
    transformer_branches = np.array(branch_list[-grid_transformer_features.shape[0]:])
    

    # matches[e, t] == True when H5 edge e matches OPFData transformer t
    matches = (
        (datakit_edge_list[0][:, None] == transformer_branches[:, 0]) &
        (datakit_edge_list[1][:, None] == transformer_branches[:, 1])
    )

    transformer_h5_local_idx = np.where(matches.any(axis=1))[0]  # H5 rows = transformers
    line_h5_local_idx        = np.where(~matches.any(axis=1))[0]  # H5 rows = AC lines

    transformer_h5_indices = active_h5_edge_indices[transformer_h5_local_idx]
    line_h5_indices        = active_h5_edge_indices[line_h5_local_idx]

    print(f"H5 AC-line rows: {len(line_h5_indices)}, "
          f"H5 transformer rows: {len(transformer_h5_indices)}")
    
    assert len(transformer_h5_indices) == grid_transformer_features.shape[0], (
    f"Transformer count mismatch: got {len(transformer_h5_indices)}, "
    f"expected {grid_transformer_features.shape[0]}")
    assert len(line_h5_indices) == grid_ac_line_features.shape[0], (
        f"AC-line count mismatch: got {len(line_h5_indices)}, "
        f"expected {grid_ac_line_features.shape[0]}")

    # ── Use constant (sample-independent) base features ──────────────────────
    # In N-1, r,x do NOT change across scenarios, so we use OPFData features as-is.
    samp_grid_bus               = grid_bus.copy()
    samp_grid_generator         = grid_generator.copy()
    samp_grid_shunt             = grid_shunt.copy()
    samp_grid_ac_line_features  = grid_ac_line_features.copy()
    samp_grid_transformer_features = grid_transformer_features.copy()

    return (
        samp_grid_bus, samp_grid_generator, samp_grid_shunt,
        samp_grid_ac_line_features, samp_grid_transformer_features,
        generator_indices, load_indices, shunt_indices,
        grid_ac_line_senders, grid_ac_line_receivers,
        grid_transformer_senders, grid_transformer_receivers,
        line_h5_indices, transformer_h5_indices, datakit_active_gen_index,
    )


# ============================================================================
# TRAIN/VAL/TEST SPLIT  (unchanged)
# ============================================================================

def train_val_test_split(data_len, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, seed=42):
    """Reproducibly split scenario indices into train / val / test sets."""
    torch.manual_seed(seed)
    indices    = torch.randperm(data_len)
    train_size = int(train_ratio * data_len)
    val_size   = int(val_ratio   * data_len)
    return (
        indices[:train_size].numpy(),
        indices[train_size : train_size + val_size].numpy(),
        indices[train_size + val_size:].numpy(),
    )


# ============================================================================
# MODEL ARCHITECTURE  (identical to full-topology version)
# ============================================================================

class MLP(torch.nn.Module):
    """Multi-Layer Perceptron with optional LayerNorm and LeakyReLU."""

    def __init__(self, input_size, hidden_size, output_size, layers,
                 layernorm=True, use_leaky=False):
        super().__init__()
        modules = []
        for i in range(layers):
            modules.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                modules.append(torch.nn.ReLU())
            if use_leaky:
                modules.append(torch.nn.LeakyReLU(negative_slope=0.02))
        if layernorm:
            modules.append(torch.nn.LayerNorm(output_size))
        self.network = torch.nn.Sequential(*modules)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.network:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        return self.network(x)


class HeteroPerformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn = PerformerAttention(
            channels=hidden_dim,
            heads=num_heads
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Post-attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
    def forward(self, x_dict, xm_dict, batch_dict):
            # Flatten all node types
            flat_x, flat_xm, flat_batch = [], [], []
            slices = {}
            offset = 0
    
            for ntype in x_dict:
                x = x_dict[ntype]
                xm = xm_dict[ntype]
                b = batch_dict[ntype]  # batch indices for each node
    
                slices[ntype] = slice(offset, offset + x.size(0))
                flat_x.append(x)
                flat_xm.append(xm)
                flat_batch.append(b)
                offset += x.size(0)
    
            x_all = torch.cat(flat_x, dim=0)         # [N, D]
            xm_all = torch.cat(flat_xm, dim=0)       # [N, D]
            global_batch_all = torch.cat(flat_batch, dim=0) # [N]

            _, batch_all = torch.unique(global_batch_all, return_inverse=True)
            sorted_indices = torch.argsort(batch_all) # resort batch index to be ascending but that means I have to sort the xs myself too
            batch_all_sorted = batch_all[sorted_indices]
            x_all_sorted = x_all[sorted_indices]
            xm_all_sorted = xm_all[sorted_indices]
            # Convert to [B, N_max, D] and mask            
            x_dense, mask = to_dense_batch(x_all_sorted, batch_all_sorted)   # [B, N, D], [B, N]
            
            # Apply masked Performer attention
            x_attn = self.attn(x_dense, mask=mask)              # [B, N, D]
            # Residual + Norm
            xt_out = self.norm1(x_dense + x_attn)          
            # Unpad: [real_nodes, D]
            xt_out = xt_out[mask]
    
            x_comb = self.norm2(xt_out + xm_all_sorted)
            x_final = self.mlp(x_comb)

            # dont forget to give x_final it unsorted arrangement
            unsorted_x = torch.empty_like(x_final)
            unsorted_x[sorted_indices] = x_final
            x_final = unsorted_x
            # x_final = x_final[sorted_indices]

            # Unflatten by slice
            return {ntype: x_final[slices[ntype]] for ntype in x_dict.keys()}


class HeteroInteractionNetwork(nn.Module):
    """
    One message-passing step over a heterogeneous graph.
    Physical edges (AC lines, transformers) include edge features in messages;
    pseudo-edges (gen/load/shunt → bus) use only node features.
    """

    def __init__(self, node_types, edge_types, physical_edge_types, hidden_size, layers):
        super().__init__()
        self.physical_edge_types = physical_edge_types

        self.edge_updaters = nn.ModuleDict()
        for src, rel, dst in edge_types:
            key       = f"{src}_{rel}_{dst}"
            edge_type = (src, rel, dst)
            in_dim    = hidden_size * 3 if edge_type in physical_edge_types else hidden_size * 2
            self.edge_updaters[key] = MLP(in_dim, hidden_size, hidden_size, layers)

        self.node_updaters = nn.ModuleDict({
            nt: MLP(hidden_size * 2, hidden_size, hidden_size, layers)
            for nt in node_types
        })

    def forward(self, x_dict, edge_indices_dict, edge_features_dict):
        updated_edge_features = {}
        aggregated_messages   = {nt: torch.zeros_like(feat) for nt, feat in x_dict.items()}

        for edge_type, edge_index in edge_indices_dict.items():
            src_type, rel_type, dst_type = edge_type
            edge_key = f"{src_type}_{rel_type}_{dst_type}"
            src, dst = edge_index
            x_i = x_dict[dst_type][dst]
            x_j = x_dict[src_type][src]

            if edge_type in self.physical_edge_types:
                edge_feature = edge_features_dict[edge_type]
                updated_edge = self.edge_updaters[edge_key](
                    torch.cat((x_i, x_j, edge_feature), dim=-1)
                )
                updated_edge = updated_edge + edge_feature   # residual
                updated_edge_features[edge_type] = updated_edge
            else:
                updated_edge = self.edge_updaters[edge_key](
                    torch.cat((x_i, x_j), dim=-1)
                )
                updated_edge_features[edge_type] = updated_edge

            # Cast before scatter (needed for mixed-precision)
            aggregated_messages[dst_type] = aggregated_messages[dst_type].to(updated_edge.dtype)
            aggregated_messages[src_type] = aggregated_messages[src_type].to(updated_edge.dtype)
            aggregated_messages[dst_type] = torch_scatter.scatter_add(
                updated_edge, dst, dim=0, out=aggregated_messages[dst_type])
            aggregated_messages[src_type] = torch_scatter.scatter_add(
                updated_edge, src, dim=0, out=aggregated_messages[src_type])

        updated_nodes = {}
        for nt, x in x_dict.items():
            node_update = self.node_updaters[nt](
                torch.cat((x, aggregated_messages[nt]), dim=-1)
            )
            updated_nodes[nt] = x + node_update  # residual

        return updated_nodes, updated_edge_features


class HeteroInteractGNN(torch.nn.Module):
    """
    Heterogeneous GPS-style GNN for ACOPF.
    Architecture identical to the full-topology version — variable graph sizes
    are handled naturally by PyG batching.
    """

    def __init__(self, hidden_size=256, n_mp_layers=5, bus_features=4,
                 gen_features=11, load_features=2, shunt_features=2,
                 ac_line_features=9, transformer_features=11,
                 connects_to_features=3, output_dim=2):
        super().__init__()

        self.node_types = ['bus', 'generator', 'load', 'shunt']
        self.edge_types = [
            ('bus', 'ac_line',     'bus'),
            ('bus', 'transformer', 'bus'),
            ('generator', 'connects_to', 'bus'),
            ('load',      'connects_to', 'bus'),
            ('shunt',     'connects_to', 'bus'),
        ]
        self.physical_edge_types = [
            ('bus', 'ac_line',     'bus'),
            ('bus', 'transformer', 'bus'),
        ]

        self.node_encoders = nn.ModuleDict({
            'bus':       MLP(bus_features,   hidden_size, hidden_size - 5, 2),
            'generator': MLP(gen_features,   hidden_size, hidden_size - 5, 2),
            'load':      MLP(load_features,  hidden_size, hidden_size - 5, 2),
            'shunt':     MLP(shunt_features, hidden_size, hidden_size - 5, 2),
        })
        self.global_attn_layers = nn.ModuleList([
            HeteroPerformerLayer(hidden_size) for _ in range(n_mp_layers)
        ])
        self.edge_encoders = nn.ModuleDict({
            'ac_line':     MLP(ac_line_features,      hidden_size, hidden_size, 2),
            'transformer': MLP(transformer_features,  hidden_size, hidden_size, 2),
        })
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([
            HeteroInteractionNetwork(
                self.node_types, self.edge_types, self.physical_edge_types, hidden_size, 2
            )
            for _ in range(n_mp_layers)
        ])
        self.node_decoders = nn.ModuleDict({
            'bus':       MLP(hidden_size, hidden_size, output_dim, 2, layernorm=False),
            'generator': MLP(hidden_size, hidden_size, output_dim, 2, layernorm=False),
        })

    def forward(self, data):
        x_dict = {}
        for nt in self.node_types:
            if hasattr(data[nt], 'x'):
                enc = self.node_encoders[nt](data[nt].x)
                x_dict[nt] = torch.cat([enc, data[nt].pe], dim=-1)

        edge_feature_dict = {}
        for src, rel, dst in self.edge_types:
            et = (src, rel, dst)
            if (et in self.physical_edge_types and et in data.edge_types
                    and hasattr(data[et], 'edge_attr')):
                edge_feature_dict[et] = self.edge_encoders[rel](data[et].edge_attr)

        edge_index_dict = {
            et: data[et].edge_index
            for et in self.edge_types
            if et in data.edge_types and hasattr(data[et], 'edge_index')
        }
        batch_dict = {nt: data[nt].batch for nt in x_dict}

        for i in range(self.n_mp_layers):
            xm_dict, edge_feature_dict = self.layers[i](x_dict, edge_index_dict, edge_feature_dict)
            x_dict = self.global_attn_layers[i](x_dict, xm_dict, batch_dict)

        return {
            'bus':       torch.sigmoid(self.node_decoders['bus'](x_dict['bus'])),
            'generator': torch.sigmoid(self.node_decoders['generator'](x_dict['generator'])),
        }


# ============================================================================
# BOUND HELPERS  (unchanged)
# ============================================================================

def convert_voltage_bounds(model_input):
    num_nodes   = model_input.shape[0]
    vmin        = model_input[:, 2:3];  vmax = model_input[:, 3:4]
    thetamin    = torch.tensor([-2.00]).to(device).tile((num_nodes, 1))
    thetamax    = torch.tensor([ 2.00]).to(device).tile((num_nodes, 1))
    bounds_up   = torch.cat((thetamax, vmax), dim=1)
    bounds_down = torch.cat((thetamin, vmin), dim=1)
    return bounds_up, bounds_down


def convert_power_bounds(model_input):
    pmin = model_input[:, 2:3];  pmax = model_input[:, 3:4]
    qmin = model_input[:, 5:6];  qmax = model_input[:, 6:7]
    bounds_up   = torch.cat((pmax, qmax), dim=1)
    bounds_down = torch.cat((pmin, qmin), dim=1)
    return bounds_up, bounds_down


# ============================================================================
# TRAINING / VALIDATION  (unchanged from full-topology version)
# ============================================================================

def train_model(model, trainloader, optimizer):
    """Train model for one epoch; returns mean MSE loss."""
    model.train()
    total_loss = 0
    criterion  = nn.MSELoss()

    for batch in trainloader:
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device, non_blocking=True)

        with autocast(device_type=str(device), dtype=torch.bfloat16):
            pred = model(batch)

            v_up, v_down = convert_voltage_bounds(batch['bus'].x)
            voltages = torch.clamp(
                pred['bus'] * (v_up - v_down) + v_down, min=v_down, max=v_up)

            p_up, p_down = convert_power_bounds(batch['generator'].x)
            powers = torch.clamp(
                pred['generator'] * (p_up - p_down) + p_down, min=p_down, max=p_up)

            combined_targets = torch.cat([batch['bus'].y, batch['generator'].y], dim=0)
            combined_outputs = torch.cat([voltages, powers], dim=0)
            loss = criterion(combined_targets, combined_outputs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(trainloader)


@torch.no_grad()
def validate_model(model, val_loader):
    """Evaluate on validation set; returns mean MSE loss."""
    model.eval()
    total_loss = 0
    criterion  = nn.MSELoss()

    for batch in val_loader:
        batch = batch.to(device, non_blocking=True)
        with autocast(device_type=str(device), dtype=torch.bfloat16):
            pred = model(batch)

            v_up, v_down = convert_voltage_bounds(batch['bus'].x)
            voltages = torch.clamp(
                pred['bus'] * (v_up - v_down) + v_down, min=v_down, max=v_up)

            p_up, p_down = convert_power_bounds(batch['generator'].x)
            powers = torch.clamp(
                pred['generator'] * (p_up - p_down) + p_down, min=p_down, max=p_up)

            combined_targets = torch.cat([batch['bus'].y, batch['generator'].y], dim=0)
            combined_outputs = torch.cat([voltages, powers], dim=0)
            total_loss += criterion(combined_targets, combined_outputs).item()

    return total_loss / len(val_loader)


# ============================================================================
# TEST FUNCTION  (N-1 adapted)
# Because N-1 graphs have a VARIABLE number of active generators, we cannot
# simply reshape the generator predictions to (n_test, n_gens, 2).  Instead we
# collect per-scenario tensors by splitting on batch['generator'].batch.
# ============================================================================

@torch.no_grad()
def test_model_n1(model, testloader, system_size):
    """
    Run inference on the test set and collect predictions.

    Returns:
        test_loss        : float
        v_predictions    : Tensor  (n_test, n_buses, 2)   — voltage  [Va, Vm]
        v_targets        : Tensor  (n_test, n_buses, 2)
        p_predictions    : list of Tensor  (n_active_gens_i, 2) per test scenario
        p_targets        : list of Tensor  (n_active_gens_i, 2) per test scenario
        gen_inputs_list  : list of Tensor  (n_active_gens_i, n_gen_feat) for optimality
    """
    model.eval()
    criterion   = nn.MSELoss()
    total_loss  = 0.0

    all_v_preds   = []
    all_v_targets = []
    p_predictions   = []   # one tensor per scenario
    p_targets       = []
    gen_inputs_list = []
    gen_bus_idx_list = []  # per-scenario tensor of active generator bus indices

    for batch in testloader:
        batch = batch.to(device, non_blocking=True)

        with autocast(device_type=str(device), dtype=torch.bfloat16):
            pred = model(batch)

            v_up, v_down = convert_voltage_bounds(batch['bus'].x)
            voltages = torch.clamp(
                pred['bus'] * (v_up - v_down) + v_down, min=v_down, max=v_up)

            p_up, p_down = convert_power_bounds(batch['generator'].x)
            powers = torch.clamp(
                pred['generator'] * (p_up - p_down) + p_down, min=p_down, max=p_up)

            combined_targets = torch.cat([batch['bus'].y, batch['generator'].y], dim=0)
            combined_outputs = torch.cat([voltages, powers], dim=0)
            total_loss += criterion(combined_targets, combined_outputs).item()

        # ── Voltage: same bus count for every scenario in N-1 ────────────────
        n_graphs = int(batch['bus'].batch.max().item()) + 1
        # All buses are always present; reshape directly
        all_v_preds.append(voltages.cpu().view(n_graphs, system_size, 2))
        all_v_targets.append(batch['bus'].y.cpu().view(n_graphs, system_size, 2))

        # ── Power + bus mapping: variable generator count per scenario ────────
        gen_batch_ids    = batch['generator'].batch.cpu()
        powers_cpu       = powers.cpu()
        gen_targets_cpu  = batch['generator'].y.cpu()
        gen_inputs_cpu   = batch['generator'].x.cpu()
        # bus_idx was stored in the converter; PyG batches it as a flat tensor
        gen_bus_ids_cpu  = batch['generator'].bus_idx.cpu()

        for g in range(n_graphs):
            mask = gen_batch_ids == g
            p_predictions.append(powers_cpu[mask])
            p_targets.append(gen_targets_cpu[mask])
            gen_inputs_list.append(gen_inputs_cpu[mask])
            gen_bus_idx_list.append(gen_bus_ids_cpu[mask])  # (n_active_gens_g,)

    v_predictions = torch.cat(all_v_preds,   dim=0)  # (n_test, n_buses, 2)
    v_targets     = torch.cat(all_v_targets, dim=0)  # (n_test, n_buses, 2)

    return (total_loss / len(testloader),
            v_predictions, v_targets,
            p_predictions, p_targets,
            gen_inputs_list,
            gen_bus_idx_list)


# ============================================================================
# EVALUATION UTILITIES  (N-1 adapted)
# ============================================================================

def compute_gandb_3d(edge_inputs):
    """Compute (g, b) from 3-D edge tensor (batch, edges, 11)."""
    r = edge_inputs[:, :, 4:5];  x = edge_inputs[:, :, 5:6]
    return r / (r**2 + x**2), -x / (r**2 + x**2)


def convert_to_complex_voltage(voltage_tensor):
    """[Va_rad, Vm] → [Vreal, Vimag]."""
    va, vm = voltage_tensor[:, :, 0:1], voltage_tensor[:, :, 1:]
    return torch.cat((vm * torch.cos(va), vm * torch.sin(va)), dim=2)


def convert_to_complex_rectangle_3D(tensor_3d):
    """(mag, angle) → (real, imag) for 3-D tensors."""
    mag, ang = tensor_3d[:, :, 0:1], tensor_3d[:, :, 1:]
    return torch.cat((mag * torch.cos(ang), mag * torch.sin(ang)), dim=2)


def calculate_only_branch_flows(demand, voltage, branches, Yks, Yij, Yijc, Tij):
    """
    Compute branch power flows and net generator injections.
    Identical to the full-topology version — operates on the ACTIVE branch list
    provided for each call.
    """
    def cmul(a, b):
        return torch.stack([a[...,0]*b[...,0] - a[...,1]*b[...,1],
                            a[...,0]*b[...,1] + a[...,1]*b[...,0]], dim=-1)
    def cconj(x):
        return torch.stack([x[...,0], -x[...,1]], dim=-1)
    def cdiv(a, b):
        d = b[...,0]**2 + b[...,1]**2
        return torch.stack([(a[...,0]*b[...,0]+a[...,1]*b[...,1])/d,
                            (a[...,1]*b[...,0]-a[...,0]*b[...,1])/d], dim=-1)

    num_nodes       = voltage.shape[1]
    generator_power = torch.zeros_like(demand)
    v_sq = torch.cat([torch.sum(voltage**2,dim=-1,keepdim=True),
                      torch.zeros(demand.shape[0], num_nodes, 1)], dim=-1)
    shunt_power = cmul(cconj(Yks), v_sq)
    branch_flows = {}

    for idx, (i, j) in enumerate(branches):
        vi, vj = voltage[:, i], voltage[:, j]
        vi_sq  = torch.cat([torch.sum(vi**2,dim=-1,keepdim=True),
                            torch.zeros(vi.shape[0],1)], dim=-1)
        tij_sq = torch.cat([torch.sum(Tij[:,idx]**2,dim=-1,keepdim=True),
                            torch.zeros(vi.shape[0],1)], dim=-1)
        Y_tot  = Yij[:,idx] + Yijc[:,idx]
        term1  = cmul(cconj(Y_tot), cdiv(vi_sq, tij_sq))
        term2  = cmul(cconj(Yij[:,idx]), cdiv(cmul(vi, cconj(vj)), Tij[:,idx]))
        branch_flows[(i, j, idx)] = term1 - term2

        vj_sq = torch.cat([torch.sum(vj**2,dim=-1,keepdim=True),
                           torch.zeros(vj.shape[0],1)], dim=-1)
        t1ji  = cmul(cconj(Y_tot), vj_sq)
        t2ji  = cmul(cconj(Yij[:,idx]),
                     cdiv(cmul(cconj(vi), vj), cconj(Tij[:,idx])))
        branch_flows[(j, i, idx)] = t1ji - t2ji

    for i in range(num_nodes):
        for idx, (fb, tb) in enumerate(branches):
            if fb == i:
                generator_power[:, i] += branch_flows[(fb, tb, idx)]
            if tb == i:
                generator_power[:, i] += branch_flows[(tb, fb, idx)]

    generator_power += demand + shunt_power
    return generator_power, branch_flows


def convert_to_power_magnitude(power_flow):
    """Complex (real, imag) → magnitude, shape (..., 1)."""
    return torch.sqrt(power_flow[...,0]**2 + power_flow[...,1]**2).unsqueeze(-1)


def compute_optimality_n1(
    gen_inputs_list: list,   # per-scenario Tensor(n_active_gens, n_feat)
    gen_preds_list:  list,   # per-scenario Tensor(n_active_gens, 2)
    test_objectives: np.ndarray,   # per-scenario true optimal cost
) -> torch.Tensor:
    """
    Compute optimality gap for N-1 dataset.

    Handles variable generator counts per scenario by processing each
    scenario individually and aggregating afterwards.

    Costs are stored in generator input features:
      col 8  = cp2  (quadratic)
      col 9  = cp1  (linear)
      col 10 = cp0  (constant)
    Outputs are in per-unit; multiply by 100 to get MW before applying $/MW costs.
    """
    model_objectives = []
    for gen_inputs, gen_preds in zip(gen_inputs_list, gen_preds_list):
        c2 = gen_inputs[:, 8:9]
        c1 = gen_inputs[:, 9:10]
        c0 = gen_inputs[:, 10:11]
        p_gens = gen_preds[:, 0:1]                   # active power in p.u.
        # Cost polynomial evaluated at MW output (p.u. × 100 → MW)
        obj = c2 * (100 * p_gens)**2 + c1 * (100 * p_gens) + c0
        model_objectives.append(obj.sum().item())

    model_obj  = torch.tensor(model_objectives, dtype=torch.float32)
    test_obj_t = torch.tensor(test_objectives,  dtype=torch.float32)

    print(f"Average model objective : {model_obj.mean():.4f}")
    print(f"Average IPOPT objective : {test_obj_t.mean():.4f}")

    opt_gap = (model_obj / test_obj_t) * 100
    return opt_gap.mean()


def calculate_angle_differences(angles: np.ndarray, edges: list) -> np.ndarray:
    """Compute voltage angle differences across each edge."""
    diffs = np.zeros(len(edges), dtype=np.float32)
    for k, (n1, n2) in enumerate(edges):
        diffs[k] = angles[n2] - angles[n1]
    return diffs


def evaluate_constraints_n1(
    v_predictions:         torch.Tensor,    # (n_test, n_buses, 2)
    p_predictions:         list,            # per-scenario Tensor(n_active_gens, 2)
    test_bus_inputs:       torch.Tensor,    # (n_test, n_buses, n_feat)
    test_generator_inputs: list,            # per-scenario Tensor(n_active_gens, n_feat)
    gen_bus_idx_list:      list,            # per-scenario Tensor(n_active_gens,) bus indices
    samp_grid_ac_line_features:     np.ndarray,
    samp_grid_transformer_features: np.ndarray,
    samp_grid_shunt:       np.ndarray,
    branch_list:           list,            # full OPFData branch list
    load_indices:          np.ndarray,
    shunt_indices:         np.ndarray,
    generator_indices:     np.ndarray,      # full OPFData generator bus indices (unused now)
    test_scenario_keys:    list,            # contingency key per test scenario
    test_bus_data:         np.ndarray,      # (n_test, n_buses, 4) from H5
    line_h5_indices:       np.ndarray,
    transformer_h5_indices: np.ndarray,
    grid_ac_line_senders:  np.ndarray,
    grid_ac_line_receivers: np.ndarray,
    grid_transformer_senders:  np.ndarray,
    grid_transformer_receivers: np.ndarray,
    system_size:           int,
) -> dict:
    """
    Compute all constraint violations for N-1 test predictions.

    Strategy for branch-flow–dependent constraints (power balance, branch
    flow limits): iterate scenario-by-scenario using the per-scenario
    active branch topology.  Voltage and generator bound checks are
    vectorised across scenarios.
    """
    n_test  = v_predictions.shape[0]
    metrics = {}

    # ── Voltage angle difference violations (vectorised) ─────────────────────
    n_branches = len(branch_list)
    n_ac_branches = len(line_h5_indices)   # number of AC lines in full topology
 
    # active_branch_mask[i, b] == True  ⟺  branch b is active in scenario i
    active_branch_mask = np.zeros((n_test, n_branches), dtype=bool)
    for i, (active_ac, active_xfmr) in enumerate(test_scenario_keys):
        active_branch_mask[i, list(active_ac)] = True
        # transformer local indices are offset by n_ac_branches in the global list
        global_xfmr_idx = [n_ac_branches + k for k in active_xfmr]
        active_branch_mask[i, global_xfmr_idx] = True
    active_branch_mask = torch.tensor(active_branch_mask)   # (n_test, n_branches)
 
    # Compute angle differences for all branches (cheap vectorised operation)
    pred_angle_diffs = np.zeros((n_test, n_branches), dtype=np.float32)
    for j in range(n_test):
        pred_angle_diffs[j] = calculate_angle_differences(
            v_predictions[j, :, 0].numpy(), branch_list
        )
    pred_angle_diffs = torch.tensor(pred_angle_diffs)
 
    ang_upper = torch.full(pred_angle_diffs.shape,  0.5236)
    ang_lower = torch.full(pred_angle_diffs.shape, -0.5236)
    ang_viols_raw = (torch.clamp(ang_lower - pred_angle_diffs, min=0) +
                     torch.clamp(pred_angle_diffs - ang_upper, min=0))
 
    # Zero out violations on branches that are inactive in each scenario
    ang_viols = ang_viols_raw * active_branch_mask.float()
 
    # Report statistics over active branches only (exclude the zeroed-out entries)
    active_ang_viols = ang_viols[active_branch_mask]
    metrics['max_angle_diff_violation'] = active_ang_viols.max().item() if active_ang_viols.numel() > 0 else 0.0
    metrics['avg_angle_diff_violation'] = active_ang_viols.mean().item() if active_ang_viols.numel() > 0 else 0.0

    # ── Voltage magnitude violations (vectorised) ─────────────────────────────
    vmin = test_bus_inputs[:, :, 2:3]
    vmax = test_bus_inputs[:, :, 3:4]
    vmag_viols = (torch.clamp(vmin - v_predictions[:, :, 1:2], min=0) +
                  torch.clamp(v_predictions[:, :, 1:2] - vmax, min=0))
    metrics['max_vmag_violation'] = vmag_viols.max().item()
    metrics['avg_vmag_violation'] = vmag_viols.mean().item()

    # ── Generator bound violations (per-scenario, variable gen count) ─────────
    all_pgen_viols, all_qgen_viols = [], []
    for i, (gen_inp, gen_pred) in enumerate(zip(test_generator_inputs, p_predictions)):
        pmin = gen_inp[:, 2:3];  pmax = gen_inp[:, 3:4]
        qmin = gen_inp[:, 5:6];  qmax = gen_inp[:, 6:7]
        p_gens = gen_pred[:, 0:1];  q_gens = gen_pred[:, 1:2]
        all_pgen_viols.append((torch.clamp(pmin - p_gens, min=0) +
                               torch.clamp(p_gens - pmax, min=0)).flatten())
        all_qgen_viols.append((torch.clamp(qmin - q_gens, min=0) +
                               torch.clamp(q_gens - qmax, min=0)).flatten())

    pgen_viols = torch.cat(all_pgen_viols)
    qgen_viols = torch.cat(all_qgen_viols)
    metrics['max_pgen_violation'] = pgen_viols.max().item()
    metrics['avg_pgen_violation'] = pgen_viols.mean().item()
    metrics['max_qgen_violation'] = qgen_viols.max().item()
    metrics['avg_qgen_violation'] = qgen_viols.mean().item()

    # ── Branch flow and power balance (per-scenario loop) ────────────────────
    # These depend on the active topology and therefore cannot be easily vectorised
    # across N-1 scenarios without grouping by contingency type first.
    all_fwd_viols, all_rev_viols = [], []
    all_real_balance, all_react_balance = [], []

    for i in range(n_test):
        active_ac, active_xfmr = test_scenario_keys[i]
        active_ac_idx   = np.array(active_ac,   dtype=int)
        active_xfmr_idx = np.array(active_xfmr, dtype=int)

        # Build this scenario's active branch list
        ac_s = grid_ac_line_senders[active_ac_idx]
        ac_r = grid_ac_line_receivers[active_ac_idx]
        tr_s = grid_transformer_senders[active_xfmr_idx]
        tr_r = grid_transformer_receivers[active_xfmr_idx]
        active_branch_list = (list(zip(ac_s, ac_r)) + list(zip(tr_s, tr_r)))

        n_active_branches = len(active_branch_list)
        if n_active_branches == 0:
            continue   # degenerate: skip islanded scenario

        # Assemble edge_inputs (11 columns, same layout as hybridMPNN_for_datakit.py)
        ei = np.zeros((n_active_branches, 11))
        n_ac = len(active_ac_idx)
        if n_ac > 0:
            ei[:n_ac, :9]   = samp_grid_ac_line_features[active_ac_idx]
            ei[:n_ac, 9:10] = 1.0
        if len(active_xfmr_idx) > 0:
            xf = samp_grid_transformer_features[active_xfmr_idx]
            ei[n_ac:, :2]  = xf[:, :2]
            ei[n_ac:, 2:4] = xf[:, 9:]
            ei[n_ac:, 4:9] = xf[:, 2:7]
            ei[n_ac:, 9:]  = xf[:, 7:9]

        # Build admittance tensors (batch dim = 1 for single scenario)
        ei_t = torch.tensor(ei, dtype=torch.float32).unsqueeze(0)  # (1, n_br, 11)
        g_b  = compute_gandb_3d(ei_t)                              # (1, n_br, 1) each
        Yij  = torch.cat(g_b, dim=2)                               # (1, n_br, 2)

        Yijc = torch.zeros_like(Yij)
        Yijc[:, :, 1:] = ei_t[:, :, 2:3]   # charging susceptance

        Tij = ei_t[:, :, 9:]                                  # (1, n_br, 2) [mag, ang]
        Tij_t = torch.zeros(1, n_active_branches, 2)
        Tij_t[:, :n_ac, 0] = 1.0                             # AC lines: tap = 1+0j
        if len(active_xfmr_idx) > 0:
            Tij_t[:, n_ac:] = Tij[:, n_ac:]                  # transformer taps
        Tij_rec = convert_to_complex_rectangle_3D(Tij_t)

        Yks = torch.zeros(1, system_size, 2, dtype=torch.float32)
        Yks[:, shunt_indices, :] = torch.tensor(
            samp_grid_shunt[np.newaxis], dtype=torch.float32)
        Yks = Yks[:, :, [1, 0]]  # swap column order to match convention

        # Load demand (from H5 bus_data, columns 0:2 = Pd, Qd)
        load_demand = torch.zeros(1, system_size, 2, dtype=torch.float32)
        load_demand[:, load_indices, :] = torch.tensor(
            test_bus_data[i, load_indices, :2][np.newaxis], dtype=torch.float32)

        # Complex voltage for this scenario
        complex_v = convert_to_complex_voltage(
            v_predictions[i].unsqueeze(0).float())  # (1, n_buses, 2)

        inj, bflows = calculate_only_branch_flows(
            load_demand, complex_v, active_branch_list, Yks, Yij, Yijc, Tij_rec)

        # ── Branch flow magnitude violations ─────────────────────────────────
        # Long-term rating is in samp_grid_ac_line_features col 6
        fwd_keys = [(s, r, k) for k, (s, r) in enumerate(active_branch_list)]
        rev_keys = [(r, s, k) for k, (s, r) in enumerate(active_branch_list)]

        fwd_flows = torch.stack([bflows[k] for k in fwd_keys], dim=1)  # (1, n_br, 2)
        rev_flows = torch.stack([bflows[k] for k in rev_keys], dim=1)

        fwd_mag = convert_to_power_magnitude(fwd_flows)   # (1, n_br, 1)
        rev_mag = convert_to_power_magnitude(rev_flows)

        # Branch rating: use AC-line ratings first, then transformer ratings
        ratings_ac   = samp_grid_ac_line_features[active_ac_idx, 6:7]        # col 6 = rating
        ratings_xfmr = samp_grid_transformer_features[active_xfmr_idx, 6:7]  # same column
        ratings = torch.tensor(
            np.concatenate([ratings_ac, ratings_xfmr]), dtype=torch.float32
        ).view(1, -1, 1)

        all_fwd_viols.append(torch.clamp(fwd_mag - ratings, min=0).flatten())
        all_rev_viols.append(torch.clamp(rev_mag - ratings, min=0).flatten())

        # ── Power balance mismatches ──────────────────────────────────────────
        # Scatter active-generator predictions onto their buses using the bus
        # index tensor that was stored per-scenario during data conversion.
        # This avoids any size mismatch: gen_bus_idx_list[i] and p_predictions[i]
        # are both of length n_active_gens_i by construction.
        gen_power   = torch.zeros(1, system_size, 2)
        p_pred_i    = p_predictions[i]          # (n_active_gens_i, 2)
        bus_ids_i   = gen_bus_idx_list[i]        # (n_active_gens_i,)  long tensor

        for k in range(p_pred_i.shape[0]):
            gen_power[0, bus_ids_i[k].item(), :] += p_pred_i[k]

        real_mismatch = (inj[0, :, 0] - gen_power[0, :, 0]).abs()
        reac_mismatch = (inj[0, :, 1] - gen_power[0, :, 1]).abs()
        all_real_balance.append(real_mismatch)
        all_react_balance.append(reac_mismatch)

    # Aggregate across all test scenarios
    if all_fwd_viols:
        metrics['max_forward_flow_violation']  = torch.cat(all_fwd_viols).max().item()
        metrics['avg_forward_flow_violation']  = torch.cat(all_fwd_viols).mean().item()
        metrics['max_reverse_flow_violation']  = torch.cat(all_rev_viols).max().item()
        metrics['avg_reverse_flow_violation']  = torch.cat(all_rev_viols).mean().item()
        metrics['max_real_power_balance']      = torch.cat(all_real_balance).max().item()
        metrics['avg_real_power_balance']      = torch.cat(all_real_balance).mean().item()
        metrics['max_reactive_power_balance']  = torch.cat(all_react_balance).max().item()
        metrics['avg_reactive_power_balance']  = torch.cat(all_react_balance).mean().item()
    else:
        for k in ['max_forward_flow_violation','avg_forward_flow_violation',
                  'max_reverse_flow_violation','avg_reverse_flow_violation',
                  'max_real_power_balance','avg_real_power_balance',
                  'max_reactive_power_balance','avg_reactive_power_balance']:
            metrics[k] = float('nan')

    return metrics


# ============================================================================
# MAIN  (N-1 version)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Hybrid MPNN for ACOPF (N-1 Dataset)')
    parser.add_argument('--system_size', type=int, required=True,
                        choices=[14, 30, 57, 118, 500, 2000])
    parser.add_argument('--datakit_dir', type=str,
                        default='TGGNN4ACOPF/Datakit_N1',
                        help='Directory containing N-1 Datakit HDF5 file')
    parser.add_argument('--opfdata_dir', type=str,
                        default='TGGNN4ACOPF/OPFData_fulltop',
                        help='Directory containing OPFData (full topology reference)')
    parser.add_argument('--output_dir',   type=str, default='./outputs')
    parser.add_argument('--batch_size',   type=int, default=256)
    parser.add_argument('--hidden_size',  type=int, default=256)
    parser.add_argument('--n_mp_layers',  type=int, default=5)
    parser.add_argument('--num_epochs',   type=int, default=100)
    parser.add_argument('--learning_rate',type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-8)
    parser.add_argument('--wandb_api_key',type=str, default='')
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--resume',       action='store_true')
    parser.add_argument('--start_epoch',  type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Training Hybrid MPNN for {args.system_size}-bus N-1 Contingency Dataset")
    print(f"{'='*80}\n")


    # File naming convention for N-1: pglib_opf_case{N}_ieee.h5
    raw_path    = os.path.join(args.datakit_dir,
                               f'pglib_opf_case{args.system_size}_ieee.h5')
    h5_file_path = move_to_scratch(raw_path)

    # ── 2. Read H5 metadata and reference data ────────────────────────────────
    print("Reading H5 metadata...")
    with h5py.File(h5_file_path, 'r') as f:
        datakit_edge_index       = f['edge_index'][:]          # (2, n_edges)
        # Only load first-scenario gen data for reference-topology generator detection
        gen_in_service_all       = f['gen_data'][:, :, -1]           # (n_gens, 6)
        full_topo_gen_in_service = gen_in_service_all.any(axis=0).astype(np.float32)
        n_scenarios              = int(f.attrs['n_scenarios'])
        branch_status_all         = f['edge_data'][:, :, -1]   # (n_scenarios, n_h5_edges)
        full_topo_branch_active   = branch_status_all.any(axis=0)  # (n_h5_edges,) bool
        del branch_status_all
        del gen_in_service_all

    print(f"Total N-1 scenarios in H5: {n_scenarios}")

    # ── 3. Load OPFData (full-topology reference topology + features) ─────────
    (grid_bus, grid_generator, grid_load, grid_shunt, grid_ac_line_features,
     grid_ac_line_senders, grid_ac_line_receivers, grid_transformer_features,
     grid_transformer_senders, grid_transformer_receivers, solution_bus,
     solution_generator, solution_objective, branch_list, generator_indices,
     load_indices, shunt_indices) = load_grid_data(args.system_size, args.opfdata_dir)

    # ── 4. Prepare N-1 topology metadata ──────────────────────────────────────
    print("\nPreparing N-1 dataset topology metadata...")
    (samp_grid_bus, samp_grid_generator, samp_grid_shunt,
     samp_grid_ac_line_features, samp_grid_transformer_features,
     generator_indices, load_indices, shunt_indices,
     grid_ac_line_senders, grid_ac_line_receivers,
     grid_transformer_senders, grid_transformer_receivers,
     line_h5_indices, transformer_h5_indices,
     datakit_active_gen_index) = prepare_datasets_n1(
         datakit_edge_index, grid_bus, grid_generator, grid_shunt,
         grid_ac_line_features, grid_transformer_features,
         grid_ac_line_senders, grid_ac_line_receivers,
         grid_transformer_senders, grid_transformer_receivers,
         branch_list, generator_indices, load_indices, shunt_indices,
         full_topo_gen_in_service, full_topo_branch_active
     )

    # ── 5. Build PE cache for all unique contingency topologies ───────────────
    print("\nBuilding contingency PE cache...")
    contingency_pe_cache, scenario_keys = compute_pe_cache(
        h5_file_path,
        line_h5_indices,
        transformer_h5_indices,
        samp_grid_ac_line_features,
        samp_grid_transformer_features,
        grid_ac_line_senders,
        grid_ac_line_receivers,
        grid_transformer_senders,
        grid_transformer_receivers,
        system_size=args.system_size,
        n_scenarios=n_scenarios,
    )

    # Build the flat per-scenario PE list for the converter.
    # Many entries share the same tensor object (generator contingencies reuse
    # the full-topology PE), so memory overhead is minimal.
    scenario_pe_list = [contingency_pe_cache[key] for key in scenario_keys]

    # ── 6. Create train / val / test data loaders ─────────────────────────────
    print("\nCreating data loaders...")
    (train_loader, val_loader, test_loader,
     train_indices, val_indices, test_indices) = create_n1_h5_dataloaders(
        h5_file_path,
        samp_grid_bus,
        samp_grid_generator,
        samp_grid_shunt,
        samp_grid_ac_line_features,
        samp_grid_transformer_features,
        generator_indices,
        load_indices,
        shunt_indices,
        grid_ac_line_senders,
        grid_ac_line_receivers,
        grid_transformer_senders,
        grid_transformer_receivers,
        line_h5_indices,
        transformer_h5_indices,
        datakit_active_gen_index,
        scenario_pe_list,
        batch_size=args.batch_size,
        num_workers=16,
        seed=args.seed,
        data_len=n_scenarios,
    )
    print(f"Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    # ── 7. Initialise model ────────────────────────────────────────────────────
    print("\nInitialising model...")
    model = HeteroInteractGNN(
        hidden_size=args.hidden_size,
        n_mp_layers=args.n_mp_layers,
    ).to(device)
    tot_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {tot_params:,}")

    model_path = os.path.join(
        args.output_dir,
        f'{args.system_size}_bus_HybridHeteroGNN_{args.n_mp_layers}_{args.hidden_size}_PQVT_Datakit.pth'
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate,
        fused=True, weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # ── 8. W&B initialisation ─────────────────────────────────────────────────
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
        wandb.init(
            project="Towards_Generalization_of_GNN_for_ACOPF",
            name=f"Datakit_{args.system_size}_HybridHeteroGNN"
                 f"_{args.n_mp_layers}_{args.hidden_size}_PQVT",
            config={
                "architecture": "HybridHeteroGNN",
                "dataset": "N1_Datakit",
                "system_size": args.system_size,
                "hidden_size": args.hidden_size,
                "n_mp_layers": args.n_mp_layers,
                "batch_size":  args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay":  args.weight_decay,
                "epochs": args.num_epochs,
                "n_unique_contingencies": len(contingency_pe_cache),
            },
        )

    # ── 9. (Optional) Resume from checkpoint ──────────────────────────────────
    best_valid_loss  = float('inf')
    best_model_state = None

    if args.resume:
        print(f"\nResuming from saved model at epoch {args.start_epoch}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        best_model_state = deepcopy(model.state_dict())
        print("Loaded model weights.")

    # ── 10. Training loop ─────────────────────────────────────────────────────
    print(f"\n{'='*80}\nStarting training...\n{'='*80}\n")
    training_losses   = []
    validation_losses = []

    for epoch in tqdm(range(args.start_epoch, args.num_epochs), desc="Training"):
        train_loss = train_model(model, train_loader, optimizer)
        valid_loss = validate_model(model, val_loader)

        training_losses.append(train_loss)
        validation_losses.append(valid_loss)

        if args.wandb_api_key:
            wandb.log({"training_loss": train_loss, "validation_loss": valid_loss})

        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss  = valid_loss
            best_model_state = deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val. Loss : {valid_loss:.6f}")

    # ── 11. Save training plot ─────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses,   'r', label='Train loss')
    plt.plot(validation_losses, 'g', label='Val loss')
    plt.legend();  plt.semilogy();  plt.grid(True, alpha=0.3)
    plt.title(f'N-1 GNN Training — {args.system_size}-bus', fontsize=15)
    plt.xlabel('Epoch');  plt.ylabel('MSE Loss')
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir,
                             f'{args.system_size}_bus_N1_training_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved training plot: {plot_path}")

    # ── 12. Load best model and test ──────────────────────────────────────────
    print("\n" + "="*80)
    print("Evaluating best model on test set...")

#    torch.save(best_model_state, model_path)
#    model.load_state_dict(best_model_state)
    model.eval()

    if args.wandb_api_key:
        wandb.save(model_path)

    (test_loss, v_predictions, v_targets,
     p_predictions, p_targets,
     gen_inputs_list,
     gen_bus_idx_list) = test_model_n1(model, test_loader, args.system_size)

    print(f"Test MSE Loss: {test_loss:.6f}")

    # ── 13. Per-output MSE metrics ─────────────────────────────────────────────
    calc_loss = nn.MSELoss()
    va_loss = calc_loss(v_predictions[:, :, 0], v_targets[:, :, 0])
    vm_loss = calc_loss(v_predictions[:, :, 1], v_targets[:, :, 1])

    # Stack generator predictions for aggregate metrics (variable size; pad to max)
    p_all = torch.cat(p_predictions, dim=0)
    t_all = torch.cat(p_targets,     dim=0)
    pg_loss = calc_loss(p_all[:, 0], t_all[:, 0])
    qg_loss = calc_loss(p_all[:, 1], t_all[:, 1])

    print(f"\nVoltage angle MSE      : {va_loss:.6f}")
    print(f"Voltage magnitude MSE  : {vm_loss:.6f}")
    print(f"Active power MSE       : {pg_loss:.6f}")
    print(f"Reactive power MSE     : {qg_loss:.6f}")

    # ── 14. Load per-scenario H5 data needed for evaluation ───────────────────
    print("\nLoading test scenario data for constraint evaluation...")
    with h5py.File(h5_file_path, 'r') as f:
        sorted_test    = np.sort(test_indices)
        unsort_indices = np.argsort(np.argsort(test_indices))

        bus_data_test = f['bus_data'][sorted_test][unsort_indices]   # (n_test, n_buses, 4)
        gen_data_test = f['gen_data'][sorted_test][unsort_indices]   # (n_test, n_gens,  6)

    # Build per-test-scenario objective (IPOPT solution cost)
    gencosts_test  = gen_data_test[:, datakit_active_gen_index, 2:5]   # (n_test, n_opf_gens, 3)
    sol_pg_test    = gen_data_test[:, datakit_active_gen_index, 0] / 100.0  # p.u.
    test_obj_np = (
        gencosts_test[:, :, 0] * (100 * sol_pg_test)**2 +
        gencosts_test[:, :, 1] * (100 * sol_pg_test) +
        gencosts_test[:, :, 2]
    ).sum(axis=1)

    # Per-test-scenario contingency keys (needed for topology-aware eval)
    test_scenario_keys = [scenario_keys[int(idx)] for idx in test_indices]

    # ── 15. Optimality gap ────────────────────────────────────────────────────
    print("\nComputing optimality gap...")
    opt_gap = compute_optimality_n1(gen_inputs_list, p_predictions, test_obj_np)
    print(f"Optimality gap: {opt_gap:.4f}%")

    # ── 16. Constraint violations ─────────────────────────────────────────────
    print("\nEvaluating constraint violations...")
    test_bus_inputs = torch.tensor(
        np.broadcast_to(samp_grid_bus[np.newaxis],
                        (len(test_indices), *samp_grid_bus.shape)).copy(),
        dtype=torch.float32,
    )

    # test_generator_inputs: we already have gen_inputs_list (per-scenario active gen features)
    constraint_metrics = evaluate_constraints_n1(
        v_predictions,
        p_predictions,
        test_bus_inputs,
        gen_inputs_list,
        gen_bus_idx_list,
        samp_grid_ac_line_features,
        samp_grid_transformer_features,
        samp_grid_shunt,
        branch_list,
        load_indices,
        shunt_indices,
        generator_indices,
        test_scenario_keys,
        bus_data_test,
        line_h5_indices,
        transformer_h5_indices,
        grid_ac_line_senders,
        grid_ac_line_receivers,
        grid_transformer_senders,
        grid_transformer_receivers,
        args.system_size,
    )

    # ── 17. Print all results ─────────────────────────────────────────────────
    print("\nConstraint Violations:")
    print("-" * 80)
    for metric, value in constraint_metrics.items():
        print(f"  {metric:<45}: {value:.6f}")

    # ── 18. Log to W&B ────────────────────────────────────────────────────────
    if args.wandb_api_key:
        tbl = wandb.Table(columns=["metric", "value"])
        tbl.add_data("optimality_gap",       opt_gap.item())
        tbl.add_data("test_loss",            test_loss)
        tbl.add_data("voltage_angle_mse",    va_loss.item())
        tbl.add_data("voltage_magnitude_mse",vm_loss.item())
        tbl.add_data("active_power_mse",     pg_loss.item())
        tbl.add_data("reactive_power_mse",   qg_loss.item())
        for k, v in constraint_metrics.items():
            tbl.add_data(k, v)
        wandb.log({"model_metrics_table": tbl})
        wandb.finish()

    # ── 19. Cleanup ───────────────────────────────────────────────────────────
    del bus_data_test, gen_data_test
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n{'='*80}")
    print(f"N-1 training complete for {args.system_size}-bus system!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
