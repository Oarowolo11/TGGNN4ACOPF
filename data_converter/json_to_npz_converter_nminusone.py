import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from typing import Dict, List, Any

def read_json_file(file_path: Path) -> Dict[str, Any]:
    """Read a single JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_array_features(data_list: List) -> np.ndarray:
    """Convert list of lists into a numpy array."""
    return np.array(data_list, dtype=np.float32)

def process_single_file(data: Dict) -> Dict[str, np.ndarray]:
    """Process a single file's data into numpy arrays."""
    processed = {}
    
    # Process grid nodes
    for node_type in ['bus', 'generator', 'load', 'shunt']:
        if node_type in data['grid']['nodes']:
            processed[f'grid_{node_type}'] = extract_array_features(data['grid']['nodes'][node_type])
    
    # Process grid edges
    for edge_type in ['ac_line', 'transformer', 'generator_link', 'load_link', 'shunt_link']:
        if edge_type in data['grid']['edges']:
            edge_data = data['grid']['edges'][edge_type]
            processed[f'grid_{edge_type}_senders'] = np.array(edge_data['senders'], dtype=np.int32)
            processed[f'grid_{edge_type}_receivers'] = np.array(edge_data['receivers'], dtype=np.int32)
            if 'features' in edge_data:
                processed[f'grid_{edge_type}_features'] = extract_array_features(edge_data['features'])
    
    # Process solution nodes
    for node_type in ['bus', 'generator']:
        if node_type in data['solution']['nodes']:
            processed[f'solution_{node_type}'] = extract_array_features(data['solution']['nodes'][node_type])
    
    # Process solution edges
    for edge_type in ['ac_line', 'transformer']:
        if edge_type in data['solution']['edges']:
            edge_data = data['solution']['edges'][edge_type]
            processed[f'solution_{edge_type}_senders'] = np.array(edge_data['senders'], dtype=np.int32)
            processed[f'solution_{edge_type}_receivers'] = np.array(edge_data['receivers'], dtype=np.int32)
            if 'features' in edge_data:
                processed[f'solution_{edge_type}_features'] = extract_array_features(edge_data['features'])
    
    # Process metadata
    processed['metadata_objective'] = np.array(data['metadata']['objective'], dtype=np.float32)
    
    return processed

def process_batch(file_paths: List[Path], num_workers: int) -> List[Dict[str, Any]]:
    """Process a batch of files."""
    processed_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(read_json_file, file_path): file_path 
            for file_path in file_paths
        }
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_file),
            total=len(future_to_file),
            desc="Processing batch"
        ):
            file_path = future_to_file[future]
            try:
                data = future.result()
                processed_data.append(process_single_file(data))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    return processed_data

def process_multiple_folders(folders: List[str], batch_size: int = 100, num_workers: int = 4) -> str:
    """Process JSON files from multiple folders and combine them into a single NPZ file."""
    all_file_paths = []
    for folder in folders:
        folder_path = Path(folder)
        all_file_paths.extend(sorted(folder_path.glob("example_*.json"), key=lambda x: int(x.stem.split('_')[1])))

    num_files = len(all_file_paths)
    print(f"Total files to process: {num_files}")

    all_processed_data = []
    for batch_start in tqdm(range(0, num_files, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, num_files)
        batch_files = all_file_paths[batch_start:batch_end]
        batch_data = process_batch(batch_files, num_workers)
        all_processed_data.extend(batch_data)

    # Extract all keys and initialize combined_data
    combined_data = {}
    all_keys = set()
    for data in all_processed_data:
        all_keys.update(data.keys())

    for key in all_keys:
        combined_data[key] = np.array(
            [data.get(key, np.array([])) for data in all_processed_data],
            dtype=object
        )

    output_file = Path(folders[0]).parent / "combined_dataset_nminusone.npz"
    print("Saving compressed NPZ file...")
    np.savez_compressed(output_file, **combined_data)
    print(f"Dataset saved to: {output_file}")
    return str(output_file)

if __name__ == "__main__":
    # List of dataset folders
    dataset_folders = [f"dataset_release_2000bus_nminusone_{i}" for i in range(20)]
    output_path = process_multiple_folders(dataset_folders, batch_size=100, num_workers=4)
