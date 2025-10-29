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

def initialize_arrays(first_data: Dict[str, Any], num_files: int) -> Dict[str, np.ndarray]:
    """Initialize the numpy arrays based on the first file's structure."""
    combined_data = {}
    
    # Initialize grid arrays
    combined_data['grid_bus'] = np.zeros((num_files, len(first_data['grid']['nodes']['bus']), 
                                        len(first_data['grid']['nodes']['bus'][0])), dtype=np.float32)
    combined_data['grid_generator'] = np.zeros((num_files, len(first_data['grid']['nodes']['generator']), 
                                              len(first_data['grid']['nodes']['generator'][0])), dtype=np.float32)
    combined_data['grid_load'] = np.zeros((num_files, len(first_data['grid']['nodes']['load']), 
                                         len(first_data['grid']['nodes']['load'][0])), dtype=np.float32)
    combined_data['grid_shunt'] = np.zeros((num_files, len(first_data['grid']['nodes']['shunt']), 
                                          len(first_data['grid']['nodes']['shunt'][0])), dtype=np.float32)
    
    combined_data['solution_bus'] = np.zeros((num_files, len(first_data['solution']['nodes']['bus']), 
                                            len(first_data['solution']['nodes']['bus'][0])), dtype=np.float32)
    combined_data['solution_generator'] = np.zeros((num_files, len(first_data['solution']['nodes']['generator']), 
                                                  len(first_data['solution']['nodes']['generator'][0])), dtype=np.float32)
    
    # Initialize edge arrays
    for edge_type in ['ac_line', 'transformer']:
        grid_edge_features = first_data['grid']['edges'][edge_type]['features']
        combined_data[f'grid_{edge_type}_senders'] = np.zeros((num_files, len(first_data['grid']['edges'][edge_type]['senders'])), dtype=np.int32)
        combined_data[f'grid_{edge_type}_receivers'] = np.zeros((num_files, len(first_data['grid']['edges'][edge_type]['receivers'])), dtype=np.int32)
        combined_data[f'grid_{edge_type}_features'] = np.zeros((num_files, len(grid_edge_features), len(grid_edge_features[0])), dtype=np.float32)
        
        solution_edge_features = first_data['solution']['edges'][edge_type]['features']
        combined_data[f'solution_{edge_type}_senders'] = np.zeros((num_files, len(first_data['solution']['edges'][edge_type]['senders'])), dtype=np.int32)
        combined_data[f'solution_{edge_type}_receivers'] = np.zeros((num_files, len(first_data['solution']['edges'][edge_type]['receivers'])), dtype=np.int32)
        combined_data[f'solution_{edge_type}_features'] = np.zeros((num_files, len(solution_edge_features), len(solution_edge_features[0])), dtype=np.float32)
    
    for edge_type in ['generator_link', 'load_link', 'shunt_link']:
        combined_data[f'grid_{edge_type}_senders'] = np.zeros((num_files, len(first_data['grid']['edges'][edge_type]['senders'])), dtype=np.int32)
        combined_data[f'grid_{edge_type}_receivers'] = np.zeros((num_files, len(first_data['grid']['edges'][edge_type]['receivers'])), dtype=np.int32)
    
    combined_data['metadata_objective'] = np.zeros(num_files, dtype=np.float32)
    
    return combined_data

def process_batch(file_paths: List[Path], start_idx: int, arrays: Dict[str, np.ndarray], num_workers: int) -> None:
    """Process a batch of files and update the arrays in-place."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(read_json_file, file_path): idx 
            for idx, file_path in enumerate(file_paths, start_idx)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                data = future.result()
                arrays['grid_bus'][idx] = extract_array_features(data['grid']['nodes']['bus'])
                arrays['grid_generator'][idx] = extract_array_features(data['grid']['nodes']['generator'])
                arrays['grid_load'][idx] = extract_array_features(data['grid']['nodes']['load'])
                arrays['grid_shunt'][idx] = extract_array_features(data['grid']['nodes']['shunt'])
                arrays['solution_bus'][idx] = extract_array_features(data['solution']['nodes']['bus'])
                arrays['solution_generator'][idx] = extract_array_features(data['solution']['nodes']['generator'])
                for edge_type in ['ac_line', 'transformer']:
                    arrays[f'grid_{edge_type}_senders'][idx] = data['grid']['edges'][edge_type]['senders']
                    arrays[f'grid_{edge_type}_receivers'][idx] = data['grid']['edges'][edge_type]['receivers']
                    arrays[f'grid_{edge_type}_features'][idx] = extract_array_features(data['grid']['edges'][edge_type]['features'])
                    arrays[f'solution_{edge_type}_senders'][idx] = data['solution']['edges'][edge_type]['senders']
                    arrays[f'solution_{edge_type}_receivers'][idx] = data['solution']['edges'][edge_type]['receivers']
                    arrays[f'solution_{edge_type}_features'][idx] = extract_array_features(data['solution']['edges'][edge_type]['features'])
                for edge_type in ['generator_link', 'load_link', 'shunt_link']:
                    arrays[f'grid_{edge_type}_senders'][idx] = data['grid']['edges'][edge_type]['senders']
                    arrays[f'grid_{edge_type}_receivers'][idx] = data['grid']['edges'][edge_type]['receivers']
                arrays['metadata_objective'][idx] = data['metadata']['objective']
            except Exception as e:
                print(f"Error processing file {idx}: {e}")

def process_multiple_folders(folders: List[str], batch_size: int = 100, num_workers: int = 4) -> str:
    """Process JSON files from multiple folders and combine them into a single NPZ file."""
    all_file_paths = []
    for folder in folders:
        folder_path = Path(folder)
        all_file_paths.extend(sorted(folder_path.glob("example_*.json"), key=lambda x: int(x.stem.split('_')[1])))

    num_files = len(all_file_paths)
    print(f"Total files to process: {num_files}")

    # Read the first file to initialize arrays
    print("Reading first file to initialize arrays...")
    first_data = read_json_file(all_file_paths[0])
    combined_data = initialize_arrays(first_data, num_files)

    # Process files in batches
    print("Processing files in batches...")
    for batch_start in tqdm(range(0, num_files, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, num_files)
        batch_files = all_file_paths[batch_start:batch_end]
        process_batch(batch_files, batch_start, combined_data, num_workers)

    # Save to NPZ
    output_file = Path(folders[0]).parent / "combined_dataset.npz"
    print("Saving compressed NPZ file...")
    np.savez_compressed(output_file, **combined_data)
    print(f"Dataset saved to: {output_file}")
    return str(output_file)

if __name__ == "__main__":
    # List of dataset folders
    dataset_folders = [f"dataset_release_2000bus_{i}" for i in range(20)]
    output_path = process_multiple_folders(dataset_folders, batch_size=1000, num_workers=4)
