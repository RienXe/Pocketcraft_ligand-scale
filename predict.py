"""
Inference and Pocket Mapping Pipeline.

This script performs inference using trained models to predict binding pockets 
on protein structures. It maps the point-wise predictions from surface points 
back to the original protein residues for downstream analysis.

Outputs:
- CSV file: List of residues with their maximum pocket confidence scores.
- TXT file: List of predicted residue IDs grouped by chain for easy visualization.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import biotite.structure as struc
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from utils.io import load_config, load_structure
from utils.geometry import generate_surface_points
from utils.features import get_weighted_features
from models.gnn import GNNModel
from models.pointnet2 import PointNet2Segmenter

# Default Configuration Path
FEATURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'protein_feature_defaultv2.json')

def process_single_file(input_path, output_path, model, device, feat_config, args):
    """
    Processes a single structure file, performs inference, and saves results.
    
    Args:
        input_path (str): Path to input PDB/CIF file.
        output_path (str): Base path for output CSV and TXT files.
        model (nn.Module): The loaded prediction model.
        device (torch.device): Inference device.
        feat_config (dict): Feature configuration.
        args: Parsed command-line arguments.
    """
    try:
        atoms = load_structure(input_path)
        protein_atoms = atoms[struc.filter_amino_acids(atoms)]
        
        if len(protein_atoms) == 0:
            return f"No protein atoms found in {input_path}"

        surface_coords = generate_surface_points(protein_atoms, args.points_per_atom, args.probe_radius)
        
        if len(surface_coords) == 0:
            return f"No surface points generated for {input_path}"
            
        atom_coords_tensor = torch.tensor(protein_atoms.coord).float()
        cloud_features = get_weighted_features(surface_coords, atom_coords_tensor, protein_atoms, feat_config)
        
        data = Data(x=cloud_features, pos=surface_coords)
        batch = Batch.from_data_list([data]).to(device)
        
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        predicted_mask = probs > args.threshold
        predicted_coords = surface_coords[predicted_mask]
        
        if len(predicted_coords) == 0:
            pd.DataFrame(columns=['chain_id', 'res_id', 'res_name', 'score']).to_csv(output_path, index=False)
            return True

        seen_residues = {}

        if args.mapping_method == "radius":
            dist_to_atoms = torch.cdist(atom_coords_tensor, predicted_coords)
            min_dist, point_idx = dist_to_atoms.min(dim=1)
            within_radius_mask = min_dist <= args.radius
            
            for atom_idx in torch.where(within_radius_mask)[0]:
                atom = protein_atoms[atom_idx.item()]
                res_key = (atom.chain_id, atom.res_id, atom.res_name)
                score = probs[predicted_mask][point_idx[atom_idx].item()]
                
                if res_key not in seen_residues or score > seen_residues[res_key]['score']:
                    seen_residues[res_key] = {
                        'chain_id': atom.chain_id,
                        'res_id': atom.res_id,
                        'res_name': atom.res_name,
                        'score': float(score)
                    }
        else:
            dist_to_atoms = torch.cdist(predicted_coords, atom_coords_tensor)
            nearest_atom_indices = dist_to_atoms.argmin(dim=1).numpy()
            
            for i, atom_idx in enumerate(nearest_atom_indices):
                atom = protein_atoms[atom_idx]
                res_key = (atom.chain_id, atom.res_id, atom.res_name)
                score = probs[predicted_mask][i]
                
                if res_key not in seen_residues or score > seen_residues[res_key]['score']:
                    seen_residues[res_key] = {
                        'chain_id': atom.chain_id,
                        'res_id': atom.res_id,
                        'res_name': atom.res_name,
                        'score': float(score)
                    }

        results = list(seen_residues.values())
        df = pd.DataFrame(results)
        df = df.sort_values(by=['chain_id', 'res_id'])
        df.to_csv(output_path, index=False)
        
        # Output txt of residue_ids 
        dict_residues = {}
        with open(output_path.replace('.csv', '_residues.txt'), 'w') as f:
            for res in results:
                chain_id = res['chain_id']
                if chain_id not in dict_residues:
                    dict_residues[chain_id] = []
                dict_residues[chain_id].append(str(res['res_id']))
                
            for chain_id in dict_residues:
                f.write(f"{chain_id}:")
                str_res_ids = ",".join(dict_residues[chain_id])
                f.write(str_res_ids)
                f.write("\n")
        return True
    except Exception as e:
        return f"Error processing {input_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Predict binding pocket residues on protein structures using trained models.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input protein structure files (.pdb or .cif).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights file (.pth).")
    parser.add_argument("--model_type", choices=["dgcnn", "pointnet", "pointnet2"], default="pointnet2", help="Neural network architecture to use for predictions.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the output prediction files (CSV and TXT) will be saved.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence score threshold for identifying a point as part of a binding pocket.")
    parser.add_argument("--mapping_method", choices=["radius", "closest"], default="radius", help="Method to map predicted surface points back to protein residues ('radius' or 'closest').")
    parser.add_argument("--radius", type=float, default=4.5, help="Distance radius for mapping surface points to residues when using the 'radius' mapping method.")
    parser.add_argument("--config", type=str, default=FEATURE_CONFIG_PATH, help="Path to the JSON configuration file defining atomic and structural features.")
    parser.add_argument("--points_per_atom", type=int, default=20, help="Number of points to sample per atom when generating the Solvent Accessible Surface (SAS).")
    parser.add_argument("--probe_radius", type=float, default=1.4, help="Radius of the probe sphere used to generate the Solvent Accessible Surface (in Angstroms).")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimensionality of the hidden layers in the prediction model.")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors to consider in graph-based models (e.g., DGCNN).")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    feat_config = load_config(args.config)
    in_channels = 11 + len(feat_config['regular_aa'])
    
    if args.model_type == "pointnet2":
        model = PointNet2Segmenter(in_channels, 1, args.hidden_dim).to(device)
    else:
        model = GNNModel(args.model_type, in_channels=in_channels, hidden_dim=args.hidden_dim, k=args.k).to(device)
    
    print(f"Loading model weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.cif', '.pdb'))]
    print(f"Found {len(input_files)} structure files.")

    errors = []
    for filename in tqdm(input_files, desc="Processing files"):
        input_path = os.path.join(args.input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output_dir, base_name + '.csv')
        
        result = process_single_file(input_path, output_path, model, device, feat_config, args)
        if isinstance(result, str):
            errors.append(result)

    print(f"\nProcessing complete. Results saved to {args.output_dir}")
    if errors:
        for err in errors[:10]:
            print(f"  {err}")

if __name__ == "__main__":
    main()
