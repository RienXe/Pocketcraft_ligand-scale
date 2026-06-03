"""
Data Preprocessing Pipeline for Protein Surface Point Clouds.

This script converts protein structure files (.cif or .pdb) into PyTorch Geometric 
compatible tensors (.pt). It generates Solvent Accessible Surface (SAS) points, 
extracts localized spatial features, and labels points based on their proximity 
to ligands.

Key functionalities:
- SAS point generation using Fibonacci sphere sampling.
- Multi-threaded processing for large datasets.
- Targeted labeling by ligand size class.
"""

import os
import argparse
import torch
import numpy as np
import biotite.structure as struc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from utils.io import load_config, load_structure
from utils.geometry import generate_surface_points
from utils.features import get_weighted_features

# 1. Default Configurations
FEATURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'protein_feature_defaultv2.json')
DEFAULT_LIGAND_SIZE_CONFIG = os.path.join(os.path.dirname(__file__), 'ccd_size_class.json')

def process_file(filename, input_dir, output_dir, feat_config, points_per_atom, threshold, ligand_dir, probe_radius, allowed_ligands=None):
    """
    Processes a single structure file: generates surface points, extracts features, 
    and saves the resulting data dictionary as a .pt file.
    
    Args:
        filename (str): Name of the file to process.
        input_dir (str): Directory containing input structures.
        output_dir (str): Directory to save processed tensors.
        feat_config (dict): Feature configuration dictionary.
        points_per_atom (int): Number of surface points per atom.
        threshold (float): Distance threshold for pocket labeling.
        ligand_dir (str): Optional path to separate ligand files.
        probe_radius (float): SAS probe radius.
        allowed_ligands (set): Set of residue names to consider for labeling.
    """
    input_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, base_name + '.pt')
    
    if os.path.exists(output_path):
        return None # Skip

    try:
        atoms = load_structure(input_path)
        protein_atoms = atoms[struc.filter_amino_acids(atoms)]
        
        surface_coords = generate_surface_points(protein_atoms, points_per_atom, probe_radius)
        atom_coords_tensor = torch.tensor(protein_atoms.coord).float()
        
        if len(surface_coords) > 0:
            cloud_features = get_weighted_features(surface_coords, atom_coords_tensor, protein_atoms, feat_config)
        else:
            cloud_features = torch.empty((0, 11 + len(feat_config['regular_aa'])))
        
        ligand_atoms = None
        if ligand_dir:
            ligand_path = os.path.join(ligand_dir, filename)
            if os.path.exists(ligand_path):
                ligand_atoms_all = load_structure(ligand_path)
                ligand_atoms = ligand_atoms_all[~struc.filter_solvent(ligand_atoms_all)]
        else:
            ligand_mask = atoms.hetero & ~struc.filter_solvent(atoms) & ~struc.filter_amino_acids(atoms)
            ligand_atoms = atoms[ligand_mask]
        
        # Filter by ligand size if specified
        if ligand_atoms is not None and len(ligand_atoms) > 0 and allowed_ligands is not None:
            mask = np.isin(ligand_atoms.res_name, list(allowed_ligands))
            ligand_atoms = ligand_atoms[mask]

        if ligand_atoms is not None and len(ligand_atoms) > 0:
            ligand_coords = torch.tensor(ligand_atoms.coord).float()
            if len(surface_coords) > 0:
                dist_surf = torch.cdist(surface_coords, ligand_coords)
                y = (dist_surf.min(dim=1)[0] <= threshold).float()
            else:
                y = torch.empty(0)
            dist_atom = torch.cdist(atom_coords_tensor, ligand_coords)
            y_atom = (dist_atom.min(dim=1)[0] <= threshold).float()
        else:
            y = torch.zeros(len(surface_coords))
            y_atom = torch.zeros(len(protein_atoms))

        data = {
            'x': cloud_features,
            'pos': surface_coords,
            'y': y,
            'atom_pos': atom_coords_tensor,
            'atom_y': y_atom,
        }
        torch.save(data, output_path)
        return True
    except Exception as e:
        return f"{filename}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Parallel generation of protein surface point clouds from structure files for ML model training.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input protein structure files (.pdb or .cif).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated PyTorch Geometric surface point cloud tensors (.pt).")
    parser.add_argument("--config", type=str, default=FEATURE_CONFIG_PATH, help="Path to the JSON configuration file defining atom and residue level features.")
    parser.add_argument("--points_per_atom", type=int, default=20, help="Number of points to sample per atom during Fibonacci sphere surface generation.")
    parser.add_argument("--threshold", type=float, default=4.5, help="Distance threshold (in Angstroms) between a surface point and a ligand to label it as a pocket.")
    parser.add_argument("--ligand_dir", type=str, help="Optional directory containing separate isolated ligand structure files for proximity labeling.")
    parser.add_argument("--probe_radius", type=float, default=1.4, help="Radius of the water probe sphere (in Angstroms) for Solvent Accessible Surface generation.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel worker processes to use for point cloud generation.")
    parser.add_argument("--ligand_size", type=str, choices=["small_ligands", "medium_ligands", "large_ligands"], help="Filter targets by specific ligand size classes during data generation.")
    parser.add_argument("--ligand_size_config", type=str, default=DEFAULT_LIGAND_SIZE_CONFIG, help="Path to the JSON configuration defining the valid residues for each ligand size class.")
    
    args = parser.parse_args()
    feat_config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    
    allowed_ligands = None
    if args.ligand_size:
        size_config = load_config(args.ligand_size_config)
        allowed_ligands = set(size_config[args.ligand_size])
        print(f"Filtering ligands by size: {args.ligand_size} ({len(allowed_ligands)} allowed residue names)")

    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.cif', '.pdb'))]
    
    process_func = partial(
        process_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        feat_config=feat_config,
        points_per_atom=args.points_per_atom,
        threshold=args.threshold,
        ligand_dir=args.ligand_dir,
        probe_radius=args.probe_radius,
        allowed_ligands=allowed_ligands
    )

    print(f"Processing {len(input_files)} files using {args.num_workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(process_func, input_files), total=len(input_files)))

    skipped = results.count(None)
    errors = [r for r in results if isinstance(r, str)]
    success = results.count(True)
    
    print(f"\nProcessing complete:")
    print(f"  - Successfully processed: {success}")
    print(f"  - Skipped: {skipped}")
    print(f"  - Errors: {len(errors)}")
    
    if errors:
        print("\nError Summary (first 10):")
        for err in errors[:10]:
            print(f"  {err}")

if __name__ == "__main__":
    main()
