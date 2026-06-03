import numpy as np
import torch
import biotite.structure as struc
import biotite.structure.info as info

def fibonacci_sphere(samples=1):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def generate_surface_points(protein_atoms, num_points_per_atom=20, probe_radius=1.4):
    """
    Generates Solvent Accessible Surface (SAS) points using a SASA-based pre-filter.
    """
    atom_sasa = struc.sasa(protein_atoms, probe_radius=probe_radius)
    surface_atom_mask = atom_sasa > 0
    surface_atoms = protein_atoms[surface_atom_mask]
    
    if len(surface_atoms) == 0:
        return torch.empty((0, 3))

    radii = np.array([info.vdw_radius_single(e) + probe_radius for e in surface_atoms.element])
    atom_coords = surface_atoms.coord
    unit_sphere_pts = fibonacci_sphere(num_points_per_atom)
    
    all_pts = atom_coords[:, np.newaxis, :] + unit_sphere_pts[np.newaxis, :, :] * radii[:, np.newaxis, np.newaxis]
    all_pts = all_pts.reshape(-1, 3)
    
    all_radii = np.array([info.vdw_radius_single(e) + probe_radius for e in protein_atoms.element])
    all_coords = protein_atoms.coord
    cell_list = struc.CellList(protein_atoms, cell_size=np.max(all_radii))
    
    keep_mask = np.ones(len(all_pts), dtype=bool)
    for i, p in enumerate(all_pts):
        neighbor_indices = cell_list.get_atoms(p, radius=np.max(all_radii))
        dists_sq = np.sum((p - all_coords[neighbor_indices])**2, axis=1)
        if np.any(dists_sq < (all_radii[neighbor_indices] - 0.01)**2):
            keep_mask[i] = False
            
    return torch.tensor(all_pts[keep_mask]).float()
