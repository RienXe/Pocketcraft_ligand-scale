import json
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb
import biotite.structure as struc

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_structure(path, model=1):
    if path.endswith('.cif'):
        cif_file = pdbx.CIFFile.read(path)
        atoms = pdbx.get_structure(cif_file, model=model)
    elif path.endswith('.pdb'):
        pdb_file = pdb.PDBFile.read(path)
        atoms = pdb.get_structure(pdb_file, model=model)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return atoms
