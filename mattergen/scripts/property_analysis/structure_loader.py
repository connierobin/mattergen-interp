import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pickle

import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter


def load_structures_from_directory(directory_path: Path, 
                                 file_pattern: str = "*.cif") -> List[Tuple[Structure, str]]:
    """Load structures from CIF files in a directory."""
    structures = []
    directory = Path(directory_path)
    
    for file_path in directory.glob(file_pattern):
        try:
            structure = Structure.from_file(file_path)
            structure_id = file_path.stem
            structures.append((structure, structure_id))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return structures


def load_structures_from_pickle(pickle_path: Path) -> List[Tuple[Structure, str]]:
    """Load structures from a pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    structures = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, Structure):
                structures.append((item, f"structure_{i}"))
            elif isinstance(item, dict) and 'structure' in item:
                structure_id = item.get('id', f"structure_{i}")
                structures.append((item['structure'], structure_id))
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, Structure):
                structures.append((value, str(key)))
    
    return structures


def load_structures_from_json(json_path: Path) -> List[Tuple[Structure, str]]:
    """Load structures from JSON file with structure data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    structures = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'structure' in item:
                try:
                    structure = Structure.from_dict(item['structure'])
                    structure_id = item.get('id', f"structure_{i}")
                    structures.append((structure, structure_id))
                except Exception as e:
                    print(f"Error loading structure {i}: {e}")
    
    return structures


def save_structures_as_cifs(structures: List[Tuple[Structure, str]], 
                           output_dir: Path) -> None:
    """Save structures as individual CIF files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for structure, structure_id in structures:
        cif_path = output_dir / f"{structure_id}.cif"
        cif_writer = CifWriter(structure)
        cif_writer.write_file(cif_path)


def discover_structure_files(base_directory: Path) -> Dict[str, List[Path]]:
    """Discover structure files in a directory and categorize by type."""
    base_dir = Path(base_directory)
    
    file_types = {
        'cif': list(base_dir.rglob("*.cif")),
        'json': list(base_dir.rglob("*.json")),
        'pickle': list(base_dir.rglob("*.pkl")) + list(base_dir.rglob("*.pickle")),
        'vasp': list(base_dir.rglob("POSCAR*")) + list(base_dir.rglob("CONTCAR*")),
    }
    
    return {k: v for k, v in file_types.items() if v}


def load_structures_auto(file_path: Path) -> List[Tuple[Structure, str]]:
    """Automatically detect file type and load structures."""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.cif':
        return [(Structure.from_file(file_path), file_path.stem)]
    elif file_path.suffix.lower() == '.json':
        return load_structures_from_json(file_path)
    elif file_path.suffix.lower() in ['.pkl', '.pickle']:
        return load_structures_from_pickle(file_path)
    elif file_path.name.startswith(('POSCAR', 'CONTCAR')):
        return [(Structure.from_file(file_path), file_path.stem)]
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def create_structure_summary(structures: List[Tuple[Structure, str]]) -> pd.DataFrame:
    """Create a summary DataFrame of loaded structures."""
    summary_data = []
    
    for structure, structure_id in structures:
        summary_data.append({
            'structure_id': structure_id,
            'formula': structure.composition.reduced_formula,
            'num_atoms': len(structure),
            'volume': structure.volume,
            'density': structure.density,
            'chemical_system': structure.composition.chemical_system,
            'lattice_a': structure.lattice.a,
            'lattice_b': structure.lattice.b,
            'lattice_c': structure.lattice.c,
            'lattice_alpha': structure.lattice.alpha,
            'lattice_beta': structure.lattice.beta,
            'lattice_gamma': structure.lattice.gamma,
        })
    
    return pd.DataFrame(summary_data)