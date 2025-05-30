#!/usr/bin/env python3
"""
Debug script to verify that late insertion trajectories start from the correct point.

This mini analysis compares:
1. The structure at timestep 800 from the original trajectory
2. The first structure in the late insertion trajectory 
3. Verifies they are identical
"""

import zipfile
import io
import numpy as np
from pathlib import Path

import ase
import ase.io
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def load_original_at_timestep_800():
    """Load the structure at timestep 800 from original trajectory."""
    original_path = "generated_structures/three_property_study/unconditional/generated_trajectories.zip"
    
    with zipfile.ZipFile(original_path, 'r') as zf:
        with zf.open('gen_0.extxyz') as f:
            content = io.StringIO(f.read().decode())
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            
            # Timestep 800 = index 1600 (2 structures per timestep)
            atoms = atoms_list[1600]
            structure = AseAtomsAdaptor.get_structure(atoms)
            
            return structure


def load_late_insertion_first_structure():
    """Load the first structure from the late insertion trajectory."""
    late_insertion_path = "generated_structures/late_insertion_trajectories_debug/energy_above_hull_0.0_debug/generated_trajectories.zip"
    
    if not Path(late_insertion_path).exists():
        print(f"Error: Late insertion trajectory not found: {late_insertion_path}")
        return None
    
    with zipfile.ZipFile(late_insertion_path, 'r') as zf:
        with zf.open('gen_0.extxyz') as f:
            content = io.StringIO(f.read().decode())
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            
            # First structure (index 0)
            atoms = atoms_list[0]
            structure = AseAtomsAdaptor.get_structure(atoms)
            
            return structure


def compare_structures(struct1: Structure, struct2: Structure, name1: str, name2: str):
    """Compare two structures in detail."""
    print(f"\n{'='*60}")
    print(f"STRUCTURE COMPARISON: {name1} vs {name2}")
    print(f"{'='*60}")
    
    # Basic properties
    print(f"\n{name1}:")
    print(f"  Composition: {struct1.composition}")
    print(f"  Formula: {struct1.formula}")
    print(f"  Volume: {struct1.lattice.volume:.6f}")
    print(f"  Density: {struct1.density:.6f}")
    print(f"  Num atoms: {len(struct1)}")
    
    print(f"\n{name2}:")
    print(f"  Composition: {struct2.composition}")
    print(f"  Formula: {struct2.formula}")
    print(f"  Volume: {struct2.lattice.volume:.6f}")
    print(f"  Density: {struct2.density:.6f}")
    print(f"  Num atoms: {len(struct2)}")
    
    # Lattice parameters
    params1 = struct1.lattice.parameters
    params2 = struct2.lattice.parameters
    
    print(f"\nLattice Parameters:")
    print(f"  {name1}: a={params1[0]:.6f}, b={params1[1]:.6f}, c={params1[2]:.6f}")
    print(f"            α={params1[3]:.6f}, β={params1[4]:.6f}, γ={params1[5]:.6f}")
    print(f"  {name2}: a={params2[0]:.6f}, b={params2[1]:.6f}, c={params2[2]:.6f}")
    print(f"            α={params2[3]:.6f}, β={params2[4]:.6f}, γ={params2[5]:.6f}")
    
    # Differences
    print(f"\nDifferences:")
    print(f"  Composition match: {struct1.composition == struct2.composition}")
    print(f"  Volume difference: {abs(struct1.lattice.volume - struct2.lattice.volume):.8f}")
    print(f"  Density difference: {abs(struct1.density - struct2.density):.8f}")
    
    # Lattice parameter differences
    param_diffs = [abs(p1 - p2) for p1, p2 in zip(params1, params2)]
    param_names = ['a', 'b', 'c', 'α', 'β', 'γ']
    
    print(f"\n  Lattice parameter differences:")
    for name, diff in zip(param_names, param_diffs):
        print(f"    {name}: {diff:.8f}")
    
    print(f"  Max lattice difference: {max(param_diffs):.8f}")
    
    # Atomic positions (if same composition)
    if struct1.composition == struct2.composition:
        print(f"\nAtomic positions:")
        
        # Compare fractional coordinates
        frac_coords1 = struct1.frac_coords
        frac_coords2 = struct2.frac_coords
        
        if len(frac_coords1) == len(frac_coords2):
            coord_diffs = np.abs(frac_coords1 - frac_coords2)
            max_coord_diff = np.max(coord_diffs)
            avg_coord_diff = np.mean(coord_diffs)
            
            print(f"  Max coordinate difference: {max_coord_diff:.8f}")
            print(f"  Avg coordinate difference: {avg_coord_diff:.8f}")
            
            # Check if structures are essentially identical
            tolerance = 1e-6
            structures_identical = (
                max(param_diffs) < tolerance and 
                max_coord_diff < tolerance and
                struct1.composition == struct2.composition
            )
            
            print(f"\n  Structures identical (tolerance {tolerance}): {structures_identical}")
        else:
            print(f"  Different number of atoms: {len(frac_coords1)} vs {len(frac_coords2)}")
    
    return {
        'composition_match': struct1.composition == struct2.composition,
        'volume_diff': abs(struct1.lattice.volume - struct2.lattice.volume),
        'density_diff': abs(struct1.density - struct2.density),
        'max_lattice_diff': max(param_diffs),
        'structures_identical': max(param_diffs) < 1e-6 and struct1.composition == struct2.composition
    }


def load_saved_starting_structure():
    """Load the saved starting structure from debug run."""
    saved_path = "generated_structures/late_insertion_trajectories_debug/energy_above_hull_0.0_debug/start_structure_intermediate.extxyz"
    
    if not Path(saved_path).exists():
        print(f"Error: Saved starting structure not found: {saved_path}")
        return None
    
    atoms = ase.io.read(saved_path)
    structure = AseAtomsAdaptor.get_structure(atoms)
    
    return structure


def main():
    """Main debug function."""
    print("="*60)
    print("DEBUG: Late Insertion Trajectory Starting Point Verification")
    print("="*60)
    
    # Load structures
    print("\nLoading structures...")
    
    original_800 = load_original_at_timestep_800()
    print(f"✓ Loaded original trajectory at timestep 800")
    
    saved_start = load_saved_starting_structure()
    if saved_start:
        print(f"✓ Loaded saved starting structure")
    else:
        print(f"✗ Could not load saved starting structure")
        return
    
    late_insertion_first = load_late_insertion_first_structure()
    if late_insertion_first:
        print(f"✓ Loaded first structure from late insertion trajectory")
    else:
        print(f"✗ Could not load late insertion trajectory")
        return
    
    # Compare original vs saved starting structure
    result1 = compare_structures(
        original_800, saved_start,
        "Original @ timestep 800", "Saved starting structure"
    )
    
    # Compare saved starting structure vs late insertion first
    result2 = compare_structures(
        saved_start, late_insertion_first,
        "Saved starting structure", "Late insertion first step"
    )
    
    # Compare original vs late insertion first (should be identical)
    result3 = compare_structures(
        original_800, late_insertion_first,
        "Original @ timestep 800", "Late insertion first step"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n1. Original vs Saved Starting Structure:")
    print(f"   Identical: {result1['structures_identical']}")
    print(f"   Max difference: {result1['max_lattice_diff']:.8f}")
    
    print(f"\n2. Saved Starting vs Late Insertion First:")
    print(f"   Identical: {result2['structures_identical']}")
    print(f"   Max difference: {result2['max_lattice_diff']:.8f}")
    
    print(f"\n3. Original vs Late Insertion First (KEY TEST):")
    print(f"   Identical: {result3['structures_identical']}")
    print(f"   Max difference: {result3['max_lattice_diff']:.8f}")
    
    if result3['structures_identical']:
        print(f"\n🎉 SUCCESS: Late insertion trajectory starts from the correct point!")
        print(f"   The fix worked - trajectories now share the same starting structure.")
    else:
        print(f"\n❌ FAILURE: Late insertion trajectory does NOT start from the correct point!")
        print(f"   There may still be a bug in the loading logic.")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()