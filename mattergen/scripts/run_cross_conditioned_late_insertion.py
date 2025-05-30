#!/usr/bin/env python3
"""
Run cross-conditioned late insertion trajectories.

This script takes structures from late timepoints in conditioned trajectories
and continues the diffusion process with different conditioning properties.
For example, take a structure that was generated as a conductor and continue
with insulator conditioning.

The script is designed for parallel execution on clusters with clear file organization.
"""

import os
import io
import torch
import zipfile
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import argparse
import numpy as np
import ase
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.generator import CrystalGenerator
from mattergen.common.utils.data_classes import PRETRAINED_MODEL_NAME
from typing import cast


def structure_to_chemgraph(structure: Structure, device: torch.device) -> ChemGraph:
    """Convert a pymatgen Structure to a ChemGraph."""
    # Get atomic numbers
    atomic_numbers = torch.tensor([site.specie.Z for site in structure], 
                                dtype=torch.long, device=device)
    
    # Get fractional coordinates
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32, device=device)
    
    # Get lattice matrix
    lattice_matrix = torch.tensor(structure.lattice.matrix, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Create batch indices
    num_atoms = len(structure)
    batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
    ptr = torch.tensor([0, num_atoms], dtype=torch.long, device=device)
    
    # Create ChemGraph
    return ChemGraph(
        atomic_numbers=atomic_numbers,
        pos=frac_coords,
        cell=lattice_matrix,
        num_atoms=torch.tensor([num_atoms], dtype=torch.long, device=device),
        num_nodes=torch.tensor(num_atoms, dtype=torch.long, device=device),
        batch=batch,
        ptr=ptr,
        _batch_idx={
            "atomic_numbers": batch, 
            "pos": batch, 
            "cell": torch.zeros(1, dtype=torch.long, device=device)
        }
    )


def load_structure_from_trajectory(zip_path: str, trajectory_idx: int = 0, 
                                 timestep: int = 800, device: torch.device = None) -> Tuple[ChemGraph, Structure]:
    """Load a specific structure from a trajectory at a given timestep."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List all trajectory files
        trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
        
        if not trajectory_files:
            raise ValueError(f"No trajectory files found in {zip_path}")
        
        if trajectory_idx >= len(trajectory_files):
            raise ValueError(f"Trajectory index {trajectory_idx} not available. Found {len(trajectory_files)} trajectories.")
        
        chosen_file = trajectory_files[trajectory_idx]
        print(f"Loading trajectory: {chosen_file}")
        
        with zf.open(chosen_file) as f:
            content = io.StringIO(f.read().decode())
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            
            # Convert timestep to trajectory index (2 structures per timestep: predictor + corrector)
            trajectory_index = timestep * 2
            
            if trajectory_index >= len(atoms_list):
                raise ValueError(f"Timestep {timestep} (index {trajectory_index}) not available. Trajectory has {len(atoms_list)} steps.")
            
            # Get the structure at the specified timestep
            atoms = atoms_list[trajectory_index]
            print(f"Loading structure from trajectory index {trajectory_index} (timestep {timestep})")
            structure = AseAtomsAdaptor.get_structure(atoms)
            
            # Convert to ChemGraph
            chemgraph = structure_to_chemgraph(structure, device)
            
            return chemgraph, structure


def get_successful_conditioning_strategies() -> Dict[str, Dict[str, Any]]:
    """
    Get all successful conditioning strategies from the conditioned trajectories run.
    Based on job_39633.out results, these are the strategies that worked.
    """
    
    strategies = {
        # Space group conditioning
        "space_group_cubic_fcc": {
            "checkpoint_path": "space_group",
            "conditions": {"space_group": 225},  # Fm-3m (FCC)
            "description": "Space group 225 (Fm-3m, FCC cubic)",
            "property_type": "space_group"
        },
        "space_group_cubic_primitive": {
            "checkpoint_path": "space_group",
            "conditions": {"space_group": 221},  # Pm-3m
            "description": "Space group 221 (Pm-3m, primitive cubic)",
            "property_type": "space_group"
        },
        "space_group_hexagonal": {
            "checkpoint_path": "space_group",
            "conditions": {"space_group": 194},  # P63/mmc
            "description": "Space group 194 (P63/mmc, hexagonal)",
            "property_type": "space_group"
        },
        "space_group_tetragonal": {
            "checkpoint_path": "space_group",
            "conditions": {"space_group": 139},  # I4/mmm
            "description": "Space group 139 (I4/mmm, tetragonal)",
            "property_type": "space_group"
        },
        "space_group_orthorhombic": {
            "checkpoint_path": "space_group",
            "conditions": {"space_group": 62},   # Pnma
            "description": "Space group 62 (Pnma, orthorhombic)",
            "property_type": "space_group"
        },
        
        # Energy above hull conditioning
        "energy_above_hull_stable": {
            "checkpoint_path": "chemical_system_energy_above_hull",
            "conditions": {"energy_above_hull": 0.0},  # Perfectly stable
            "description": "Perfectly stable (0.0 eV/atom above hull)",
            "property_type": "energy_above_hull"
        },
        "energy_above_hull_metastable": {
            "checkpoint_path": "chemical_system_energy_above_hull",
            "conditions": {"energy_above_hull": 0.1},  # Stability threshold
            "description": "Marginally stable (0.1 eV/atom above hull)",
            "property_type": "energy_above_hull"
        },
        "energy_above_hull_unstable": {
            "checkpoint_path": "chemical_system_energy_above_hull",
            "conditions": {"energy_above_hull": 0.3},  # Unstable
            "description": "Unstable (0.3 eV/atom above hull)",
            "property_type": "energy_above_hull"
        },
        
        # DFT band gap conditioning
        "dft_band_gap_metallic": {
            "checkpoint_path": "dft_band_gap",
            "conditions": {"dft_band_gap": 0.0},  # Metallic (zero band gap)
            "description": "Metallic (0.0 eV band gap)",
            "property_type": "dft_band_gap"
        },
        "dft_band_gap_narrow_semiconductor": {
            "checkpoint_path": "dft_band_gap",
            "conditions": {"dft_band_gap": 1.0},  # Narrow gap semiconductor
            "description": "Narrow gap semiconductor (1.0 eV band gap)",
            "property_type": "dft_band_gap"
        },
        "dft_band_gap_wide_semiconductor": {
            "checkpoint_path": "dft_band_gap",
            "conditions": {"dft_band_gap": 3.0},  # Wide gap semiconductor
            "description": "Wide gap semiconductor (3.0 eV band gap)",
            "property_type": "dft_band_gap"
        },
        "dft_band_gap_insulator": {
            "checkpoint_path": "dft_band_gap",
            "conditions": {"dft_band_gap": 6.0},  # Wide band gap insulator
            "description": "Wide band gap insulator (6.0 eV band gap)",
            "property_type": "dft_band_gap"
        },
    }
    
    return strategies


def generate_cross_conditioning_pairs(strategies: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Generate all pairs of (source_condition, target_condition) for cross-conditioning.
    This creates combinations where we take a structure conditioned on one property
    and continue with conditioning on a different property value.
    """
    pairs = []
    strategy_names = list(strategies.keys())
    
    for source in strategy_names:
        for target in strategy_names:
            if source != target:  # Don't pair a condition with itself
                pairs.append((source, target))
    
    return pairs


def run_late_insertion_trajectory(
    chemgraph: ChemGraph,
    target_strategy: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    start_timestep: int = 800,
    total_timesteps: int = 1000,
    source_condition_name: str = "unknown"
) -> List[Structure]:
    """Run a single late insertion trajectory with cross-conditioning."""
    
    # Calculate time parameters for remaining steps
    dt = -0.001  # Standard step size
    eps_t = 0.001  # End time
    start_t = 1.0 + start_timestep * dt  # Time at given timestep
    N = total_timesteps - start_timestep  # Number of remaining steps
    
    time_config = {
        "start_t": start_t,
        "eps_t": eps_t, 
        "N": N,
        "dt": dt
    }
    
    print(f"Running cross-conditioning: {source_condition_name} -> {target_strategy['description']}")
    print(f"Target conditions: {target_strategy['conditions']}")
    print(f"Time config: start_t={start_t:.3f}, eps_t={eps_t:.3f}, N={N}")
    
    # Create checkpoint info using Hugging Face Hub
    checkpoint_path = target_strategy["checkpoint_path"]
    model_name_mapping = {
        "mattergen_base": "mattergen_base",
        "space_group": "space_group",
        "chemical_system_energy_above_hull": "chemical_system_energy_above_hull",
        "dft_band_gap": "dft_band_gap"
    }
    
    if checkpoint_path in model_name_mapping:
        hf_model_name = model_name_mapping[checkpoint_path]
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(cast(PRETRAINED_MODEL_NAME, hf_model_name))
        print(f"Using Hugging Face model: {hf_model_name}")
    else:
        raise ValueError(f"Unknown checkpoint path: {checkpoint_path}")
    
    # Convert ChemGraph to BatchedData
    start_structure = collate([chemgraph])
    start_structure = start_structure.to(device)
    
    # Get number of atoms for generator
    num_atoms_int = start_structure.num_atoms[0].item()
    
    # Initialize generator with conditioning
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        properties_to_condition_on=target_strategy["conditions"],
        batch_size=1,
        num_batches=1,
        num_atoms_distribution=f"JUST_{num_atoms_int}",
        record_trajectories=True
    )
    
    # Create output directory for this specific run
    run_output_dir = output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration
    with open(run_output_dir / "run_config.json", "w") as f:
        json.dump({
            "source_condition": source_condition_name,
            "target_strategy": target_strategy,
            "time_config": time_config,
            "start_timestep": start_timestep,
            "total_timesteps": total_timesteps,
            "num_atoms": num_atoms_int
        }, f, indent=2)
    
    try:
        # Generate structures starting from the late timepoint
        generated_structures = generator.generate(
            output_dir=str(run_output_dir),
            time_config=time_config,
            start_structure=start_structure
        )
        
        print(f"✓ Successfully generated {len(generated_structures)} structures")
        return generated_structures
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        
        # Save error details
        with open(run_output_dir / "error.txt", "w") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Source condition: {source_condition_name}\n")
            f.write(f"Target strategy: {target_strategy}\n")
            f.write(f"Time config: {time_config}\n")
        
        return []


def main():
    """Main function for running cross-conditioned late insertion trajectories."""
    
    parser = argparse.ArgumentParser(description="Run cross-conditioned late insertion trajectories")
    parser.add_argument("--source_condition", type=str, required=True,
                       help="Source conditioning strategy name (e.g., 'dft_band_gap_metallic')")
    parser.add_argument("--target_condition", type=str, required=True,
                       help="Target conditioning strategy name (e.g., 'dft_band_gap_insulator')")
    parser.add_argument("--start_timestep", type=int, required=True,
                       help="Timestep to start late insertion from (e.g., 600, 800, 900, 950)")
    parser.add_argument("--trajectory_idx", type=int, default=0,
                       help="Which trajectory to use from the source condition (default: 0)")
    parser.add_argument("--total_timesteps", type=int, default=1000,
                       help="Total number of timesteps in original trajectory (default: 1000)")
    parser.add_argument("--output_base_dir", type=str, default="generated_structures/cross_conditioned_late_insertion",
                       help="Base output directory")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get available strategies
    strategies = get_successful_conditioning_strategies()
    
    # Validate source and target conditions
    if args.source_condition not in strategies:
        print(f"Error: Source condition '{args.source_condition}' not found in available strategies")
        print(f"Available strategies: {list(strategies.keys())}")
        return
    
    if args.target_condition not in strategies:
        print(f"Error: Target condition '{args.target_condition}' not found in available strategies")
        print(f"Available strategies: {list(strategies.keys())}")
        return
    
    source_strategy = strategies[args.source_condition]
    target_strategy = strategies[args.target_condition]
    
    # Create descriptive output directory structure
    output_base_path = Path(args.output_base_dir)
    
    # Organize by source condition, timestep, then target condition
    run_name = f"{args.source_condition}_to_{args.target_condition}_step{args.start_timestep}_traj{args.trajectory_idx}"
    run_output_dir = output_base_path / f"step_{args.start_timestep}" / args.source_condition / args.target_condition / f"trajectory_{args.trajectory_idx}"
    
    print(f"\nCross-Conditioned Late Insertion Trajectory")
    print(f"{'='*60}")
    print(f"Source condition: {args.source_condition}")
    print(f"  -> {source_strategy['description']}")
    print(f"Target condition: {args.target_condition}")
    print(f"  -> {target_strategy['description']}")
    print(f"Start timestep: {args.start_timestep}")
    print(f"Trajectory index: {args.trajectory_idx}")
    print(f"Output directory: {run_output_dir}")
    print(f"{'='*60}")
    
    # Load structure from source condition trajectory
    trajectory_zip_path = f"generated_structures/three_property_study/{args.source_condition}/generated_trajectories.zip"
    
    if not Path(trajectory_zip_path).exists():
        print(f"Error: Source trajectory file not found: {trajectory_zip_path}")
        print("Make sure you have run the conditioned trajectories first.")
        return
    
    print(f"Loading structure from trajectory at timestep {args.start_timestep}")
    try:
        chemgraph, structure = load_structure_from_trajectory(
            trajectory_zip_path, args.trajectory_idx, args.start_timestep, device
        )
        print(f"Loaded structure with {len(structure)} atoms")
        print(f"Composition: {structure.composition}")
        print(f"Lattice parameters: {structure.lattice.parameters}")
        
    except Exception as e:
        print(f"Error loading structure: {e}")
        return
    
    # Save experiment configuration
    experiment_config = {
        "source_condition": args.source_condition,
        "target_condition": args.target_condition,
        "start_timestep": args.start_timestep,
        "trajectory_idx": args.trajectory_idx,
        "total_timesteps": args.total_timesteps,
        "source_strategy": source_strategy,
        "target_strategy": target_strategy,
        "source_trajectory_path": trajectory_zip_path,
        "starting_structure_info": {
            "composition": str(structure.composition),
            "lattice_parameters": list(structure.lattice.parameters)
        }
    }
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    with open(run_output_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    
    # Run the late insertion trajectory
    print(f"\nStarting late insertion trajectory generation...")
    
    try:
        generated_structures = run_late_insertion_trajectory(
            chemgraph=chemgraph,
            target_strategy=target_strategy,
            output_dir=run_output_dir,
            device=device,
            start_timestep=args.start_timestep,
            total_timesteps=args.total_timesteps,
            source_condition_name=args.source_condition
        )
        
        # Save results summary
        result = {
            "success": len(generated_structures) > 0,
            "num_generated": len(generated_structures),
            "source_condition": args.source_condition,
            "target_condition": args.target_condition,
            "start_timestep": args.start_timestep,
            "trajectory_idx": args.trajectory_idx
        }
        
        with open(run_output_dir / "result_summary.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{'='*60}")
        if result["success"]:
            print(f"✓ Cross-conditioning successful!")
            print(f"Generated {result['num_generated']} structures")
        else:
            print(f"✗ Cross-conditioning failed")
        print(f"Results saved in: {run_output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error in cross-conditioning: {e}")
        
        # Save error summary
        error_result = {
            "success": False,
            "error": str(e),
            "source_condition": args.source_condition,
            "target_condition": args.target_condition,
            "start_timestep": args.start_timestep,
            "trajectory_idx": args.trajectory_idx
        }
        
        with open(run_output_dir / "result_summary.json", "w") as f:
            json.dump(error_result, f, indent=2)


if __name__ == "__main__":
    main()