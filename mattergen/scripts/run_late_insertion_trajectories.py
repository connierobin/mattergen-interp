#!/usr/bin/env python3
"""
Run late-stage insertion trajectories with property conditioning.

This script takes a structure from a late timepoint in an unconditioned trajectory
and continues the diffusion process for the last 200 timesteps with different
property conditioning.
"""

import os
import io
import torch
import zipfile
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
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


def analyze_starting_space_group(structure: Structure) -> Dict[str, Any]:
    """Analyze the space group of the starting structure."""
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5.0)
        return {
            'space_group_number': analyzer.get_space_group_number(),
            'space_group_symbol': analyzer.get_space_group_symbol(),
            'crystal_system': analyzer.get_crystal_system(),
            'lattice_type': analyzer.get_lattice_type()
        }
    except Exception as e:
        print(f"Warning: Space group analysis failed: {e}")
        return {
            'space_group_number': 1,
            'space_group_symbol': 'P1',
            'crystal_system': 'triclinic',
            'lattice_type': 'triclinic'
        }


def get_conditioning_strategies(starting_space_group: int) -> List[Dict[str, Any]]:
    """Get conditioning strategies for late-stage insertion."""
    
    # Pick 3 different space groups (avoiding the starting one)
    common_space_groups = [225, 221, 194, 139, 62, 166, 227, 229]  # Common high-symmetry groups
    selected_space_groups = [sg for sg in common_space_groups if sg != starting_space_group][:3]
    
    strategies = []
    
    # Space group conditioning (3 different target space groups)
    for i, sg in enumerate(selected_space_groups):
        space_group_info = {
            225: "Fm-3m (FCC cubic)",
            221: "Pm-3m (primitive cubic)", 
            194: "P63/mmc (hexagonal)",
            139: "I4/mmm (tetragonal)",
            62: "Pnma (orthorhombic)",
            166: "R-3m (rhombohedral)",
            227: "Fd-3m (diamond cubic)",
            229: "Im-3m (BCC cubic)"
        }
        
        strategies.append({
            "name": f"space_group_{sg}_{i+1}",
            "checkpoint_path": "space_group",
            "conditions": {"space_group": sg},
            "description": f"Space group {sg} ({space_group_info.get(sg, 'Unknown')})"
        })
    
    # Energy above hull conditioning (3 different target energies)
    energy_targets = [0.0, 0.05, 0.1]  # Stable, marginally stable, metastable
    for i, energy in enumerate(energy_targets):
        strategies.append({
            "name": f"energy_above_hull_{energy}_{i+1}",
            "checkpoint_path": "chemical_system_energy_above_hull", 
            "conditions": {"energy_above_hull": energy},
            "description": f"Energy above hull: {energy} eV/atom"
        })
    
    # Formation energy per atom conditioning (3 different target energies)
    formation_targets = [-1.0, 0.0, 0.5]  # Favorable, neutral, unfavorable
    for i, energy in enumerate(formation_targets):
        strategies.append({
            "name": f"formation_energy_{energy}_{i+1}",
            "checkpoint_path": "mattergen_base",  # Base model for formation energy
            "conditions": {"formation_energy_per_atom": energy},
            "description": f"Formation energy: {energy} eV/atom"
        })
    
    return strategies


def run_late_insertion_trajectory(
    chemgraph: ChemGraph,
    strategy: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    start_timestep: int = 800,
    total_timesteps: int = 1000
) -> List[Structure]:
    """Run a single late insertion trajectory with conditioning."""
    
    # Calculate time parameters for last 200 steps
    dt = -0.001  # Standard step size
    eps_t = 0.001  # End time (same as normal trajectory, non-zero to avoid numerical issues)
    start_t = 1.0 + start_timestep * dt  # Time at timestep 800: 1.0 + 800*(-0.001) = 0.2
    N = total_timesteps - start_timestep  # Number of remaining steps
    
    time_config = {
        "start_t": start_t,
        "eps_t": eps_t, 
        "N": N,
        "dt": dt
    }
    
    print(f"\nRunning strategy: {strategy['name']}")
    print(f"Conditions: {strategy['conditions']}")
    print(f"Time config: start_t={start_t:.3f}, eps_t={eps_t:.3f}, N={N}")
    
    # Create checkpoint info using Hugging Face Hub
    checkpoint_path = strategy["checkpoint_path"]
    model_name_mapping = {
        "mattergen_base": "mattergen_base",
        "space_group": "space_group",
        "chemical_system_energy_above_hull": "chemical_system_energy_above_hull"
    }
    
    if checkpoint_path in model_name_mapping:
        hf_model_name = model_name_mapping[checkpoint_path]
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(hf_model_name)
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
        properties_to_condition_on=strategy["conditions"],
        batch_size=1,
        num_batches=1,
        num_atoms_distribution=f"JUST_{num_atoms_int}",
        record_trajectories=True
    )
    
    # Create output directory for this strategy
    strategy_output_dir = output_dir / strategy["name"]
    strategy_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save strategy configuration
    with open(strategy_output_dir / "strategy_config.json", "w") as f:
        json.dump({
            "strategy": strategy,
            "time_config": time_config,
            "start_timestep": start_timestep,
            "total_timesteps": total_timesteps
        }, f, indent=2)
    
    try:
        # Generate structures starting from the late timepoint
        generated_structures = generator.generate(
            output_dir=strategy_output_dir,
            time_config=time_config,
            start_structure=start_structure
        )
        
        print(f" Successfully generated {len(generated_structures)} structures")
        return generated_structures
        
    except Exception as e:
        print(f" Error in strategy {strategy['name']}: {str(e)}")
        
        # Save error details
        with open(strategy_output_dir / "error.txt", "w") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Time config: {time_config}\n")
        
        return []


def main():
    """Main function to run late insertion trajectories."""

    # NOTE: failed with an error last time it ran, moved over to analyze_late_insertion_results.py instead
    
    # Configuration
    trajectory_zip_path = "generated_structures/three_property_study/unconditional/generated_trajectories.zip"
    output_base_dir = Path("generated_structures/late_insertion_trajectories")
    trajectory_idx = 0  # Use first trajectory
    start_timestep = 800  # Start from timestep 800 (last 200 steps)
    total_timesteps = 1000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load structure from late timepoint
    print(f"Loading structure from trajectory at timestep {start_timestep}")
    try:
        chemgraph, structure = load_structure_from_trajectory(
            trajectory_zip_path, trajectory_idx, start_timestep, device
        )
        print(f"Loaded structure with {len(structure)} atoms")
        print(f"Composition: {structure.composition}")
        print(f"Lattice parameters: {structure.lattice.parameters}")
        
    except Exception as e:
        print(f"Error loading structure: {e}")
        return
    
    # Analyze starting space group
    print("\nAnalyzing starting structure space group...")
    sg_info = analyze_starting_space_group(structure)
    print(f"Starting space group: {sg_info['space_group_number']} ({sg_info['space_group_symbol']})")
    print(f"Crystal system: {sg_info['crystal_system']}")
    
    # Get conditioning strategies - just energy above hull for now
    strategies = []
    
    # Energy above hull conditioning (3 different target energies)
    energy_targets = [0.0, 0.05, 0.1]  # Stable, marginally stable, metastable
    for i, energy in enumerate(energy_targets):
        strategies.append({
            "name": f"energy_above_hull_{energy}_{i+1}",
            "checkpoint_path": "chemical_system_energy_above_hull", 
            "conditions": {"energy_above_hull": energy},
            "description": f"Energy above hull: {energy} eV/atom"
        })
    
    print(f"\nWill run {len(strategies)} conditioning strategies:")
    for strategy in strategies:
        print(f"  - {strategy['name']}: {strategy['description']}")
    
    # Create output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    experiment_config = {
        "source_trajectory": trajectory_zip_path,
        "trajectory_idx": trajectory_idx,
        "start_timestep": start_timestep,
        "total_timesteps": total_timesteps,
        "starting_structure_info": {
            "composition": str(structure.composition),
            "space_group": sg_info,
            "lattice_parameters": list(structure.lattice.parameters)
        },
        "strategies": strategies
    }
    
    with open(output_base_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    
    # Run all strategies
    print(f"\n{'='*60}")
    print("Starting late insertion trajectory generation")
    print(f"{'='*60}")
    
    results = []
    successful = 0
    failed = 0
    
    for i, strategy in enumerate(strategies):
        print(f"\n[{i+1}/{len(strategies)}] {strategy['description']}")
        
        try:
            generated_structures = run_late_insertion_trajectory(
                chemgraph=chemgraph,
                strategy=strategy,
                output_dir=output_base_dir,
                device=device,
                start_timestep=start_timestep,
                total_timesteps=total_timesteps
            )
            
            if generated_structures:
                successful += 1
                results.append({
                    "strategy_name": strategy["name"],
                    "success": True,
                    "num_generated": len(generated_structures),
                    "conditions": strategy["conditions"]
                })
            else:
                failed += 1
                results.append({
                    "strategy_name": strategy["name"],
                    "success": False,
                    "error": "No structures generated",
                    "conditions": strategy["conditions"]
                })
                
        except Exception as e:
            failed += 1
            print(f" Error in strategy {strategy['name']}: {str(e)}")
            results.append({
                "strategy_name": strategy["name"],
                "success": False,
                "error": str(e),
                "conditions": strategy["conditions"]
            })
    
    # Save results summary
    results_summary = {
        "experiment_config": experiment_config,
        "results": results,
        "summary": {
            "total_strategies": len(strategies),
            "successful": successful,
            "failed": failed
        }
    }
    
    with open(output_base_dir / "results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("Late Insertion Trajectory Generation Complete!")
    print(f"Successful strategies: {successful}/{len(strategies)}")
    print(f"Failed strategies: {failed}")
    if successful > 0:
        total_structures = sum(r.get("num_generated", 0) for r in results if r["success"])
        print(f"Total structures generated: {total_structures}")
    print(f"Results saved in: {output_base_dir}")
    print(f"{'='*60}")
    
    # Print successful strategies
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        print(f"\nSuccessful strategies:")
        for result in successful_results:
            print(f"  - {result['strategy_name']}: {result['num_generated']} structures")
    
    # Print failed strategies
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        print(f"\nFailed strategies:")
        for result in failed_results:
            print(f"  - {result['strategy_name']}: {result['error']}")


if __name__ == "__main__":
    main()