#!/usr/bin/env python3
"""
Debug version of run_late_insertion_trajectories.py with error handling and intermediate saves.
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


def save_intermediate_structure(chemgraph: ChemGraph, output_path: Path, step_name: str):
    """Save intermediate structure for debugging."""
    try:
        # Convert ChemGraph to pymatgen Structure for saving
        from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
        
        # Extract data from ChemGraph
        pos = chemgraph.pos.cpu().numpy()
        atomic_numbers = chemgraph.atomic_numbers.cpu().numpy()
        cell_matrix = chemgraph.cell[0].cpu().numpy()  # Remove batch dimension
        
        # Create a simple structure for saving
        atoms = ase.Atoms(
            numbers=atomic_numbers,
            positions=pos @ cell_matrix,  # Convert frac to cart coords
            cell=cell_matrix,
            pbc=True
        )
        
        # Save as extxyz
        output_file = output_path / f"{step_name}_intermediate.extxyz"
        ase.io.write(output_file, atoms, format="extxyz")
        print(f"  Saved intermediate structure: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  Warning: Could not save intermediate structure {step_name}: {e}")
        return False


def run_late_insertion_trajectory_debug(
    chemgraph: ChemGraph,
    strategy: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    start_timestep: int = 800,
    total_timesteps: int = 1000
) -> List[Structure]:
    """Run a single late insertion trajectory with extensive debugging."""
    
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
    
    # Create output directory for this strategy
    strategy_output_dir = output_dir / strategy["name"]
    strategy_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save starting structure for debugging
    save_intermediate_structure(chemgraph, strategy_output_dir, "start_structure")
    
    # Save strategy configuration
    with open(strategy_output_dir / "strategy_config.json", "w") as f:
        json.dump({
            "strategy": strategy,
            "time_config": time_config,
            "start_timestep": start_timestep,
            "total_timesteps": total_timesteps
        }, f, indent=2)
    
    try:
        # Initialize generator with conditioning
        generator = CrystalGenerator(
            checkpoint_info=checkpoint_info,
            properties_to_condition_on=strategy["conditions"],
            batch_size=1,
            num_batches=1,
            num_atoms_distribution=f"JUST_{num_atoms_int}",
            record_trajectories=True
        )
        
        print(f"Generator initialized successfully")
        
        # Instead of using generator.generate directly, let's use the lower-level components
        # to have more control over error handling
        from mattergen.generator import draw_samples_from_sampler
        from hydra.utils import instantiate
        
        # Get sampling config and setup
        sampling_config = generator.load_sampling_config(
            batch_size=1,
            num_batches=1,
            target_compositions_dict=None,
        )
        
        condition_loader = generator.get_condition_loader(sampling_config, None)
        sampler_partial = instantiate(sampling_config.sampler_partial)
        sampler = sampler_partial(pl_module=generator.model)
        
        print(f"Sampler setup complete, starting generation...")
        
        # Generate structures starting from the late timepoint
        generated_structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=condition_loader,
            cfg=generator.cfg,
            output_path=strategy_output_dir,
            properties_to_condition_on=strategy["conditions"],
            record_trajectories=True,
            time_config=time_config,
            start_structure=start_structure
        )
        
        print(f"✓ Successfully generated {len(generated_structures)} structures")
        return generated_structures
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error in strategy {strategy['name']}: {error_msg}")
        
        # Check if it's the SVD error we're expecting
        if "linalg.svd" in error_msg and "failed to converge" in error_msg:
            print(f"  This is the expected SVD convergence error in lattice processing")
            print(f"  Checking if we can recover intermediate structures...")
            
            # Try to load any trajectories that were saved
            trajectory_zip = strategy_output_dir / "generated_trajectories.zip"
            if trajectory_zip.exists():
                print(f"  Found trajectory file: {trajectory_zip}")
                try:
                    # Load the last few structures from trajectory
                    with zipfile.ZipFile(trajectory_zip, 'r') as zf:
                        trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
                        if trajectory_files:
                            chosen_file = trajectory_files[0]
                            print(f"  Loading structures from: {chosen_file}")
                            
                            with zf.open(chosen_file) as f:
                                content = io.StringIO(f.read().decode())
                                atoms_list = ase.io.read(content, index=":", format="extxyz")
                                
                                print(f"  Found {len(atoms_list)} trajectory steps")
                                
                                # Save the last few structures that worked
                                for i, atoms in enumerate(atoms_list[-5:]):  # Last 5 structures
                                    try:
                                        structure = AseAtomsAdaptor.get_structure(atoms)
                                        save_path = strategy_output_dir / f"recovered_structure_{len(atoms_list)-5+i}.extxyz"
                                        ase.io.write(save_path, atoms, format="extxyz")
                                        print(f"    Saved recovery structure: {save_path}")
                                    except Exception as recover_error:
                                        print(f"    Could not save recovery structure {i}: {recover_error}")
                                
                                # Try to return the second-to-last structure as the final result
                                if len(atoms_list) >= 2:
                                    try:
                                        final_atoms = atoms_list[-2]  # Second to last
                                        final_structure = AseAtomsAdaptor.get_structure(final_atoms)
                                        print(f"  ✓ Successfully recovered final structure from step {len(atoms_list)-2}")
                                        return [final_structure]
                                    except Exception as final_error:
                                        print(f"  Could not convert final recovery structure: {final_error}")
                                        
                except Exception as recovery_error:
                    print(f"  Recovery attempt failed: {recovery_error}")
        
        # Save error details
        with open(strategy_output_dir / "error.txt", "w") as f:
            f.write(f"Error: {error_msg}\n")
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Time config: {time_config}\n")
        
        return []


def main():
    """Main function to run late insertion trajectories with debugging."""
    
    # Configuration
    trajectory_zip_path = "generated_structures/three_property_study/unconditional/generated_trajectories.zip"
    output_base_dir = Path("generated_structures/late_insertion_trajectories_debug")
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
    
    # Just test one strategy for debugging - using energy instead of space group to see if that helps
    strategies = [
        {
            "name": "energy_above_hull_0.0_debug",
            "checkpoint_path": "chemical_system_energy_above_hull",
            "conditions": {"energy_above_hull": 0.0},
            "description": "Energy above hull: 0.0 eV/atom - DEBUG"
        }
    ]
    
    print(f"\nRunning debug test with 1 strategy:")
    for strategy in strategies:
        print(f"  - {strategy['name']}: {strategy['description']}")
    
    # Create output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the strategy
    print(f"\n{'='*60}")
    print("Starting late insertion trajectory generation (DEBUG)")
    print(f"{'='*60}")
    
    try:
        generated_structures = run_late_insertion_trajectory_debug(
            chemgraph=chemgraph,
            strategy=strategies[0],
            output_dir=output_base_dir,
            device=device,
            start_timestep=start_timestep,
            total_timesteps=total_timesteps
        )
        
        if generated_structures:
            print(f"\n✓ SUCCESS: Generated {len(generated_structures)} structures")
            print(f"Results saved in: {output_base_dir}")
        else:
            print(f"\n✗ No structures generated, but error recovery attempted")
            
    except Exception as e:
        print(f"\n✗ Complete failure: {str(e)}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()