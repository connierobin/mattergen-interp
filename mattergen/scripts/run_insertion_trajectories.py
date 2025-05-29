from pathlib import Path
import json
import random
import zipfile
import io
from typing import List, Optional, Dict

import ase.io
import numpy as np
import torch
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from hydra.utils import instantiate
from omegaconf import DictConfig

from mattergen.generator import CrystalGenerator
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.common.data.collate import collate
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch

def structure_to_chemgraph(structure: Structure) -> ChemGraph:
    """Convert a pymatgen Structure to a ChemGraph.
    
    Args:
        structure: Pymatgen Structure object
        
    Returns:
        ChemGraph object
    """
    # Get atomic numbers
    atomic_numbers = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    
    # Get fractional coordinates
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float)
    
    # Get lattice matrix
    lattice_matrix = torch.tensor(structure.lattice.matrix, dtype=torch.float).unsqueeze(0)  # Add batch dimension
    
    # Create ChemGraph
    return ChemGraph(
        atomic_numbers=atomic_numbers,
        pos=frac_coords,
        cell=lattice_matrix,
        num_atoms=torch.tensor(len(structure), dtype=torch.long),
        num_nodes=len(structure)  # Required for PyG batching
    )

def load_trajectory_from_zip(traj_file: Path, traj_idx: int) -> List[ChemGraph]:
    """Load a specific trajectory from the generated_trajectories.zip file.
    
    Args:
        traj_file: Path to the trajectories zip file
        traj_idx: Index of the trajectory to load
        
    Returns:
        List of ChemGraph objects representing the trajectory
    """
    with zipfile.ZipFile(traj_file) as zf:
        filename = f"gen_{traj_idx}.extxyz"
        if filename not in zf.namelist():
            raise ValueError(f"Trajectory {traj_idx} not found in {traj_file}")
            
        with zf.open(filename) as f:
            content = io.StringIO(f.read().decode())
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
            return [structure_to_chemgraph(struct) for struct in structures]

def select_trajectories(input_dir: Path, num_trajectories: int, seed: int) -> List[int]:
    """Randomly select trajectory indices from the available trajectories.
    
    Args:
        input_dir: Directory containing the generated_trajectories.zip
        num_trajectories: Number of trajectories to select
        seed: Random seed for reproducibility
        
    Returns:
        List of selected trajectory indices
    """
    traj_file = input_dir / "generated_trajectories.zip"
    if not traj_file.exists():
        raise FileNotFoundError(f"No trajectory file found at {traj_file}")
    
    # Get total number of trajectories
    with zipfile.ZipFile(traj_file) as zf:
        total_trajectories = len([f for f in zf.namelist() if f.endswith('.extxyz')])
    
    if num_trajectories > total_trajectories:
        raise ValueError(f"Requested {num_trajectories} trajectories but only {total_trajectories} available")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Randomly select trajectories
    selected_indices = random.sample(range(total_trajectories), num_trajectories)
    return selected_indices

def create_sampler_from_config(
    sampling_config: DictConfig,
    pl_module,
    start_t: float,
    start_structure: ChemGraph
) -> PredictorCorrector:
    """Create a PredictorCorrector sampler from config and parameters.
    
    Args:
        sampling_config: The Hydra config containing sampler parameters
        pl_module: The PyTorch Lightning module to create the sampler from
        start_t: The starting time for the diffusion process
        start_structure: The structure to start from
        
    Returns:
        Configured PredictorCorrector sampler
    """
    # Extract and instantiate predictor/corrector partials
    predictor_partials = {
        k: instantiate(v) 
        for k, v in sampling_config.sampler_partial.predictor_partials.items()
    } if hasattr(sampling_config.sampler_partial, 'predictor_partials') else None
    
    corrector_partials = {
        k: instantiate(v)
        for k, v in sampling_config.sampler_partial.corrector_partials.items()
    } if hasattr(sampling_config.sampler_partial, 'corrector_partials') else None
    
    # Extract only the relevant arguments from sampler_partial
    sampler_args = {
        'N': sampling_config.sampler_partial.N,
        'eps_t': sampling_config.sampler_partial.eps_t,
        'predictor_partials': predictor_partials,
        'corrector_partials': corrector_partials,
        'n_steps_corrector': sampling_config.sampler_partial.n_steps_corrector,
        'start_t': start_t,
        'start_structure': start_structure
    }
    
    return PredictorCorrector.from_pl_module(
        pl_module=pl_module,
        **sampler_args
    )

def run_insertion_trajectories(
    input_dir: str = "generated_structures",
    output_dir: str = "insertion_trajectories",
    num_trajectories: int = 1,
    step_interval: int = 200,
    seed: int = 42,
    batch_size: int = 32,
    num_batches: int = 1,
    record_trajectories: bool = True,
    test_mode: bool = True,
    dt: float = 0.001  # Add dt parameter
):
    """Run trajectories starting from intermediate diffusion steps.
    
    Args:
        input_dir: Directory containing the original trajectories
        output_dir: Directory to save the new trajectories
        num_trajectories: Number of trajectories to select
        step_interval: Interval between diffusion steps to start from
        seed: Random seed for reproducibility
        batch_size: Batch size for generation
        num_batches: Number of batches to generate
        record_trajectories: Whether to save the generation trajectories
        test_mode: If True, only run one test case from step 600
        dt: Time step size for diffusion (default: 0.001)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration
    config = {
        "input_dir": str(input_dir),
        "num_trajectories": num_trajectories,
        "step_interval": step_interval,
        "seed": seed,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "record_trajectories": record_trajectories,
        "test_mode": test_mode,
        "dt": dt
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Select trajectories
    selected_indices = select_trajectories(input_path, num_trajectories, seed)
    print(f"Selected trajectories: {selected_indices}")
    
    # For each selected trajectory
    for traj_idx in selected_indices:
        print(f"\nProcessing trajectory {traj_idx}")
        
        # Load the trajectory
        print("Loading trajectory...")
        trajectory = load_trajectory_from_zip(input_path / "generated_trajectories.zip", traj_idx)
        print(f"Loaded trajectory with {len(trajectory)} steps")
        
        # Create directory for this trajectory's results
        traj_dir = output_path / f"trajectory_{traj_idx}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        if test_mode:
            # Just run one test case from step 600
            start_step = 600
            print(f"\nTEST MODE: Running single case from step {start_step}")
            
            # Calculate start_t and N
            start_t = start_step / len(trajectory)
            max_t = 1.0
            N = int((max_t - start_t) / dt)
            print(f"\nDiffusion parameters:")
            print(f"dt: {dt}")
            print(f"start_t: {start_t}")
            print(f"max_t: {max_t}")
            print(f"eps_t: {dt}")
            print(f"N (calculated): {N}")
            
            # Initialize generator with the correct N value
            print("\nInitializing generator...")
            generator = CrystalGenerator(
                checkpoint_info=MatterGenCheckpointInfo.from_hf_hub("mattergen_base"),
                batch_size=batch_size,
                num_batches=num_batches,
                record_trajectories=record_trajectories,
                config_overrides=[
                    "model.corruption.discrete_corruptions.atomic_numbers.N=700",
                    "model.corruption.discrete_corruptions.pos.N=700",
                    "model.corruption.discrete_corruptions.cell.N=700"
                ],
                sampling_config_overrides=[
                    f"sampler_partial.N={N}",  # Set N based on start_t and dt
                    f"sampler_partial.eps_t={dt}"  # Set eps_t to match dt
                ]
            )
            
            # Get the starting structure as a ChemGraph
            start_chemgraph = trajectory[start_step]
            print(f"Starting structure fields: {list(start_chemgraph.keys())}")
            print(f"Starting structure attributes:")
            for key in start_chemgraph.keys():
                value = start_chemgraph[key]
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {value}")
            
            # Create a batch from the single ChemGraph
            batch = collate([start_chemgraph])
            print(f"Batch size: {batch.get_batch_size()}")
            print(f"Batch attributes:")
            for key in batch.keys():
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {value}")
            
            # Get sampling config
            sampling_config = generator.load_sampling_config(
                batch_size=batch_size,
                num_batches=num_batches,
                target_compositions_dict=None
            )
            
            # Create sampler with start_t and start_structure
            print("\nSampling config contents:")
            print("sampler_partial fields:", list(sampling_config.sampler_partial.keys()))
            print("sampler_partial values:", dict(sampling_config.sampler_partial))
            
            sampler = create_sampler_from_config(
                sampling_config=sampling_config,
                pl_module=generator.model,
                start_t=start_t,
                start_structure=batch
            )
            
            # Get condition loader
            print("Getting condition loader...")
            condition_loader = generator.get_condition_loader(sampling_config)
            
            # Generate new trajectory from this point
            print("\nStarting generation...")
            if record_trajectories:
                sample, mean, intermediate_samples = sampler.sample_with_record(batch, mask=None)
                # Save intermediate samples
                print("\nSaving intermediate samples...")
                with zipfile.ZipFile(traj_dir / f"start_step_{start_step}_trajectories.zip", 'w') as zf:
                    for i, sample_batch in enumerate(intermediate_samples):
                        # Convert to structures
                        for j, chemgraph in enumerate(sample_batch.to_data_list()):
                            cell = chemgraph.cell
                            lengths, angles = lattice_matrix_to_params_torch(cell)
                            struct = Structure(
                                lattice=chemgraph.cell[0].cpu().numpy(),
                                species=[int(z) for z in chemgraph.atomic_numbers.cpu()],
                                coords=chemgraph.pos.cpu().numpy(),
                                coords_are_cartesian=False
                            )
                            # Save as extxyz
                            atoms = AseAtomsAdaptor.get_atoms(struct)
                            str_io = io.StringIO()
                            ase.io.write(str_io, atoms, format="extxyz")
                            zf.writestr(f"step_{i}_sample_{j}.extxyz", str_io.getvalue())
            else:
                sample, mean = sampler.sample(batch, mask=None)
            
            # Save the final structures
            print("\nSaving final structures...")
            final_structures = []
            for chemgraph in mean.to_data_list():
                cell = chemgraph.cell
                lengths, angles = lattice_matrix_to_params_torch(cell)
                struct = Structure(
                    lattice=chemgraph.cell[0].cpu().numpy(),
                    species=[int(z) for z in chemgraph.atomic_numbers.cpu()],
                    coords=chemgraph.pos.cpu().numpy(),
                    coords_are_cartesian=False
                )
                final_structures.append(struct)
            
            # Save final structures
            with zipfile.ZipFile(traj_dir / f"start_step_{start_step}_final.zip", 'w') as zf:
                for i, struct in enumerate(final_structures):
                    atoms = AseAtomsAdaptor.get_atoms(struct)
                    str_io = io.StringIO()
                    ase.io.write(str_io, atoms, format="extxyz")
                    zf.writestr(f"final_{i}.extxyz", str_io.getvalue())
            
            print("\nTest case complete!")
            return  # Exit after test case
            
        # Normal mode - process all steps
        # Get total number of steps in the trajectory
        total_steps = len(trajectory)
        
        # For each starting step (skip step 0)
        for start_step in range(step_interval, total_steps, step_interval):
            # Get the starting structure as a ChemGraph
            start_chemgraph = trajectory[start_step]
            
            # Create a batch from the single ChemGraph
            batch = collate([start_chemgraph])
            
            # Calculate start_t for this starting point
            start_t = start_step / total_steps
            
            # Calculate N based on time interval and dt
            max_t = 1.0
            N = int((max_t - start_t) / dt)
            
            # Get new sampling config with adjusted N
            sampling_config = generator.load_sampling_config(
                batch_size=batch_size,
                num_batches=num_batches,
                config_overrides=[
                    f"model.corruption.discrete_corruptions.atomic_numbers.N={N}",
                    f"model.corruption.discrete_corruptions.pos.N={N}",
                    f"model.corruption.discrete_corruptions.cell.N={N}"
                ],
                sampling_config_overrides=[
                    f"sampler_partial.N={N}",
                    f"sampler_partial.eps_t={dt}"
                ]
            )
            
            # Create sampler with start_t and start_structure
            print("\nSampling config contents:")
            print("sampler_partial fields:", list(sampling_config.sampler_partial.keys()))
            print("sampler_partial values:", dict(sampling_config.sampler_partial))
            
            sampler = create_sampler_from_config(
                sampling_config=sampling_config,
                pl_module=generator.model,
                start_t=start_t,
                start_structure=batch
            )
            
            # Get condition loader
            condition_loader = generator.get_condition_loader(sampling_config)
            
            # Generate new trajectory from this point
            if record_trajectories:
                sample, mean, intermediate_samples = sampler.sample_with_record(batch, mask=None)
                # Save intermediate samples
                with zipfile.ZipFile(traj_dir / f"start_step_{start_step}_trajectories.zip", 'w') as zf:
                    for i, sample_batch in enumerate(intermediate_samples):
                        # Convert to structures
                        for j, chemgraph in enumerate(sample_batch.to_data_list()):
                            cell = chemgraph.cell
                            lengths, angles = lattice_matrix_to_params_torch(cell)
                            struct = Structure(
                                lattice=chemgraph.cell[0].cpu().numpy(),
                                species=[int(z) for z in chemgraph.atomic_numbers.cpu()],
                                coords=chemgraph.pos.cpu().numpy(),
                                coords_are_cartesian=False
                            )
                            # Save as extxyz
                            atoms = AseAtomsAdaptor.get_atoms(struct)
                            str_io = io.StringIO()
                            ase.io.write(str_io, atoms, format="extxyz")
                            zf.writestr(f"step_{i}_sample_{j}.extxyz", str_io.getvalue())
            else:
                sample, mean = sampler.sample(batch, mask=None)
            
            # Save the final structures
            final_structures = []
            for chemgraph in mean.to_data_list():
                cell = chemgraph.cell
                lengths, angles = lattice_matrix_to_params_torch(cell)
                struct = Structure(
                    lattice=chemgraph.cell[0].cpu().numpy(),
                    species=[int(z) for z in chemgraph.atomic_numbers.cpu()],
                    coords=chemgraph.pos.cpu().numpy(),
                    coords_are_cartesian=False
                )
                final_structures.append(struct)
            
            # Save final structures
            with zipfile.ZipFile(traj_dir / f"start_step_{start_step}_final.zip", 'w') as zf:
                for i, struct in enumerate(final_structures):
                    atoms = AseAtomsAdaptor.get_atoms(struct)
                    str_io = io.StringIO()
                    ase.io.write(str_io, atoms, format="extxyz")
                    zf.writestr(f"final_{i}.extxyz", str_io.getvalue())

if __name__ == "__main__":
    run_insertion_trajectories(test_mode=True, dt=0.001)  # Run in test mode with default dt
