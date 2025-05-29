import os
import io
import torch
import zipfile
from pathlib import Path
from typing import Optional
import random
import ase
import numpy as np

from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.generator import CrystalGenerator
from mattergen.scripts.run_insertion_trajectories import create_sampler_from_config

def structure_to_chemgraph(structure: Structure) -> ChemGraph:
    """Convert a pymatgen Structure to a ChemGraph."""
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

def load_structure_from_trajectory(zip_path, trajectory_name=None, step_idx=1000):
    """Load a specific structure from a trajectory."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List all files in the zip
        trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
        
        # If no specific name given, pick the first one
        if not trajectory_name:
            chosen_file = trajectory_files[0]
        else:
            matching_files = [f for f in trajectory_files if trajectory_name in f]
            if not matching_files:
                raise ValueError(f"No trajectory found matching '{trajectory_name}'")
            chosen_file = matching_files[0]
            
        print(f"\nLoading trajectory: {chosen_file}")
        with zf.open(chosen_file) as f:
            content = io.StringIO(f.read().decode())
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            
            # Get the specific structure we want
            atoms = atoms_list[step_idx]
            structure = AseAtomsAdaptor.get_structure(atoms)
            
            # Create ChemGraph with original positions and cell
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Get atomic numbers and create tensor
            atomic_numbers = torch.tensor([site.specie.Z for site in structure], 
                                       dtype=torch.long, device=device)
            num_atoms = len(atomic_numbers)
            
            # Keep original fractional coordinates
            pos = torch.tensor(structure.frac_coords, dtype=torch.float32, device=device)
            
            # Keep original cell parameters
            cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Create batch indices - using the proper format for ChemGraph
            batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
            ptr = torch.tensor([0, num_atoms], dtype=torch.long, device=device)
            
            # Create ChemGraph with proper batch indexing
            chemgraph = ChemGraph(
                atomic_numbers=atomic_numbers,
                pos=pos,
                cell=cell,
                num_atoms=torch.tensor([num_atoms], dtype=torch.long, device=device),
                num_nodes=torch.tensor(num_atoms, dtype=torch.long, device=device),
                batch=batch,
                ptr=ptr,
                _batch_idx={"atomic_numbers": batch, "pos": batch, "cell": torch.zeros(1, dtype=torch.long, device=device)}
            )
            
            return chemgraph

def load_trajectory_from_zip(zip_path: str, trajectory_name: str) -> list[BatchedData]:
    """Load a trajectory from the zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List all files in the zip
        trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
        if not trajectory_files:
            raise ValueError(f"No trajectory files found in {zip_path}")
        
        print("\nAvailable trajectories:")
        for f in trajectory_files:
            print(f"  - {f}")
        
        # If no specific name given, pick the first one
        if not trajectory_name:
            chosen_file = trajectory_files[0]
        else:
            matching_files = [f for f in trajectory_files if trajectory_name in f]
            if not matching_files:
                raise ValueError(f"No trajectory found matching '{trajectory_name}'")
            chosen_file = matching_files[0]
            
        print(f"\nLoading trajectory: {chosen_file}")
        with zf.open(chosen_file) as f:
            content = io.StringIO(f.read().decode())
            # Debug: Print raw content
            print("\nFirst few lines of extxyz file:")
            first_lines = content.getvalue().split('\n')[:5]
            for line in first_lines:
                print(f"  {line}")
            content.seek(0)  # Reset file pointer
            
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            print(f"\nLoaded {len(atoms_list)} ASE Atoms objects")
            print(f"First structure has {len(atoms_list[0])} atoms")
            print(f"Cell parameters: {atoms_list[0].cell.cellpar()}")
            
            # Check ASE cell for NaN
            first_cell = atoms_list[0].cell
            if any(np.isnan(first_cell.array.flatten())):
                print("WARNING: NaN found in ASE cell matrix!")
                print(f"Cell matrix:\n{first_cell.array}")
            
            structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
            print(f"\nConverted to {len(structures)} pymatgen Structures")
            print(f"First structure lattice parameters: {structures[0].lattice.parameters}")
            
            # Check pymatgen lattice for NaN
            first_lattice = structures[0].lattice
            if any(np.isnan(first_lattice.matrix.flatten())):
                print("WARNING: NaN found in pymatgen lattice matrix!")
                print(f"Lattice matrix:\n{first_lattice.matrix}")
            
            chemgraphs = [structure_to_chemgraph(struct) for struct in structures]
            print(f"\nConverted to {len(chemgraphs)} ChemGraphs")
            print(f"First ChemGraph cell shape: {chemgraphs[0].cell.shape}")
            
            # Check ChemGraph tensors for NaN
            first_cg = chemgraphs[0]
            if torch.isnan(first_cg.cell).any():
                print("WARNING: NaN found in ChemGraph cell!")
                print(f"Cell tensor:\n{first_cg.cell}")
            if torch.isnan(first_cg.pos).any():
                print("WARNING: NaN found in ChemGraph positions!")
                print(f"Positions:\n{first_cg.pos}")
            
            # Convert each ChemGraph to BatchedData
            trajectory = [collate([cg]) for cg in chemgraphs]
            print(f"\nConverted to {len(trajectory)} BatchedData objects")
            print(f"First BatchedData cell shape: {trajectory[0].cell.shape}")
            print(f"First BatchedData pos shape: {trajectory[0].pos.shape}")
            print(f"First BatchedData atomic numbers: {trajectory[0].atomic_numbers}")
            
            # Check BatchedData tensors for NaN
            first_bd = trajectory[0]
            if torch.isnan(first_bd.cell).any():
                print("WARNING: NaN found in BatchedData cell!")
                print(f"Cell tensor:\n{first_bd.cell}")
            if torch.isnan(first_bd.pos).any():
                print("WARNING: NaN found in BatchedData positions!")
                print(f"Positions:\n{first_bd.pos}")
    
    return trajectory

def run_trajectory_from_midpoint(
    trajectory_path: str,
    trajectory_name: str = "",  # Made optional
    device: torch.device = None,
    dt: float = -0.001,
    output_dir: Optional[str] = None,
):
    """Run a trajectory starting from its midpoint.
    
    Args:
        trajectory_path: Path to the zip file containing trajectories
        trajectory_name: Name/pattern of trajectory to load. If empty, uses first trajectory
        device: Device to run on. If None, uses CUDA if available
        dt: Time step size
        output_dir: Directory to save results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the midpoint structure directly
    print(f"Loading trajectory from {trajectory_path}")
    chemgraph = load_structure_from_trajectory(trajectory_path, trajectory_name, step_idx=1000)
    
    # Convert ChemGraph to BatchedData
    start_structure = collate([chemgraph])
    start_structure = start_structure.to(device)
    
    # Check midpoint structure for NaN values
    print("\nChecking midpoint structure:")
    print(f"Cell shape: {start_structure.cell.shape}")
    print(f"Cell values:\n{start_structure.cell}")
    print(f"Positions shape: {start_structure.pos.shape}")
    print(f"Atomic numbers: {start_structure.atomic_numbers}")
    
    if torch.isnan(start_structure.cell).any():
        print("WARNING: NaN found in midpoint structure cell!")
        # Try to identify which elements are NaN
        nan_mask = torch.isnan(start_structure.cell)
        print(f"NaN locations in cell:\n{nan_mask}")
    
    if torch.isnan(start_structure.pos).any():
        print("WARNING: NaN found in midpoint structure positions!")
        # Try to identify which elements are NaN
        nan_mask = torch.isnan(start_structure.pos)
        print(f"NaN locations in positions:\n{nan_mask}")

    num_atoms = start_structure.num_atoms
    
    # Calculate time parameters
    T = 1.0  # Standard maximum time
    eps_t = -dt  # Use dt as the minimum time step
    start_t = 0.5  # Since we're starting from midpoint
    # For negative dt, we need to calculate N differently since we're going backwards
    N = int(abs((start_t - eps_t) / -dt))  # Use abs() since dt is negative
    
    print(f"\nTrajectory parameters:")
    print(f"start_t: {start_t:.6f}")
    print(f"dt: {dt}")
    print(f"eps_t: {eps_t}")
    print(f"New N: {N}")

    # Initialize generator
    print("\nInitializing generator...")
    num_atoms_int = num_atoms[0].item()  # Get integer from tensor
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo.from_hf_hub("mattergen_base"),
        batch_size=1,  # We're only processing one structure
        num_batches=1,
        num_atoms_distribution=f"JUST_{num_atoms_int}",  # Use the integer value
        record_trajectories=True
    )
    
    # Create time config
    time_config = {
        "start_t": start_t,
        "eps_t": eps_t,
        "N": N,
        "dt": dt
    }
    
    # Generate structures
    print("\nStarting sampling...")
    output_path = Path(output_dir) if output_dir else Path("generated_structures/continued_trajectories")
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_structures = generator.generate(
        output_dir=output_path,
        time_config=time_config,
        start_structure=start_structure
    )
    
    print(f"\nGenerated {len(generated_structures)} structures")
    print(f"Results saved in: {output_path}")
    return generated_structures

if __name__ == "__main__":
    # Example usage
    trajectory_path = "generated_structures/generated_trajectories.zip"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generated_structures = run_trajectory_from_midpoint(
        trajectory_path=trajectory_path,
        device=device,
        dt=-0.001,  # Use negative dt to go backwards in time
        output_dir="generated_structures/continued_trajectories"
    )
