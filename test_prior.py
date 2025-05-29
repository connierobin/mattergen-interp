import torch
from mattergen.generator import CrystalGenerator
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.sampling.pc_sampler import _sample_prior
from mattergen.common.data.condition_factory import get_number_of_atoms_condition_loader
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
import zipfile
import io
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor

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

# Initialize generator with base model
generator = CrystalGenerator(
    checkpoint_info=MatterGenCheckpointInfo.from_hf_hub("mattergen_base"),
    batch_size=1,
    num_batches=1,
    record_trajectories=True
)

# First, let's see what a normal prior sample looks like
sampling_config = generator.load_sampling_config(
    batch_size=1,
    num_batches=1,
    target_compositions_dict=None
)

# Get condition loader
condition_loader = generator.get_condition_loader(sampling_config)

print("\n=== Default Prior Sample Structure ===")
# Get first batch of conditioning data
for conditioning_data, mask in condition_loader:
    print("\nConditioning data structure:")
    print("Type:", type(conditioning_data))
    print("\nFields:")
    for key in conditioning_data.keys():
        if isinstance(conditioning_data[key], torch.Tensor):
            print(f"{key}:", conditioning_data[key].shape, conditioning_data[key].dtype)
        else:
            print(f"{key}: (not a tensor)", type(conditioning_data[key]))
        print(f"Values: {conditioning_data[key]}")
    
    # Get the multi_corruption from the generator's model
    multi_corruption = generator.model.diffusion_module.corruption
    
    # Sample prior
    print("\nSampling prior...")
    prior_sample = _sample_prior(multi_corruption, conditioning_data, mask)
    
    print("\nPrior sample structure:")
    print("Type:", type(prior_sample))
    print("\nFields:")
    for key in prior_sample.keys():
        if isinstance(prior_sample[key], torch.Tensor):
            print(f"{key}:", prior_sample[key].shape, prior_sample[key].dtype)
        else:
            print(f"{key}: (not a tensor)", type(prior_sample[key]))
        print(f"Values: {prior_sample[key]}")
    break  # Only need first batch

print("\n=== Our Structure ===")
# Now load our structure and see its shape
chemgraph = load_structure_from_trajectory("generated_structures/generated_trajectories.zip", step_idx=1000)

print("\nConverted structure:")
print("Type:", type(chemgraph))
print("\nFields:")
for key in chemgraph.keys():
    if isinstance(chemgraph[key], torch.Tensor):
        print(f"{key}:", chemgraph[key].shape, chemgraph[key].dtype)
    else:
        print(f"{key}: (not a tensor)", type(chemgraph[key]))
    print(f"Values: {chemgraph[key]}") 