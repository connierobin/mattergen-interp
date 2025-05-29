import io
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

import ase.io
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Species, Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_matcher import StructureMatcher
import pandas as pd
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from tqdm import tqdm

def inspect_xyz_content(file_content: str, max_lines: int = 10):
    """Print the first few lines of an xyz file to inspect its format."""
    print("\nXYZ File Content (first few lines):")
    for i, line in enumerate(file_content.splitlines()[:max_lines]):
        print(f"Line {i+1}: {line}")

def inspect_atomic_conversion(atoms):
    """Inspect how atomic numbers are being converted to symbols."""
    print("\nAtomic Number Conversion Analysis:")
    numbers = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()
    unique_pairs = set(zip(numbers, symbols))
    print("Atomic number to symbol mapping:")
    for num, sym in sorted(unique_pairs):
        print(f"  {num} -> {sym}")
    return unique_pairs

def load_final_structures(output_dir: Path) -> List[Structure]:
    """Load the final generated structures from either the CIF zip or extxyz file.
    
    Args:
        output_dir: Directory containing the generation results
        
    Returns:
        List of pymatgen Structure objects
    """
    structures = []
    
    # Try loading from CIF zip first
    cif_zip = output_dir / "generated_crystals_cif.zip"
    if cif_zip.exists():
        with zipfile.ZipFile(cif_zip) as zf:
            for filename in zf.namelist():
                if filename.endswith('.cif'):
                    with zf.open(filename) as f:
                        struct = Structure.from_str(f.read().decode(), fmt="cif")
                        structures.append(struct)
        return structures
    
    # Fall back to extxyz file
    xyz_file = output_dir / "generated_crystals.extxyz"
    if xyz_file.exists():
        atoms_list = ase.io.read(xyz_file, index=":")
        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
        return structures
    
    raise FileNotFoundError(f"No structure files found in {output_dir}")

def load_trajectories(output_dir: Path) -> List[List[Structure]]:
    """Load all trajectories from the generated_trajectories.zip file.
    
    Args:
        output_dir: Directory containing the generation results
        
    Returns:
        List of trajectories, where each trajectory is a list of Structure objects
    """
    traj_file = output_dir / "generated_trajectories.zip"
    if not traj_file.exists():
        raise FileNotFoundError(f"No trajectory file found at {traj_file}")
    
    trajectories = []
    with zipfile.ZipFile(traj_file) as zf:
        for filename in sorted(zf.namelist()):
            if filename.endswith('.extxyz'):
                with zf.open(filename) as f:
                    content = io.StringIO(f.read().decode())
                    atoms_list = ase.io.read(content, index=":", format="extxyz")
                    structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
                    trajectories.append(structures)
    
    return trajectories

def get_lattice_angles(structure: Structure) -> tuple[float, float, float]:
    """Get the angles between lattice vectors in degrees."""
    a, b, c = structure.lattice.matrix
    alpha = np.degrees(np.arccos(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))))
    beta = np.degrees(np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))))
    gamma = np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return alpha, beta, gamma

def analyze_trajectory_statistics(trajectories: List[List[Structure]]) -> Dict[str, Any]:
    """Compute basic statistics about the trajectories.
    
    Args:
        trajectories: List of trajectories, where each trajectory is a list of Structure objects
        
    Returns:
        Dictionary containing various statistics
    """
    # Basic statistics
    stats = {
        "num_trajectories": len(trajectories),
        "trajectory_lengths": [len(traj) for traj in trajectories],
        "avg_trajectory_length": np.mean([len(traj) for traj in trajectories]),
        "num_atoms": [traj[0].num_sites for traj in trajectories],
        "avg_num_atoms": np.mean([traj[0].num_sites for traj in trajectories]),
    }
    
    # Detailed composition analysis
    final_structures = [traj[-1] for traj in trajectories]
    compositions = [struct.composition.reduced_formula for struct in final_structures]
    stats["composition_counts"] = Counter(compositions)
    
    # Detailed structure analysis for final structures
    structure_details = []
    for struct in final_structures:
        alpha, beta, gamma = get_lattice_angles(struct)
        details = {
            "composition": struct.composition.reduced_formula,
            "num_atoms": struct.num_sites,
            "volume": struct.volume,
            "density": struct.density,
            "lattice_angles": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            },
            "lattice_params": {
                "a": struct.lattice.a,
                "b": struct.lattice.b,
                "c": struct.lattice.c
            }
        }
        structure_details.append(details)
    stats["structure_details"] = structure_details
    
    # Volume evolution
    volume_evolution = []
    for traj in trajectories:
        volumes = [struct.volume for struct in traj]
        volume_evolution.append({
            "initial": volumes[0],
            "final": volumes[-1],
            "change": volumes[-1] - volumes[0],
            "percent_change": (volumes[-1] - volumes[0]) / volumes[0] * 100
        })
    stats["volume_evolution"] = volume_evolution
    
    return stats

def analyze_initial_states(trajectories: List[List[Structure]]) -> Dict[str, Any]:
    """Analyze the initial states of all trajectories to verify consistency.
    
    Args:
        trajectories: List of trajectories
        
    Returns:
        Dictionary containing initial state statistics
    """
    initial_states = [traj[0] for traj in trajectories]
    
    # Analyze species in initial states
    initial_stats = {
        "num_trajectories": len(trajectories),
        "species_counts": [],
        "structure_details": []
    }
    
    # Get detailed information about each initial state
    for i, struct in enumerate(initial_states):
        # Get species information
        species_count = Counter()
        atomic_numbers = []
        for site in struct:
            species = site.specie
            species_count[str(species)] += 1
            atomic_numbers.append(species.Z if hasattr(species, 'Z') else -1)
            
        alpha, beta, gamma = get_lattice_angles(struct)
        details = {
            "trajectory_idx": i,
            "species_count": dict(species_count),
            "atomic_numbers": atomic_numbers,
            "num_atoms": struct.num_sites,
            "volume": struct.volume,
            "lattice_angles": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            },
            "lattice_params": {
                "a": struct.lattice.a,
                "b": struct.lattice.b,
                "c": struct.lattice.c
            }
        }
        initial_stats["structure_details"].append(details)
        
        # Add to overall species counts if not already present
        for species, count in species_count.items():
            found = False
            for existing in initial_stats["species_counts"]:
                if existing["species"] == species:
                    found = True
                    break
            if not found:
                initial_stats["species_counts"].append({
                    "species": species,
                    "atomic_number": atomic_numbers[0],  # Assuming all species are same in initial state
                    "count": count
                })
    
    return initial_stats

def print_initial_state_analysis(stats: Dict[str, Any]):
    """Print analysis of initial states."""
    print("\n=== Initial State Analysis ===")
    
    print("\nSpecies found in initial states:")
    for species_info in stats["species_counts"]:
        print(f"Species: {species_info['species']}")
        print(f"  Atomic number: {species_info['atomic_number']}")
        print(f"  Count per structure: {species_info['count']}")
    
    print("\nInitial structure details:")
    details = stats["structure_details"]
    
    # First check if all structures are identical
    first_struct = details[0]
    all_identical = all(
        np.allclose([s["lattice_params"][p] for p in ["a", "b", "c"]], 
                   [first_struct["lattice_params"][p] for p in ["a", "b", "c"]], 
                   atol=1e-3)
        and np.allclose([s["lattice_angles"][a] for a in ["alpha", "beta", "gamma"]], 
                       [first_struct["lattice_angles"][a] for a in ["alpha", "beta", "gamma"]], 
                       atol=1e-3)
        and s["species_count"] == first_struct["species_count"]
        for s in details[1:]
    )
    
    if all_identical:
        print("\nAll initial structures are identical:")
        struct = details[0]
        print(f"  Number of atoms: {struct['num_atoms']}")
        print(f"  Volume: {struct['volume']:.2f} Å³")
        print(f"  Lattice angles: α={struct['lattice_angles']['alpha']:.1f}°, "
              f"β={struct['lattice_angles']['beta']:.1f}°, "
              f"γ={struct['lattice_angles']['gamma']:.1f}°")
        print(f"  Lattice parameters: a={struct['lattice_params']['a']:.3f} Å, "
              f"b={struct['lattice_params']['b']:.3f} Å, "
              f"c={struct['lattice_params']['c']:.3f} Å")
    else:
        print("\nWARNING: Not all initial structures are identical!")
        for i, struct in enumerate(details):
            print(f"\nStructure {i+1}:")
            print(f"  Species: {struct['species_count']}")
            print(f"  Number of atoms: {struct['num_atoms']}")
            print(f"  Volume: {struct['volume']:.2f} Å³")
            print(f"  Lattice angles: α={struct['lattice_angles']['alpha']:.1f}°, "
                  f"β={struct['lattice_angles']['beta']:.1f}°, "
                  f"γ={struct['lattice_angles']['gamma']:.1f}°")
            print(f"  Lattice parameters: a={struct['lattice_params']['a']:.3f} Å, "
                  f"b={struct['lattice_params']['b']:.3f} Å, "
                  f"c={struct['lattice_params']['c']:.3f} Å")

def print_detailed_analysis(stats: Dict[str, Any]):
    """Print detailed analysis of the structures."""
    print("\n=== Detailed Analysis ===")
    
    print("\nComposition Distribution:")
    for formula, count in stats["composition_counts"].items():
        print(f"{formula}: {count} structures")
    
    print("\nDetailed Structure Analysis:")
    for i, details in enumerate(stats["structure_details"]):
        print(f"\nStructure {i+1}:")
        print(f"  Composition: {details['composition']}")
        print(f"  Number of atoms: {details['num_atoms']}")
        print(f"  Volume: {details['volume']:.2f} Å³")
        print(f"  Density: {details['density']:.2f} g/cm³")
        print(f"  Lattice angles: α={details['lattice_angles']['alpha']:.1f}°, "
              f"β={details['lattice_angles']['beta']:.1f}°, "
              f"γ={details['lattice_angles']['gamma']:.1f}°")
        print(f"  Lattice parameters: a={details['lattice_params']['a']:.3f} Å, "
              f"b={details['lattice_params']['b']:.3f} Å, "
              f"c={details['lattice_params']['c']:.3f} Å")

def analyze_atom_finalization(trajectory: List[Structure]) -> Tuple[np.ndarray, np.ndarray]:
    """Analyze when atoms reach their final species, working backwards.
    
    Args:
        trajectory: List of structures representing one trajectory
        
    Returns:
        percentage_finalized: Array of percentages of atoms in their final state at each step
        num_atoms: Array of number of atoms at each step
    """
    final_structure = trajectory[-1]
    final_species = [site.specie.symbol for site in final_structure]
    num_final_atoms = len(final_species)
    
    num_steps = len(trajectory)
    percentage_finalized = np.zeros(num_steps)
    num_atoms = np.zeros(num_steps, dtype=int)
    
    # Record number of atoms at each step
    for i, struct in enumerate(trajectory):
        num_atoms[i] = len(struct)
    
    # Work backwards through the trajectory
    for step in range(num_steps-1, -1, -1):
        current_struct = trajectory[step]
        current_species = [site.specie.symbol for site in current_struct]
        current_num_atoms = len(current_species)
        
        if current_num_atoms <= num_final_atoms:
            # Count atoms that match their final species
            matches = sum(1 for i, species in enumerate(current_species)
                        if i < num_final_atoms and species == final_species[i])
            percentage_finalized[step] = (matches / num_final_atoms) * 100
        else:
            # If we have extra atoms, count them as non-matches
            matches = sum(1 for i, species in enumerate(current_species[:num_final_atoms])
                        if species == final_species[i])
            percentage_finalized[step] = (matches / current_num_atoms) * 100
            
    return percentage_finalized, num_atoms

def analyze_atom_finalization_by_element(trajectory: List[Structure]) -> Dict[str, np.ndarray]:
    """Analyze when atoms of each element reach their final state.
    
    Args:
        trajectory: List of structures representing one trajectory
        
    Returns:
        Dict mapping element symbols to their finalization percentages at each step
    """
    final_structure = trajectory[-1]
    final_species = [site.specie.symbol for site in final_structure]
    element_indices = defaultdict(list)
    
    # Group indices by final element
    for i, symbol in enumerate(final_species):
        element_indices[symbol].append(i)
    
    num_steps = len(trajectory)
    element_percentages = {}
    
    # Calculate finalization percentage for each element
    for element, indices in element_indices.items():
        percentages = np.zeros(num_steps)
        num_atoms_of_element = len(indices)
        
        for step in range(num_steps-1, -1, -1):
            current_struct = trajectory[step]
            current_species = [site.specie.symbol for site in current_struct]
            
            if len(current_species) >= len(final_species):
                matches = sum(1 for i in indices 
                            if current_species[i] == final_species[i])
                percentages[step] = (matches / num_atoms_of_element) * 100
            else:
                # Handle case where some atoms are missing
                valid_indices = [i for i in indices if i < len(current_species)]
                if valid_indices:
                    matches = sum(1 for i in valid_indices 
                                if current_species[i] == final_species[i])
                    percentages[step] = (matches / num_atoms_of_element) * 100
                else:
                    percentages[step] = 0
                    
        element_percentages[element] = percentages
        
    return element_percentages

def analyze_lattice_evolution(trajectory: List[Structure]) -> Dict[str, np.ndarray]:
    """Analyze the evolution of lattice parameters.
    
    Returns:
        Dict containing arrays for angles, lengths, and volume
    """
    num_steps = len(trajectory)
    evolution = {
        'alpha': np.zeros(num_steps),
        'beta': np.zeros(num_steps),
        'gamma': np.zeros(num_steps),
        'a': np.zeros(num_steps),
        'b': np.zeros(num_steps),
        'c': np.zeros(num_steps),
        'volume': np.zeros(num_steps)
    }
    
    for i, struct in enumerate(trajectory):
        evolution['alpha'][i] = struct.lattice.alpha
        evolution['beta'][i] = struct.lattice.beta
        evolution['gamma'][i] = struct.lattice.gamma
        evolution['a'][i] = struct.lattice.a
        evolution['b'][i] = struct.lattice.b
        evolution['c'][i] = struct.lattice.c
        evolution['volume'][i] = struct.volume
        
    return evolution

def plot_finalization_curves(trajectories: List[List[Structure]], output_dir: Path):
    """Create enhanced plots for atom finalization analysis.
    Only uses the states after corrector steps and scales x-axis to match N=1000 steps."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Prepare figure for percentage curves
    plt.figure(figsize=(12, 6))
    
    # Plot each trajectory's finalization curve
    for i, traj in enumerate(trajectories):
        # Take every other point (post-corrector states)
        post_corrector_traj = traj[1::2]  # Start from index 1, step by 2
        percentage_finalized, _ = analyze_atom_finalization(post_corrector_traj)
        
        # Scale steps to match N=1000
        steps = np.linspace(0, 1000, len(percentage_finalized))
        plt.plot(steps, percentage_finalized, alpha=0.15, color='gray')
    
    # Calculate and plot mean with error bands
    all_percentages = []
    for traj in trajectories:
        post_corrector_traj = traj[1::2]  # Start from index 1, step by 2
        percentage_finalized, _ = analyze_atom_finalization(post_corrector_traj)
        all_percentages.append(percentage_finalized)
    
    max_length = max(len(p) for p in all_percentages)
    padded_percentages = [np.pad(p, (0, max_length - len(p)), 'edge') for p in all_percentages]
    mean_percentage = np.mean(padded_percentages, axis=0)
    std_percentage = np.std(padded_percentages, axis=0)
    
    steps = np.linspace(0, 1000, max_length)
    plt.plot(steps, mean_percentage, 'k-', linewidth=2, label='Mean')
    plt.fill_between(steps, mean_percentage - std_percentage, 
                    mean_percentage + std_percentage, color='k', alpha=0.2)
    
    plt.xlabel('Diffusion Step')
    plt.ylabel('Percentage of Atoms in Final State')
    plt.title('Atom Finalization Progress')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plots_dir / 'finalization_curves.png', bbox_inches='tight')
    plt.close()
    
    # Plot number of atoms - individual curves
    plt.figure(figsize=(12, 6))
    for i, traj in enumerate(trajectories):
        post_corrector_traj = traj[1::2]  # Start from index 1, step by 2
        _, num_atoms = analyze_atom_finalization(post_corrector_traj)
        steps = np.linspace(0, 1000, len(num_atoms))
        plt.plot(steps, num_atoms, alpha=0.5, label=f'Trajectory {i+1}')
    
    plt.xlabel('Diffusion Step')
    plt.ylabel('Number of Atoms')
    plt.title('Number of Atoms Throughout Diffusion')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plots_dir / 'num_atoms_evolution.png', bbox_inches='tight')
    plt.close()
    
    # Plot per-element finalization curves
    plt.figure(figsize=(12, 6))
    
    # Get per-element data for all trajectories
    all_element_percentages = []
    for traj in trajectories:
        post_corrector_traj = traj[1::2]  # Start from index 1, step by 2
        all_element_percentages.append(analyze_atom_finalization_by_element(post_corrector_traj))
    
    # Find all unique elements
    all_elements = set()
    for percentages in all_element_percentages:
        all_elements.update(percentages.keys())
    
    # Plot mean curve for each element
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_elements)))
    for element, color in zip(sorted(all_elements), colors):
        # Gather data for this element from all trajectories
        element_data = []
        for traj_data in all_element_percentages:
            if element in traj_data:
                element_data.append(traj_data[element])
        
        # Calculate mean and std
        padded_data = [np.pad(d, (0, max_length - len(d)), 'edge') for d in element_data]
        mean_data = np.mean(padded_data, axis=0)
        std_data = np.std(padded_data, axis=0)
        
        steps = np.linspace(0, 1000, max_length)
        plt.plot(steps, mean_data, '-', color=color, linewidth=2, label=element)
        plt.fill_between(steps, mean_data - std_data, mean_data + std_data,
                        color=color, alpha=0.2)
    
    plt.xlabel('Diffusion Step')
    plt.ylabel('Percentage in Final State')
    plt.title('Element-Specific Finalization Progress')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plots_dir / 'element_finalization_curves.png', bbox_inches='tight')
    plt.close()

    # Plot lattice evolution - means with error bands
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gather lattice data for post-corrector states
    all_lattice_data = []
    for traj in trajectories:
        post_corrector_traj = traj[1::2]  # Start from index 1, step by 2
        all_lattice_data.append(analyze_lattice_evolution(post_corrector_traj))
    
    # Plot angles
    ax1.set_title('Lattice Angles')
    for angle in ['alpha', 'beta', 'gamma']:
        angle_data = [data[angle] for data in all_lattice_data]
        padded_data = [np.pad(d, (0, max_length - len(d)), 'edge') for d in angle_data]
        mean_data = np.mean(padded_data, axis=0)
        std_data = np.std(padded_data, axis=0)
        
        steps = np.linspace(0, 1000, len(mean_data))
        line = ax1.plot(steps, mean_data, '-', linewidth=2, label=f'${angle}$')[0]
        ax1.fill_between(steps, mean_data - std_data, mean_data + std_data,
                        color=line.get_color(), alpha=0.2)
    ax1.set_ylabel('Angle (degrees)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot lengths
    ax2.set_title('Lattice Vector Lengths')
    for length, label in zip(['a', 'b', 'c'], ['a', 'b', 'c']):
        length_data = [data[length] for data in all_lattice_data]
        padded_data = [np.pad(d, (0, max_length - len(d)), 'edge') for d in length_data]
        mean_data = np.mean(padded_data, axis=0)
        std_data = np.std(padded_data, axis=0)
        
        steps = np.linspace(0, 1000, len(mean_data))
        line = ax2.plot(steps, mean_data, '-', linewidth=2, label=label)[0]
        ax2.fill_between(steps, mean_data - std_data, mean_data + std_data,
                        color=line.get_color(), alpha=0.2)
    ax2.set_ylabel('Length (Å)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot volume
    ax3.set_title('Unit Cell Volume')
    volume_data = [data['volume'] for data in all_lattice_data]
    padded_data = [np.pad(d, (0, max_length - len(d)), 'edge') for d in volume_data]
    mean_data = np.mean(padded_data, axis=0)
    std_data = np.std(padded_data, axis=0)
    
    steps = np.linspace(0, 1000, len(mean_data))
    ax3.plot(steps, mean_data, 'k-', linewidth=2)
    ax3.fill_between(steps, mean_data - std_data, mean_data + std_data,
                    color='k', alpha=0.2)
    ax3.set_ylabel('Volume (Å³)')
    ax3.grid(True)
    
    # Remove the fourth subplot
    fig.delaxes(ax4)
    
    # Add common x-label
    fig.text(0.5, 0.04, 'Diffusion Step', ha='center')
    plt.tight_layout()
    plt.savefig(plots_dir / 'lattice_evolution_mean.png', bbox_inches='tight')
    plt.close()
    
    # Plot lattice evolution - individual curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    window_size = 100  # Reduced window size since we have half as many points now
    
    # Plot angles
    ax1.set_title('Lattice Angles')
    angle_colors = {'alpha': 'red', 'beta': 'green', 'gamma': 'blue'}
    for angle, color in angle_colors.items():
        for data in all_lattice_data:
            # Apply smoothing with window size 100
            smoothed_data = pd.Series(data[angle]).rolling(window=window_size, min_periods=1).mean()
            steps = np.linspace(0, 1000, len(smoothed_data))
            ax1.plot(steps, smoothed_data, '-', color=color, alpha=0.3)
        # Add one line with label
        ax1.plot([], [], '-', color=color, label=f'${angle}$')
    ax1.set_ylabel('Angle (degrees)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot lengths
    ax2.set_title('Lattice Vector Lengths')
    length_colors = {'a': 'red', 'b': 'green', 'c': 'blue'}
    for length, color in length_colors.items():
        for data in all_lattice_data:
            # Apply smoothing with window size 100
            smoothed_data = pd.Series(data[length]).rolling(window=window_size, min_periods=1).mean()
            steps = np.linspace(0, 1000, len(smoothed_data))
            ax2.plot(steps, smoothed_data, '-', color=color, alpha=0.3)
        # Add one line with label
        ax2.plot([], [], '-', color=color, label=length)
    ax2.set_ylabel('Length (Å)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot volume
    ax3.set_title('Unit Cell Volume')
    for data in all_lattice_data:
        # Apply smoothing with window size 100
        smoothed_data = pd.Series(data['volume']).rolling(window=window_size, min_periods=1).mean()
        steps = np.linspace(0, 1000, len(smoothed_data))
        ax3.plot(steps, smoothed_data, '-', color='gray', alpha=0.3)
    ax3.set_ylabel('Volume (Å³)')
    ax3.grid(True)
    
    # Remove the fourth subplot
    fig.delaxes(ax4)
    
    # Add common x-label
    fig.text(0.5, 0.04, 'Diffusion Step', ha='center')
    plt.tight_layout()
    plt.savefig(plots_dir / 'lattice_evolution_individual.png', bbox_inches='tight')
    plt.close()

def analyze_lattice_type(structure: Structure) -> Dict[str, Any]:
    """Analyze the lattice type and symmetry of a structure.
    
    Args:
        structure: Pymatgen Structure object
        
    Returns:
        Dictionary containing lattice analysis results
    """
    # Create symmetry analyzer
    sga = SpacegroupAnalyzer(structure, symprec=0.3)
    
    # Get conventional standard structure
    try:
        conv_structure = sga.get_conventional_standard_structure()
    except:
        conv_structure = structure
    
    # Create VoronoiNN analyzer for coordination numbers
    vnn = VoronoiNN()
    
    # Calculate average coordination number
    try:
        coord_numbers = [vnn.get_cn(structure, i) for i in range(len(structure))]
        avg_cn = np.mean(coord_numbers)
        cn_counts = Counter(coord_numbers)
    except:
        avg_cn = None
        cn_counts = Counter()
    
    analysis = {
        'crystal_system': sga.get_crystal_system(),
        'lattice_type': sga.get_lattice_type(),
        'space_group_symbol': sga.get_space_group_symbol(),
        'space_group_number': sga.get_space_group_number(),
        'point_group': sga.get_point_group_symbol(),
        'is_symmetric': sga.is_laue(),
        'avg_coordination_number': avg_cn,
        'coordination_number_distribution': dict(cn_counts),
        'volume_per_atom': structure.volume / len(structure),
        'density': structure.density,
        'lattice_params': {
            'a': structure.lattice.a,
            'b': structure.lattice.b,
            'c': structure.lattice.c,
            'alpha': structure.lattice.alpha,
            'beta': structure.lattice.beta,
            'gamma': structure.lattice.gamma,
        }
    }
    
    return analysis

def analyze_final_structures(trajectories: List[List[Structure]]) -> List[Dict[str, Any]]:
    """Analyze the lattice types of final structures from all trajectories."""
    final_structures = [traj[-1] for traj in trajectories]
    analyses = []
    
    for i, struct in enumerate(final_structures):
        try:
            analysis = analyze_lattice_type(struct)
            analysis['trajectory_index'] = i
            analyses.append(analysis)
        except Exception as e:
            print(f"Warning: Could not analyze structure {i}: {str(e)}")
    
    return analyses

def print_lattice_analysis(analyses: List[Dict[str, Any]]):
    """Print summary of lattice analyses."""
    print("\n=== Final Structure Analysis ===")
    
    # Count crystal systems
    crystal_systems = Counter(a['crystal_system'] for a in analyses)
    print("\nCrystal Systems:")
    for system, count in crystal_systems.most_common():
        print(f"  {system}: {count} structures")
    
    # Count space groups
    space_groups = Counter(a['space_group_symbol'] for a in analyses)
    print("\nSpace Groups:")
    for group, count in space_groups.most_common():
        print(f"  {group}: {count} structures")
    
    # Coordination number statistics
    print("\nCoordination Numbers:")
    all_cns = []
    for analysis in analyses:
        if analysis['avg_coordination_number'] is not None:
            all_cns.append(analysis['avg_coordination_number'])
    if all_cns:
        print(f"  Average: {np.mean(all_cns):.2f} ± {np.std(all_cns):.2f}")
        print("\nCoordination Number Distribution:")
        cn_dist = Counter()
        for analysis in analyses:
            cn_dist.update(analysis['coordination_number_distribution'])
        for cn, count in sorted(cn_dist.items()):
            print(f"  CN={cn}: {count} atoms")
    
    # Volume per atom statistics
    volumes = [a['volume_per_atom'] for a in analyses]
    print(f"\nVolume per atom: {np.mean(volumes):.2f} ± {np.std(volumes):.2f} Å³")
    
    # Density statistics
    densities = [a['density'] for a in analyses]
    print(f"Density: {np.mean(densities):.2f} ± {np.std(densities):.2f} g/cm³")

def plot_lattice_analysis(analyses: List[Dict[str, Any]], output_dir: Path):
    """Create visualization plots for lattice analysis."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot distribution of crystal systems
    plt.figure(figsize=(10, 6))
    crystal_systems = Counter(a['crystal_system'] for a in analyses)
    systems, counts = zip(*crystal_systems.most_common())
    plt.bar(systems, counts)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Structures')
    plt.title('Distribution of Crystal Systems')
    plt.tight_layout()
    plt.savefig(plots_dir / 'crystal_systems.png')
    plt.close()
    
    # Plot volume per atom vs density
    plt.figure(figsize=(8, 8))
    volumes = [a['volume_per_atom'] for a in analyses]
    densities = [a['density'] for a in analyses]
    plt.scatter(volumes, densities, alpha=0.6)
    plt.xlabel('Volume per Atom (Å³)')
    plt.ylabel('Density (g/cm³)')
    plt.title('Volume per Atom vs Density')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'volume_density.png')
    plt.close()
    
    # Plot lattice parameter distributions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot lengths
    lengths = np.array([[a['lattice_params'][p] for p in ['a', 'b', 'c']] for a in analyses])
    ax1.boxplot(lengths, labels=['a', 'b', 'c'])
    ax1.set_ylabel('Length (Å)')
    ax1.set_title('Lattice Vector Lengths')
    ax1.grid(True)
    
    # Plot angles
    angles = np.array([[a['lattice_params'][p] for p in ['alpha', 'beta', 'gamma']] for a in analyses])
    ax2.boxplot(angles, labels=['α', 'β', 'γ'])
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Lattice Angles')
    ax2.grid(True)
    
    # Plot coordination numbers
    cns = [a['avg_coordination_number'] for a in analyses if a['avg_coordination_number'] is not None]
    if cns:
        ax3.hist(cns, bins=20, edgecolor='black')
        ax3.set_xlabel('Coordination Number')
        ax3.set_ylabel('Count')
        ax3.set_title('Average Coordination Numbers')
        ax3.grid(True)
    
    # Plot space group numbers
    space_groups = [a['space_group_number'] for a in analyses]
    ax4.hist(space_groups, bins=30, edgecolor='black')
    ax4.set_xlabel('Space Group Number')
    ax4.set_ylabel('Count')
    ax4.set_title('Space Group Distribution')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'lattice_parameters.png')
    plt.close()

def plot_individual_trajectories(trajectories: List[List[Structure]], output_dir: Path):
    """Create detailed plots for each individual trajectory.
    
    Args:
        trajectories: List of trajectories, where each trajectory is a list of Structure objects
        output_dir: Base output directory
    """
    # Create a subdirectory for individual trajectory plots
    traj_plots_dir = output_dir / "plots" / "individual_trajectories"
    traj_plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Define crystal system order for consistent y-axis
    crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 
                      'trigonal', 'hexagonal', 'cubic']
    crystal_system_to_num = {sys: i for i, sys in enumerate(crystal_systems)}
    
    window_size = 200  # Match the window size from other plots
    
    for traj_idx, trajectory in enumerate(trajectories):
        # Create figure with 4 subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        steps = np.arange(len(trajectory))
        
        # Plot 1: Atomic numbers for each atom
        atomic_numbers = []
        for struct in trajectory:
            atomic_numbers.append([site.specie.Z for site in struct])
        
        # Convert to numpy array with padding
        max_atoms = max(len(nums) for nums in atomic_numbers)
        padded_numbers = np.full((len(trajectory), max_atoms), np.nan)
        for i, nums in enumerate(atomic_numbers):
            padded_numbers[i, :len(nums)] = nums
        
        # Plot each atom's trajectory
        for atom_idx in range(max_atoms):
            atom_trajectory = padded_numbers[:, atom_idx]
            # Only plot if the atom exists at some point
            if not np.all(np.isnan(atom_trajectory)):
                ax1.plot(steps, atom_trajectory, '-', alpha=0.5, linewidth=2.5)
        
        ax1.set_ylabel('Atomic Number')
        ax1.set_title(f'Atomic Numbers Evolution - Trajectory {traj_idx + 1}')
        ax1.grid(True)
        
        # Plot 2: Lattice vector lengths
        lengths = {'a': [], 'b': [], 'c': []}
        for struct in trajectory:
            lengths['a'].append(struct.lattice.a)
            lengths['b'].append(struct.lattice.b)
            lengths['c'].append(struct.lattice.c)
        
        for label, values in lengths.items():
            # Apply smoothing
            smoothed = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
            ax2.plot(steps, smoothed, '-', label=f'Vector {label}', linewidth=2)
        
        ax2.set_ylabel('Length (Å)')
        ax2.set_title('Lattice Vector Lengths')
        ax2.grid(True)
        ax2.legend()
        
        # Plot 3: Lattice angles
        angles = {'alpha': [], 'beta': [], 'gamma': []}
        for struct in trajectory:
            angles['alpha'].append(struct.lattice.alpha)
            angles['beta'].append(struct.lattice.beta)
            angles['gamma'].append(struct.lattice.gamma)
        
        for label, values in angles.items():
            # Apply smoothing
            smoothed = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
            ax3.plot(steps, smoothed, '-', label=f'${label}$', linewidth=2)
        
        ax3.set_ylabel('Angle (degrees)')
        ax3.set_title('Lattice Angles')
        ax3.grid(True)
        ax3.legend()
        
        # Plot 4: Crystal System Evolution
        crystal_system_nums = []
        for struct in trajectory:
            try:
                sga = SpacegroupAnalyzer(struct, symprec=0.5)  # Using larger symprec for more lenient classification
                crystal_system = sga.get_crystal_system()
                crystal_system_nums.append(crystal_system_to_num[crystal_system.lower()])
            except:
                crystal_system_nums.append(np.nan)
        
        # Plot crystal system evolution
        ax4.plot(steps, crystal_system_nums, 'k.-', linewidth=1, markersize=3)
        ax4.set_yticks(range(len(crystal_systems)))
        ax4.set_yticklabels(crystal_systems)
        ax4.set_xlabel('Diffusion Step')
        ax4.set_ylabel('Crystal System')
        ax4.set_title('Crystal System Evolution')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(traj_plots_dir / f'trajectory_{traj_idx + 1}.png', bbox_inches='tight', dpi=300)
        plt.close()

def analyze_species_changes(trajectories: List[List[Structure]], output_dir: Path):
    """Analyze how many times each atom changes its species in each trajectory.
    Creates separate plots for each number of changes, showing the distribution
    of unit cell sizes for atoms with that many changes.
    
    Args:
        trajectories: List of trajectories, where each trajectory is a list of Structure objects
        output_dir: Base output directory
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Dictionary to store unit cell sizes for each number of changes
    sizes_by_changes = defaultdict(list)
    
    for trajectory in trajectories:
        # Get number of atoms in final structure
        num_atoms = len(trajectory[-1])
        
        # Track species changes for each atom position
        atomic_numbers = []
        for struct in trajectory:
            atomic_numbers.append([site.specie.Z for site in struct])
            
        # Convert to numpy array with padding
        max_atoms = max(len(nums) for nums in atomic_numbers)
        padded_numbers = np.full((len(trajectory), max_atoms), np.nan)
        for i, nums in enumerate(atomic_numbers):
            padded_numbers[i, :len(nums)] = nums
            
        # Count species changes for each atom
        for atom_idx in range(max_atoms):
            atom_trajectory = padded_numbers[:, atom_idx]
            # Only count if the atom exists at some point
            if not np.all(np.isnan(atom_trajectory)):
                # Count number of changes (excluding nan transitions)
                changes = 0
                prev_species = None
                for species in atom_trajectory:
                    if not np.isnan(species):
                        if prev_species is not None and species != prev_species:
                            changes += 1
                        prev_species = species
                sizes_by_changes[changes].append(num_atoms)
    
    # Get the maximum number of changes
    max_changes = max(sizes_by_changes.keys()) if sizes_by_changes else 0
    
    if max_changes == 0:
        print("No species changes found in any trajectory")
        return
    
    # Create subplots, one for each number of changes
    num_plots = max_changes + 1  # Include 0 changes
    fig_height = 4 * num_plots  # Scale height by number of plots
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, fig_height))
    if num_plots == 1:
        axes = [axes]  # Make axes iterable if only one subplot
    
    # Get the range for unit cell sizes for consistent binning across plots
    all_sizes = [size for sizes in sizes_by_changes.values() for size in sizes]
    min_size = min(all_sizes)
    max_size = max(all_sizes)
    bins = range(min_size - 1, max_size + 2)  # Add padding of 1 on each side
    
    for num_changes in range(num_plots):
        ax = axes[num_changes]
        sizes = sizes_by_changes.get(num_changes, [])
        
        if sizes:
            # Create histogram with integer bins
            counts, edges, _ = ax.hist(sizes, bins=bins, align='left',
                                     color='skyblue', edgecolor='black',
                                     linewidth=1, alpha=0.7)
            
            # Add count labels on top of bars
            for i, count in enumerate(counts):
                if count > 0:  # Only label non-zero bars
                    ax.text(edges[i] + 0.5, count, str(int(count)),
                           ha='center', va='bottom')
        
        ax.set_xlabel('Number of Atoms in Unit Cell')
        ax.set_ylabel('Number of Atoms')
        ax.set_title(f'Distribution of Unit Cell Sizes for Atoms with {num_changes} Species Changes')
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks to integers
        ax.set_xticks(range(min_size, max_size + 1))
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'species_changes_by_size.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_num_atoms_distribution(output_dir: Path):
    """Plot the number of atoms distribution from NUM_ATOMS_DISTRIBUTIONS."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Get the ALEX_MP_20 distribution
    dist = NUM_ATOMS_DISTRIBUTIONS["ALEX_MP_20"]
    
    # Create bar plot
    nums = list(dist.keys())
    probs = list(dist.values())
    
    plt.bar(nums, probs, color='skyblue', edgecolor='black', linewidth=1)
    
    # Add percentage labels on top of bars
    for i, prob in enumerate(probs):
        percentage = prob * 100
        plt.text(nums[i], prob, f'{percentage:.1f}%', 
                ha='center', va='bottom')
    
    plt.xlabel('Number of Atoms in Unit Cell')
    plt.ylabel('Probability')
    plt.title('Number of Atoms Distribution (ALEX_MP_20)')
    plt.grid(True, alpha=0.3)
    
    # Set x-ticks to integers
    plt.xticks(nums)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'num_atoms_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def analyze_bond_lengths(structures: List[Structure], output_dir: Path):
    """Analyze and plot bond lengths in the final structures, grouped by bond types.
    
    Args:
        structures: List of pymatgen Structure objects
        output_dir: Base output directory
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Dictionary to store bond lengths by bond type
    bond_lengths = defaultdict(list)
    
    # Use VoronoiNN to find neighbors
    vnn = VoronoiNN()
    
    for struct in tqdm(structures, desc="Analyzing bond lengths"):
        # Get all neighbors for each site
        for i, site in enumerate(struct):
            try:
                neighbors = vnn.get_nn_info(struct, i)
                site_element = site.specie.symbol
                
                for neighbor in neighbors:
                    neighbor_element = neighbor["site"].specie.symbol
                    # Sort elements alphabetically to ensure consistent bond naming
                    bond_type = tuple(sorted([site_element, neighbor_element]))
                    # Calculate distance between sites
                    distance = site.distance(neighbor["site"])
                    bond_lengths[bond_type].append(distance)
            except Exception as e:
                print(f"Warning: Error analyzing site {i}: {str(e)}")
                continue
    
    # Create histogram plot
    num_bond_types = len(bond_lengths)
    if num_bond_types == 0:
        print("No bonds found in structures")
        return
    
    # Calculate number of rows and columns for subplots
    num_cols = min(3, num_bond_types)
    num_rows = (num_bond_types + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
    if num_bond_types == 1:
        axes = np.array([axes])  # Make it indexable if only one subplot
    axes = axes.flatten()
    
    # Plot histograms for each bond type
    for idx, (bond_type, distances) in enumerate(sorted(bond_lengths.items())):
        ax = axes[idx]
        
        # Create histogram
        counts, bins, _ = ax.hist(distances, bins=50, density=True, alpha=0.7,
                                color='skyblue', edgecolor='black')
        
        # Add labels and title
        bond_label = f"{bond_type[0]}-{bond_type[1]}"
        ax.set_xlabel("Bond Length (Å)")
        ax.set_ylabel("Density")
        ax.set_title(f"{bond_label} Bonds\n(n={len(distances)})")
        
        # Add mean and std
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        ax.axvline(mean_dist, color='red', linestyle='--', alpha=0.8)
        ax.text(0.98, 0.95, f"Mean: {mean_dist:.3f} Å\nStd: {std_dist:.3f} Å",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots if any
    for idx in range(num_bond_types, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'bond_lengths.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\nBond Length Statistics:")
    for bond_type, distances in sorted(bond_lengths.items()):
        bond_label = f"{bond_type[0]}-{bond_type[1]}"
        print(f"\n{bond_label} Bonds:")
        print(f"  Count: {len(distances)}")
        print(f"  Mean: {np.mean(distances):.3f} Å")
        print(f"  Std: {np.std(distances):.3f} Å")
        print(f"  Min: {np.min(distances):.3f} Å")
        print(f"  Max: {np.max(distances):.3f} Å")

def main():
    output_dir = Path("generated_structures")
    
    print("\nLoading trajectories...")
    trajectories = load_trajectories(output_dir)
    print(f"Loaded {len(trajectories)} trajectories")
    
    print("\nAnalyzing final structures...")
    final_structures = [traj[-1] for traj in trajectories]
    lattice_analyses = analyze_final_structures(trajectories)
    print_lattice_analysis(lattice_analyses)
    
    print("\nAnalyzing bond lengths...")
    analyze_bond_lengths(final_structures, output_dir)
    
    print("\nCreating analysis plots...")
    plot_finalization_curves(trajectories, output_dir)
    plot_lattice_analysis(lattice_analyses, output_dir)
    plot_num_atoms_distribution(output_dir)
    
    print("\nCreating individual trajectory plots...")
    plot_individual_trajectories(trajectories, output_dir)
    
    print("\nAnalyzing species changes...")
    analyze_species_changes(trajectories, output_dir)
    
    print(f"Plots saved in {output_dir}/plots/")

if __name__ == "__main__":
    main()
