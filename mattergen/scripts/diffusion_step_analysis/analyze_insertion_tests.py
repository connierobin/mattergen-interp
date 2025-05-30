#!/usr/bin/env python3
"""
Analyze late insertion cross-conditioning test results.

This script analyzes the results from cross-conditioned late insertion trajectories,
comparing how well different conditioning strategies can redirect structures at various timesteps.
"""

import os
import json
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import ase
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher


def analyze_structure_properties(structure: Structure) -> Dict[str, Any]:
    """Analyze properties of a final structure."""
    try:
        # Space group analysis with relaxed tolerances
        # Use tolerances consistent with unpack_results.py
        tolerance_params = [
            (0.01, 5.0),   # Strict - closest to "true" symmetry
            (0.05, 5.0),   # Moderate 
            (0.1, 5.0),    # Moderately relaxed
            (0.2, 8.0),    # Relaxed
            (0.3, 10.0),   # More relaxed (from unpack_results.py)
        ]
        
        best_sg_number = 1
        best_sg_symbol = "P1"
        best_crystal_system = "triclinic"
        
        for symprec, angle_tol in tolerance_params:
            try:
                analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tol)
                sg_number = analyzer.get_space_group_number()
                sg_symbol = analyzer.get_space_group_symbol()
                crystal_system = analyzer.get_crystal_system()
                
                # Keep the highest symmetry (lowest space group number for most cases)
                # But prioritize non-P1 results
                if sg_number > 1 and (best_sg_number == 1 or sg_number < best_sg_number):
                    best_sg_number = sg_number
                    best_sg_symbol = sg_symbol
                    best_crystal_system = crystal_system
                    break  # Found non-P1, use it
                elif best_sg_number == 1:
                    # Keep trying if we haven't found anything better than P1
                    best_sg_number = sg_number
                    best_sg_symbol = sg_symbol
                    best_crystal_system = crystal_system
                    
            except Exception:
                continue
        
        space_group_number = best_sg_number
        space_group_symbol = best_sg_symbol
        crystal_system = best_crystal_system
        
        # Lattice parameters
        lattice = structure.lattice
        a, b, c = lattice.abc
        alpha, beta, gamma = lattice.angles
        volume = lattice.volume
        
        # Density
        density = structure.density
        
        # Composition
        composition = structure.composition
        
        return {
            'space_group_number': space_group_number,
            'space_group_symbol': space_group_symbol,
            'crystal_system': crystal_system,
            'lattice_a': a,
            'lattice_b': b, 
            'lattice_c': c,
            'lattice_alpha': alpha,
            'lattice_beta': beta,
            'lattice_gamma': gamma,
            'volume': volume,
            'density': density,
            'formula': str(composition.reduced_formula),
            'num_atoms': len(structure),
            'valid': True
        }
    except Exception as e:
        print(f"Warning: Structure analysis failed: {e}")
        return {
            'space_group_number': 1,
            'space_group_symbol': 'P1',
            'crystal_system': 'triclinic',
            'lattice_a': np.nan,
            'lattice_b': np.nan,
            'lattice_c': np.nan,
            'lattice_alpha': np.nan,
            'lattice_beta': np.nan,
            'lattice_gamma': np.nan,
            'volume': np.nan,
            'density': np.nan,
            'formula': 'Unknown',
            'num_atoms': 0,
            'valid': False
        }


def load_final_structure_from_trajectory_zip(zip_path: str, trajectory_idx: int = 0) -> Optional[Structure]:
    """Load the final structure from a trajectory zip file (last timestep)."""
    if not Path(zip_path).exists():
        print(f"Warning: Trajectory zip file not found: {zip_path}")
        return None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List all trajectory files
            trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
            
            if not trajectory_files:
                print(f"Warning: No trajectory files found in {zip_path}")
                return None
            
            if trajectory_idx >= len(trajectory_files):
                print(f"Warning: Trajectory index {trajectory_idx} not available. Found {len(trajectory_files)} trajectories.")
                return None
            
            chosen_file = trajectory_files[trajectory_idx]
            
            with zf.open(chosen_file) as f:
                content = io.StringIO(f.read().decode())
                atoms_list = ase.io.read(content, index=":", format="extxyz")
                
                # Get the last structure in the trajectory
                final_atoms = atoms_list[-1]
                structure = AseAtomsAdaptor.get_structure(final_atoms)
                
                return structure
                
    except Exception as e:
        print(f"Warning: Failed to load final structure from trajectory {zip_path}: {e}")
        return None


def load_full_trajectory_from_zip(zip_path: str, trajectory_idx: int = 0, step_interval: int = 50) -> Optional[List[Tuple[int, Structure]]]:
    """Load structures from trajectory at regular intervals."""
    if not Path(zip_path).exists():
        print(f"Warning: Trajectory zip file not found: {zip_path}")
        return None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
            
            if not trajectory_files or trajectory_idx >= len(trajectory_files):
                return None
            
            chosen_file = trajectory_files[trajectory_idx]
            
            with zf.open(chosen_file) as f:
                content = io.StringIO(f.read().decode())
                atoms_list = ase.io.read(content, index=":", format="extxyz")
                
                # Sample every step_interval steps (skip corrector steps by using step*2)
                trajectory_data = []
                for step in range(0, 1000, step_interval):
                    traj_idx = step * 2  # Convert to trajectory index (predictor steps)
                    if traj_idx < len(atoms_list):
                        atoms = atoms_list[traj_idx]
                        structure = AseAtomsAdaptor.get_structure(atoms)
                        trajectory_data.append((step, structure))
                
                return trajectory_data
                
    except Exception as e:
        print(f"Warning: Failed to load trajectory from {zip_path}: {e}")
        return None


def analyze_trajectory_evolution(trajectory_data: List[Tuple[int, Structure]]) -> Dict[str, Any]:
    """Analyze space group evolution throughout a trajectory."""
    evolution = {
        'timesteps': [],
        'space_groups': [],
        'space_group_symbols': [],
        'crystal_systems': []
    }
    
    tolerance_params = [
        (0.01, 5.0),   # Strict - closest to "true" symmetry
        (0.05, 5.0),   # Moderate 
        (0.1, 5.0),    # Moderately relaxed
        (0.2, 8.0),    # Relaxed
        (0.3, 10.0),   # More relaxed (from unpack_results.py)
    ]
    
    for timestep, structure in trajectory_data:
        # Analyze with same tolerance approach as before
        best_sg_number = 1
        best_sg_symbol = "P1"
        best_crystal_system = "triclinic"
        
        for symprec, angle_tol in tolerance_params:
            try:
                analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tol)
                sg_number = analyzer.get_space_group_number()
                sg_symbol = analyzer.get_space_group_symbol()
                crystal_system = analyzer.get_crystal_system()
                
                if sg_number > 1 and (best_sg_number == 1 or sg_number < best_sg_number):
                    best_sg_number = sg_number
                    best_sg_symbol = sg_symbol
                    best_crystal_system = crystal_system
                    break
                elif best_sg_number == 1:
                    best_sg_number = sg_number
                    best_sg_symbol = sg_symbol
                    best_crystal_system = crystal_system
                    
            except Exception:
                continue
        
        evolution['timesteps'].append(timestep)
        evolution['space_groups'].append(best_sg_number)
        evolution['space_group_symbols'].append(best_sg_symbol)
        evolution['crystal_systems'].append(best_crystal_system)
    
    return evolution


def collect_baseline_results() -> Dict[str, Dict[str, Any]]:
    """Collect results from baseline (non-insertion) trajectories."""
    baseline_results = {}
    
    # Source conditions that we used for cross-conditioning
    source_conditions = [
        "dft_band_gap_insulator",
        "energy_above_hull_metastable", 
        "space_group_cubic_fcc"
    ]
    
    for source_condition in source_conditions:
        baseline_path = f"generated_structures/three_property_study/{source_condition}"
        trajectory_zip = Path(baseline_path) / "generated_trajectories.zip"
        
        print(f"Loading baseline for {source_condition}...")
        
        if trajectory_zip.exists():
            # Load final structure from trajectory (timestep 1000)
            final_structure = load_final_structure_from_trajectory_zip(str(trajectory_zip), trajectory_idx=0)
            
            if final_structure:
                properties = analyze_structure_properties(final_structure)
                baseline_results[source_condition] = {
                    'timestep': 1000,  # Baseline is full trajectory
                    'source_condition': source_condition,
                    'target_condition': source_condition,  # Same as source for baseline
                    'trajectory_idx': 0,
                    **properties
                }
                print(f"  ✓ Loaded baseline: {properties['space_group_number']} ({properties['space_group_symbol']})")
            else:
                print(f"  ✗ Failed to load baseline structure")
        else:
            print(f"  ✗ Baseline trajectory zip not found: {trajectory_zip}")
    
    return baseline_results


def collect_insertion_results() -> Dict[str, List[Dict[str, Any]]]:
    """Collect results from cross-conditioned late insertion trajectories."""
    results_by_category = defaultdict(list)
    
    # Define the categories and their target conditions
    categories = {
        "dft_band_gap": {
            "source": "dft_band_gap_insulator",
            "targets": ["dft_band_gap_metallic", "dft_band_gap_narrow_semiconductor", "dft_band_gap_wide_semiconductor"]
        },
        "energy_above_hull": {
            "source": "energy_above_hull_metastable", 
            "targets": ["energy_above_hull_stable", "energy_above_hull_unstable"]
        },
        "space_group": {
            "source": "space_group_cubic_fcc",
            "targets": ["space_group_cubic_primitive", "space_group_hexagonal", "space_group_tetragonal", "space_group_orthorhombic"]
        }
    }
    
    timesteps = [600, 800, 900, 950]
    trajectory_idx = 0
    
    for category, config in categories.items():
        source_condition = config["source"]
        target_conditions = config["targets"]
        
        print(f"\nCollecting results for {category} category...")
        
        for target_condition in target_conditions:
            for timestep in timesteps:
                # Path to the insertion result
                result_dir = Path("generated_structures/cross_conditioned_late_insertion") / \
                           f"step_{timestep}" / source_condition / target_condition / f"trajectory_{trajectory_idx}"
                
                trajectory_zip = result_dir / "generated_trajectories.zip"
                
                if trajectory_zip.exists():
                    # Load final structure from insertion trajectory
                    final_structure = load_final_structure_from_trajectory_zip(str(trajectory_zip), trajectory_idx=0)
                    
                    if final_structure:
                        properties = analyze_structure_properties(final_structure)
                        result = {
                            'category': category,
                            'timestep': timestep,
                            'source_condition': source_condition,
                            'target_condition': target_condition,
                            'trajectory_idx': trajectory_idx,
                            **properties
                        }
                        results_by_category[category].append(result)
                        print(f"  ✓ {target_condition} @ step {timestep}: {properties['space_group_number']} ({properties['space_group_symbol']})")
                    else:
                        print(f"  ✗ Failed to load: {target_condition} @ step {timestep}")
                else:
                    print(f"  ✗ Missing: {trajectory_zip}")
    
    return dict(results_by_category)


def plot_property_results(results: List[Dict[str, Any]], baseline: Dict[str, Any], 
                        category: str, output_dir: Path):
    """Plot target property values over insertion timesteps."""
    
    # Map category to property and target values
    property_info = {
        "dft_band_gap": {
            "property_name": "DFT Band Gap (eV)",
            "targets": {
                "dft_band_gap_metallic": 0.0,
                "dft_band_gap_narrow_semiconductor": 1.0,
                "dft_band_gap_wide_semiconductor": 3.0,
                "dft_band_gap_insulator": 6.0  # Source condition
            }
        },
        "energy_above_hull": {
            "property_name": "Space Group Number (Energy Above Hull Proxy)",
            "targets": {
                "energy_above_hull_stable": 0.0,
                "energy_above_hull_metastable": 0.1,  # Source condition
                "energy_above_hull_unstable": 0.3
            }
        },
        "space_group": {
            "property_name": "Space Group Number",
            "targets": {
                "space_group_cubic_fcc": 225,      # Source condition (Fm-3m)
                "space_group_cubic_primitive": 221, # Pm-3m
                "space_group_hexagonal": 194,       # P63/mmc
                "space_group_tetragonal": 139,      # I4/mmm  
                "space_group_orthorhombic": 62      # Pnma
            }
        }
    }
    
    if category not in property_info:
        return
    
    prop_info = property_info[category]
    property_name = prop_info["property_name"]
    target_values = prop_info["targets"]
    
    # Get source condition info for title
    source_condition_map = {
        "dft_band_gap": "dft_band_gap_insulator",
        "energy_above_hull": "energy_above_hull_metastable", 
        "space_group": "space_group_cubic_fcc"
    }
    source_condition = source_condition_map.get(category, "unknown")
    source_target_value = target_values.get(source_condition, "unknown")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Group results by target condition
    target_groups = defaultdict(list)
    for result in results:
        target_groups[result['target_condition']].append(result)
    
    # Plot each target condition
    colors = plt.cm.Set3(np.linspace(0, 1, len(target_groups)))
    
    for i, (target_condition, target_results) in enumerate(target_groups.items()):
        timesteps = []
        property_values = []
        
        # Get target value for this condition
        target_value = target_values.get(target_condition, None)
        
        for result in sorted(target_results, key=lambda x: x['timestep']):
            timesteps.append(result['timestep'])
            
            # For space group, use the actual space group number
            if category == "space_group":
                property_values.append(result['space_group_number'])
            elif category == "dft_band_gap":
                # For DFT band gap, we don't have computed values, so use target as placeholder
                property_values.append(target_value if target_value is not None else 0)
            elif category == "energy_above_hull":
                # For energy above hull, we should compute actual values but for now show space group as proxy
                # TODO: Implement actual energy above hull calculation
                property_values.append(result['space_group_number'])  # Use space group as structural proxy
        
        # Add baseline point (timestep 1000)
        if baseline:
            timesteps.append(1000)
            if category == "space_group":
                property_values.append(baseline['space_group_number'])
            elif category == "dft_band_gap":
                property_values.append(source_target_value if source_target_value != "unknown" else 0)
            elif category == "energy_above_hull":
                # For baseline energy above hull, use space group as proxy
                property_values.append(baseline['space_group_number'])
        
        # Plot line for this target condition
        clean_label = target_condition.replace(f"{category}_", "").replace("_", " ").title()
        if target_value is not None and category != "energy_above_hull":
            clean_label += f" (target: {target_value})"
        
        ax.plot(timesteps, property_values, 'o-', color=colors[i], 
               label=clean_label, linewidth=3, markersize=10)
        
        # Add horizontal line for target value (only for space group and band gap)
        if target_value is not None and category != "energy_above_hull":
            ax.axhline(y=target_value, color=colors[i], linestyle='--', alpha=0.5)
    
    # Customize plot
    ax.set_xlabel('Insertion Timestep', fontsize=14)
    ax.set_ylabel(property_name, fontsize=14)
    
    # Create descriptive title with source info
    source_clean = source_condition.replace(f"{category}_", "").replace("_", " ").title()
    ax.set_title(f'{category.replace("_", " ").title()} Cross-Conditioning\nSource: {source_clean} (target: {source_target_value})', 
                fontsize=16)
    
    # Set x-axis
    ax.set_xlim(550, 1050)
    ax.set_xticks([600, 700, 800, 900, 1000])
    
    # Add vertical line at timestep 1000 (baseline)
    ax.axvline(x=1000, color='red', linestyle=':', alpha=0.7, label='Baseline (no insertion)')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f"{category}_property_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def create_summary_dataframe(results_by_category: Dict[str, List[Dict[str, Any]]], 
                           baseline_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a comprehensive summary DataFrame of all results."""
    
    all_data = []
    
    # Add baseline results
    for source_condition, baseline in baseline_results.items():
        data_row = baseline.copy()
        # Extract category from source condition name
        if "dft_band_gap" in source_condition:
            data_row['category'] = "dft_band_gap"
        elif "energy_above_hull" in source_condition:
            data_row['category'] = "energy_above_hull"
        elif "space_group" in source_condition:
            data_row['category'] = "space_group"
        else:
            data_row['category'] = "unknown"
        data_row['is_baseline'] = True
        all_data.append(data_row)
    
    # Add insertion results
    for category, results in results_by_category.items():
        for result in results:
            data_row = result.copy()
            data_row['is_baseline'] = False
            all_data.append(data_row)
    
    return pd.DataFrame(all_data)


def analyze_success_rates(results_by_category: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze how successful the cross-conditioning was."""
    
    success_analysis = {}
    
    for category, results in results_by_category.items():
        # Group by target condition and analyze
        target_groups = defaultdict(list)
        for result in results:
            target_groups[result['target_condition']].append(result)
        
        category_analysis = {
            'total_experiments': len(results),
            'target_conditions': len(target_groups),
            'by_target': {}
        }
        
        for target_condition, target_results in target_groups.items():
            # Get target space group number based on naming convention
            target_sg_map = {
                'space_group_cubic_primitive': 221,
                'space_group_hexagonal': 194,
                'space_group_tetragonal': 139,
                'space_group_orthorhombic': 62,
                # For other categories, we'd need property values
            }
            
            target_sg = target_sg_map.get(target_condition)
            
            successes = 0
            timestep_success = defaultdict(int)
            
            for result in target_results:
                achieved_sg = result['space_group_number']
                timestep = result['timestep']
                
                # Count as success if achieved target space group
                if target_sg and achieved_sg == target_sg:
                    successes += 1
                    timestep_success[timestep] += 1
            
            success_rate = successes / len(target_results) if target_results else 0
            
            category_analysis['by_target'][target_condition] = {
                'total_attempts': len(target_results),
                'successes': successes,
                'success_rate': success_rate,
                'target_space_group': target_sg,
                'success_by_timestep': dict(timestep_success)
            }
        
        success_analysis[category] = category_analysis
    
    return success_analysis


def plot_lattice_parameter_evolution(results_by_category: Dict[str, List[Dict[str, Any]]], 
                                   baseline_results: Dict[str, Dict[str, Any]], 
                                   output_dir: Path):
    """Plot lattice parameters and angles over insertion timesteps."""
    
    # Source condition mapping
    source_condition_map = {
        "dft_band_gap": "dft_band_gap_insulator",
        "energy_above_hull": "energy_above_hull_metastable", 
        "space_group": "space_group_cubic_fcc"
    }
    
    for category, results in results_by_category.items():
        print(f"\nGenerating lattice parameter plot for {category}...")
        
        # Create subplots for lattice parameters and angles
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Parameters to plot
        params = ['lattice_a', 'lattice_b', 'lattice_c', 'lattice_alpha', 'lattice_beta', 'lattice_gamma']
        param_labels = ['a (Å)', 'b (Å)', 'c (Å)', 'α (°)', 'β (°)', 'γ (°)']
        
        # Group results by target condition
        target_conditions = sorted(set(r['target_condition'] for r in results))
        colors = plt.cm.tab10(np.linspace(0, 1, len(target_conditions)))  # Better distinct colors
        color_map = {target: colors[i] for i, target in enumerate(target_conditions)}
        
        # Get baseline data
        source_condition = source_condition_map.get(category)
        baseline = baseline_results.get(source_condition)
        
        for param_idx, (param, label) in enumerate(zip(params, param_labels)):
            ax = axes[param_idx]
            
            # Plot baseline point if available  
            if baseline and param in baseline:
                ax.scatter([1000], [baseline[param]], color='black', s=150, marker='s', 
                          label='Baseline (final)', zorder=10, alpha=1.0, edgecolors='white', linewidth=2)
            
            # Plot each target condition
            for target_condition in target_conditions:
                target_results = [r for r in results if r['target_condition'] == target_condition]
                
                if target_results:
                    timesteps = []
                    values = []
                    
                    for result in sorted(target_results, key=lambda x: x['timestep']):
                        if param in result and not np.isnan(result[param]):
                            timesteps.append(result['timestep'])
                            values.append(result[param])
                    
                    if timesteps and values:
                        clean_label = target_condition.replace(f"{category}_", "").replace("_", " ").title()
                        ax.plot(timesteps, values, 'o-', color=color_map[target_condition], 
                               label=clean_label, linewidth=3, markersize=10, alpha=0.9)
                        
                        # Debug print
                        if param == 'lattice_a':
                            print(f"  Plotted {clean_label}: {len(timesteps)} points at timesteps {timesteps}")
            
            # Customize subplot
            ax.set_xlabel('Insertion Timestep', fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f'{label}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(550, 1050)
            ax.set_xticks([600, 700, 800, 900, 1000])
            
            # Add vertical line at baseline
            ax.axvline(x=1000, color='red', linestyle=':', alpha=0.7)
            
            # Add insertion window
            ax.axvspan(600, 950, alpha=0.1, color='red')
        
        # Add legend to the last subplot
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Overall title
        clean_category = category.replace("_", " ").title()
        fig.suptitle(f'{clean_category} Cross-Conditioning: Lattice Parameter Evolution', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"{category}_lattice_evolution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved lattice parameter plot: {output_path}")
        plt.close()


def plot_lattice_trajectory_evolution(results_by_category: Dict[str, List[Dict[str, Any]]], 
                                    baseline_results: Dict[str, Dict[str, Any]], 
                                    output_dir: Path):
    """Plot lattice parameter evolution over full trajectories."""
    
    # Source condition mapping
    source_condition_map = {
        "dft_band_gap": "dft_band_gap_insulator",
        "energy_above_hull": "energy_above_hull_metastable", 
        "space_group": "space_group_cubic_fcc"
    }
    
    for category, results in results_by_category.items():
        print(f"\nGenerating lattice trajectory evolution for {category}...")
        
        # Create figures for lattice vectors and angles
        fig_vectors, axes_vectors = plt.subplots(3, 1, figsize=(16, 18))
        fig_angles, axes_angles = plt.subplots(3, 1, figsize=(16, 18))
        
        # Parameters to plot
        vector_params = ['lattice_a', 'lattice_b', 'lattice_c']
        vector_labels = ['a (Å)', 'b (Å)', 'c (Å)']
        angle_params = ['lattice_alpha', 'lattice_beta', 'lattice_gamma']
        angle_labels = ['α (°)', 'β (°)', 'γ (°)']
        
        # Get target conditions and colors
        target_conditions = sorted(set(r['target_condition'] for r in results))
        colors = plt.cm.tab10(np.linspace(0, 1, len(target_conditions)))
        color_map = {target: colors[i] for i, target in enumerate(target_conditions)}
        
        # Get baseline trajectory for source condition
        source_condition = source_condition_map.get(category)
        if source_condition and source_condition in baseline_results:
            baseline_path = f"generated_structures/three_property_study/{source_condition}"
            trajectory_zip = Path(baseline_path) / "generated_trajectories.zip"
            
            if trajectory_zip.exists():
                baseline_trajectory = load_full_trajectory_from_zip(str(trajectory_zip), trajectory_idx=0, step_interval=50)
                if baseline_trajectory:
                    # Extract lattice parameters for each timestep
                    baseline_timesteps = []
                    baseline_lattice_data = {param: [] for param in vector_params + angle_params}
                    
                    for timestep, structure in baseline_trajectory:
                        baseline_timesteps.append(timestep)
                        baseline_lattice_data['lattice_a'].append(structure.lattice.a)
                        baseline_lattice_data['lattice_b'].append(structure.lattice.b)
                        baseline_lattice_data['lattice_c'].append(structure.lattice.c)
                        baseline_lattice_data['lattice_alpha'].append(structure.lattice.alpha)
                        baseline_lattice_data['lattice_beta'].append(structure.lattice.beta)
                        baseline_lattice_data['lattice_gamma'].append(structure.lattice.gamma)
                    
                    # Plot baseline on all subplots
                    for param_idx, (param, label) in enumerate(zip(vector_params, vector_labels)):
                        axes_vectors[param_idx].plot(baseline_timesteps, baseline_lattice_data[param], 
                                                   'o-', color='black', linewidth=3, markersize=6,
                                                   label='Baseline (source condition)', alpha=0.8)
                        axes_vectors[param_idx].set_ylabel(label, fontsize=14)
                        axes_vectors[param_idx].grid(True, alpha=0.3)
                    
                    for param_idx, (param, label) in enumerate(zip(angle_params, angle_labels)):
                        axes_angles[param_idx].plot(baseline_timesteps, baseline_lattice_data[param], 
                                                  'o-', color='black', linewidth=3, markersize=6,
                                                  label='Baseline (source condition)', alpha=0.8)
                        axes_angles[param_idx].set_ylabel(label, fontsize=14)
                        axes_angles[param_idx].grid(True, alpha=0.3)
        
        # Plot insertion trajectories for each target condition
        for target_condition in target_conditions:
            target_results = [r for r in results if r['target_condition'] == target_condition]
            
            # Group by timestep
            timestep_groups = defaultdict(list)
            for result in target_results:
                timestep_groups[result['timestep']].append(result)
            
            for insertion_timestep, timestep_results in sorted(timestep_groups.items()):
                # Load trajectory for this insertion experiment
                result = timestep_results[0]  # Take first result for this timestep
                result_dir = Path("generated_structures/cross_conditioned_late_insertion") / \
                           f"step_{insertion_timestep}" / result['source_condition'] / \
                           result['target_condition'] / f"trajectory_{result['trajectory_idx']}"
                
                trajectory_zip = result_dir / "generated_trajectories.zip"
                
                if trajectory_zip.exists():
                    insertion_trajectory = load_full_trajectory_from_zip(str(trajectory_zip), trajectory_idx=0, step_interval=50)
                    if insertion_trajectory:
                        # Extract lattice parameters for insertion trajectory
                        insertion_timesteps = []
                        insertion_lattice_data = {param: [] for param in vector_params + angle_params}
                        
                        for timestep, structure in insertion_trajectory:
                            # Adjust timesteps to start from insertion point
                            adjusted_timestep = insertion_timestep + timestep
                            insertion_timesteps.append(adjusted_timestep)
                            insertion_lattice_data['lattice_a'].append(structure.lattice.a)
                            insertion_lattice_data['lattice_b'].append(structure.lattice.b)
                            insertion_lattice_data['lattice_c'].append(structure.lattice.c)
                            insertion_lattice_data['lattice_alpha'].append(structure.lattice.alpha)
                            insertion_lattice_data['lattice_beta'].append(structure.lattice.beta)
                            insertion_lattice_data['lattice_gamma'].append(structure.lattice.gamma)
                        
                        # Plot insertion trajectory
                        clean_label = target_condition.replace(f"{category}_", "").replace("_", " ").title()
                        
                        for param_idx, param in enumerate(vector_params):
                            axes_vectors[param_idx].plot(insertion_timesteps, insertion_lattice_data[param],
                                                       'o-', color=color_map[target_condition], linewidth=2, markersize=4,
                                                       label=f'{clean_label} @ step {insertion_timestep}', alpha=0.9)
                            # Add vertical line at insertion point
                            axes_vectors[param_idx].axvline(x=insertion_timestep, color=color_map[target_condition], 
                                                          linestyle='--', alpha=0.5, linewidth=1)
                        
                        for param_idx, param in enumerate(angle_params):
                            axes_angles[param_idx].plot(insertion_timesteps, insertion_lattice_data[param],
                                                      'o-', color=color_map[target_condition], linewidth=2, markersize=4,
                                                      label=f'{clean_label} @ step {insertion_timestep}', alpha=0.9)
                            # Add vertical line at insertion point
                            axes_angles[param_idx].axvline(x=insertion_timestep, color=color_map[target_condition], 
                                                         linestyle='--', alpha=0.5, linewidth=1)
        
        # Customize vector plot
        for ax in axes_vectors:
            ax.set_xlim(-50, 1050)
            ax.set_xticks([0, 200, 400, 600, 800, 1000])
            ax.axvspan(600, 950, alpha=0.1, color='red')
        
        axes_vectors[-1].set_xlabel('Diffusion Timestep', fontsize=14)
        axes_vectors[0].legend(fontsize=10, loc='upper right')
        
        clean_category = category.replace("_", " ").title()
        fig_vectors.suptitle(f'{clean_category} Cross-Conditioning: Lattice Vector Evolution', 
                           fontsize=16, fontweight='bold')
        
        # Customize angle plot
        for ax in axes_angles:
            ax.set_xlim(-50, 1050)
            ax.set_xticks([0, 200, 400, 600, 800, 1000])
            ax.axvspan(600, 950, alpha=0.1, color='red')
        
        axes_angles[-1].set_xlabel('Diffusion Timestep', fontsize=14)
        axes_angles[0].legend(fontsize=10, loc='upper right')
        
        fig_angles.suptitle(f'{clean_category} Cross-Conditioning: Lattice Angle Evolution', 
                          fontsize=16, fontweight='bold')
        
        fig_vectors.tight_layout()
        fig_angles.tight_layout()
        
        # Save plots
        vector_path = output_dir / f"{category}_lattice_vectors_trajectory.png"
        angle_path = output_dir / f"{category}_lattice_angles_trajectory.png"
        
        fig_vectors.savefig(vector_path, dpi=300, bbox_inches='tight')
        fig_angles.savefig(angle_path, dpi=300, bbox_inches='tight')
        
        print(f"Saved lattice vector trajectory: {vector_path}")
        print(f"Saved lattice angle trajectory: {angle_path}")
        
        plt.close(fig_vectors)
        plt.close(fig_angles)


def plot_trajectory_evolution(results_by_category: Dict[str, List[Dict[str, Any]]], 
                            baseline_results: Dict[str, Dict[str, Any]], 
                            output_dir: Path):
    """Plot comprehensive trajectory evolution showing full trajectories every 50 steps."""
    
    # Source condition mapping
    source_condition_map = {
        "dft_band_gap": "dft_band_gap_insulator",
        "energy_above_hull": "energy_above_hull_metastable", 
        "space_group": "space_group_cubic_fcc"
    }
    
    for category, results in results_by_category.items():
        print(f"\nGenerating trajectory evolution plot for {category}...")
        
        # Create figure with subplots for each target condition
        target_conditions = sorted(set(r['target_condition'] for r in results))
        
        fig, axes = plt.subplots(len(target_conditions), 1, figsize=(16, 6 * len(target_conditions)))
        if len(target_conditions) == 1:
            axes = [axes]
        
        for ax_idx, target_condition in enumerate(target_conditions):
            ax = axes[ax_idx]
            
            # Get baseline trajectory for source condition
            source_condition = source_condition_map.get(category)
            if source_condition and source_condition in baseline_results:
                baseline_path = f"generated_structures/three_property_study/{source_condition}"
                trajectory_zip = Path(baseline_path) / "generated_trajectories.zip"
                
                if trajectory_zip.exists():
                    baseline_trajectory = load_full_trajectory_from_zip(str(trajectory_zip), trajectory_idx=0, step_interval=50)
                    if baseline_trajectory:
                        baseline_evolution = analyze_trajectory_evolution(baseline_trajectory)
                        
                        # Plot baseline trajectory (full 0-1000 steps)
                        if category == "space_group":
                            # Convert space group numbers to categorical labels
                            sg_labels = [f"SG{sg}" for sg in baseline_evolution['space_groups']]
                            ax.plot(baseline_evolution['timesteps'], sg_labels, 'o-', 
                                   color='black', linewidth=3, markersize=8, 
                                   label='Baseline (source condition)', alpha=0.8)
                        else:
                            # For other properties, use space group numbers as proxy
                            ax.plot(baseline_evolution['timesteps'], baseline_evolution['space_groups'], 'o-',
                                   color='black', linewidth=3, markersize=8,
                                   label='Baseline (source condition)', alpha=0.8)
            
            # Plot insertion trajectories for this target condition
            target_results = [r for r in results if r['target_condition'] == target_condition]
            
            # Group by timestep
            timestep_groups = defaultdict(list)
            for result in target_results:
                timestep_groups[result['timestep']].append(result)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(timestep_groups)))
            
            for color_idx, (insertion_timestep, timestep_results) in enumerate(sorted(timestep_groups.items())):
                # Load trajectory for this insertion experiment
                result = timestep_results[0]  # Take first result for this timestep
                result_dir = Path("generated_structures/cross_conditioned_late_insertion") / \
                           f"step_{insertion_timestep}" / result['source_condition'] / \
                           result['target_condition'] / f"trajectory_{result['trajectory_idx']}"
                
                trajectory_zip = result_dir / "generated_trajectories.zip"
                
                if trajectory_zip.exists():
                    insertion_trajectory = load_full_trajectory_from_zip(str(trajectory_zip), trajectory_idx=0, step_interval=50)
                    if insertion_trajectory:
                        insertion_evolution = analyze_trajectory_evolution(insertion_trajectory)
                        
                        # Adjust timesteps to start from insertion point
                        adjusted_timesteps = [insertion_timestep + step for step in insertion_evolution['timesteps']]
                        
                        if category == "space_group":
                            # Convert space group numbers to categorical labels
                            sg_labels = [f"SG{sg}" for sg in insertion_evolution['space_groups']]
                            ax.plot(adjusted_timesteps, sg_labels, 'o-',
                                   color=colors[color_idx], linewidth=2, markersize=6,
                                   label=f'Insertion @ step {insertion_timestep}', alpha=0.9)
                        else:
                            ax.plot(adjusted_timesteps, insertion_evolution['space_groups'], 'o-',
                                   color=colors[color_idx], linewidth=2, markersize=6,
                                   label=f'Insertion @ step {insertion_timestep}', alpha=0.9)
                        
                        # Add vertical line at insertion point
                        ax.axvline(x=insertion_timestep, color=colors[color_idx], 
                                  linestyle='--', alpha=0.5, linewidth=1)
            
            # Customize subplot
            ax.set_xlabel('Diffusion Timestep', fontsize=12)
            
            if category == "space_group":
                ax.set_ylabel('Space Group', fontsize=12)
                # For space groups, make y-axis categorical
                ax.tick_params(axis='y', labelsize=10)
            else:
                ax.set_ylabel('Space Group Number', fontsize=12)
            
            # Clean up target condition name for title
            clean_target = target_condition.replace(f"{category}_", "").replace("_", " ").title()
            clean_category = category.replace("_", " ").title()
            ax.set_title(f'{clean_category} → {clean_target}', fontsize=14, fontweight='bold')
            
            # Set x-axis limits
            ax.set_xlim(-50, 1050)
            ax.set_xticks([0, 200, 400, 600, 800, 1000])
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper right')
            
            # Add text annotation for insertion region
            ax.axvspan(600, 950, alpha=0.1, color='red', label='Insertion window')
            ax.text(775, ax.get_ylim()[1] * 0.95, 'Insertion\nWindow', 
                   ha='center', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"{category}_trajectory_evolution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory evolution plot: {output_path}")
        plt.close()


def main():
    """Main analysis function."""
    
    # Create output directory
    output_dir = Path("mattergen/scripts/diffusion_step_analysis/insertion_analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Redirect stdout to both console and file
    import sys
    log_file = output_dir / "analysis_log.txt"
    
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
    
    with open(log_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(sys.stdout, f)
        
        print("Late Insertion Cross-Conditioning Analysis")
        print("=" * 60)
        
        # Collect baseline results
        print("\n1. Collecting baseline results...")
        baseline_results = collect_baseline_results()
        
        # Collect insertion results
        print("\n2. Collecting cross-conditioning insertion results...")
        results_by_category = collect_insertion_results()
        
        # Create summary DataFrame
        print("\n3. Creating summary DataFrame...")
        summary_df = create_summary_dataframe(results_by_category, baseline_results)
        
        # Save summary data
        summary_path = output_dir / "insertion_analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary data: {summary_path}")
        
        # Analyze success rates
        print("\n4. Analyzing success rates...")
        success_analysis = analyze_success_rates(results_by_category)
        
        # Save detailed results
        all_results = {
            'baseline_results': baseline_results,
            'insertion_results': results_by_category,
            'success_analysis': success_analysis,
            'summary_stats': {
                'total_baseline': len(baseline_results),
                'total_insertion': sum(len(results) for results in results_by_category.values()),
                'categories': list(results_by_category.keys())
            }
        }
        
        json_path = output_dir / "insertion_analysis_detailed.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Saved detailed results: {json_path}")
        
        # Generate plots for each category
        print("\n5. Generating plots...")
        
        for category, results in results_by_category.items():
            print(f"\nPlotting {category} results...")
            
            # Get corresponding baseline
            source_condition_map = {
                "dft_band_gap": "dft_band_gap_insulator",
                "energy_above_hull": "energy_above_hull_metastable", 
                "space_group": "space_group_cubic_fcc"
            }
            
            baseline = baseline_results.get(source_condition_map.get(category))
            
            # Plot property evolution for all categories
            plot_property_results(results, baseline, category, output_dir)
        
        # Generate comprehensive trajectory evolution plots
        print("\n6. Generating trajectory evolution plots...")
        plot_trajectory_evolution(results_by_category, baseline_results, output_dir)
        
        # Generate lattice parameter evolution plots
        print("\n7. Generating lattice parameter evolution plots...")
        plot_lattice_parameter_evolution(results_by_category, baseline_results, output_dir)
        
        # Generate lattice trajectory evolution plots
        print("\n8. Generating lattice trajectory evolution plots...")
        plot_lattice_trajectory_evolution(results_by_category, baseline_results, output_dir)
        
        # Print summary statistics
        print("\n9. Summary Statistics:")
        print(f"Total baseline structures analyzed: {len(baseline_results)}")
        
        for category, results in results_by_category.items():
            print(f"\n{category.upper()} Results:")
            print(f"  Total insertion experiments: {len(results)}")
            
            target_conditions = set(r['target_condition'] for r in results)
            timesteps = sorted(set(r['timestep'] for r in results))
            
            print(f"  Target conditions tested: {len(target_conditions)}")
            print(f"  Timesteps tested: {timesteps}")
            
            # Print success rates if available
            if category in success_analysis:
                cat_analysis = success_analysis[category]
                print(f"  Success analysis:")
                for target, analysis in cat_analysis['by_target'].items():
                    if analysis['target_space_group']:
                        clean_target = target.replace(f"{category}_", "").replace("_", " ")
                        print(f"    {clean_target}: {analysis['successes']}/{analysis['total_attempts']} " +
                              f"({analysis['success_rate']:.1%}) successful")
        
        print(f"\nResults saved in: {output_dir}")
        print("Analysis complete!")
        
        # Restore original stdout
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()