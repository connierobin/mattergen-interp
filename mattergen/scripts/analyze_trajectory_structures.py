#!/usr/bin/env python3
"""
Analyze structures from trajectories to understand late-stage insertion potential.

This script compares structures at different timesteps from the original unconditioned
trajectory to assess how much structures change in the final 200 steps and what
the baseline structural similarity looks like.
"""

import zipfile
import io
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ase
import ase.io
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher


def load_trajectory_structures(zip_path: str, trajectory_idx: int = 0) -> List[Structure]:
    """Load all structures from a trajectory."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
        
        if trajectory_idx >= len(trajectory_files):
            raise ValueError(f"Trajectory index {trajectory_idx} not available. Found {len(trajectory_files)} trajectories.")
        
        chosen_file = trajectory_files[trajectory_idx]
        print(f"Loading trajectory: {chosen_file}")
        
        with zf.open(chosen_file) as f:
            content = io.StringIO(f.read().decode())
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            
            structures = []
            for i, atoms in enumerate(atoms_list):
                try:
                    structure = AseAtomsAdaptor.get_structure(atoms)
                    structures.append(structure)
                except Exception as e:
                    print(f"Warning: Could not convert structure at timestep {i}: {e}")
                    structures.append(None)
            
            return structures


def analyze_space_group(structure: Structure) -> Dict[str, Any]:
    """Analyze space group properties of a structure."""
    if structure is None:
        return {'space_group_number': None, 'space_group_symbol': None, 
                'crystal_system': None, 'lattice_type': None}
    
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
        return {'space_group_number': 1, 'space_group_symbol': 'P1',
                'crystal_system': 'triclinic', 'lattice_type': 'triclinic'}


def calculate_structural_similarity(
    reference_structure: Structure,
    comparison_structure: Structure
) -> Dict[str, float]:
    """Calculate structural similarity metrics between two structures."""
    
    if reference_structure is None or comparison_structure is None:
        return {'avg_lattice_rel_diff': None, 'volume_rel_diff': None, 
                'density_rel_diff': None, 'structures_match': False}
    
    similarity_metrics = {}
    
    # Lattice parameter differences
    ref_params = reference_structure.lattice.parameters
    comp_params = comparison_structure.lattice.parameters
    
    # Calculate relative differences in lattice parameters
    lattice_diffs = []
    for i, (ref, comp) in enumerate(zip(ref_params, comp_params)):
        if ref > 0:  # Avoid division by zero
            rel_diff = abs(comp - ref) / ref
            lattice_diffs.append(rel_diff)
    
    similarity_metrics['avg_lattice_rel_diff'] = np.mean(lattice_diffs) if lattice_diffs else None
    similarity_metrics['max_lattice_rel_diff'] = np.max(lattice_diffs) if lattice_diffs else None
    
    # Volume difference
    ref_volume = reference_structure.lattice.volume
    comp_volume = comparison_structure.lattice.volume
    similarity_metrics['volume_rel_diff'] = abs(comp_volume - ref_volume) / ref_volume if ref_volume > 0 else None
    
    # Density difference
    ref_density = reference_structure.density
    comp_density = comparison_structure.density
    similarity_metrics['density_rel_diff'] = abs(comp_density - ref_density) / ref_density if ref_density > 0 else None
    
    # Try structure matching (may fail for very different structures)
    try:
        matcher = StructureMatcher(
            ltol=0.2,      # Lattice tolerance
            stol=0.3,      # Site tolerance  
            angle_tol=5,   # Angle tolerance
            primitive_cell=True,
            scale=True,
            attempt_supercell=True
        )
        
        is_match = matcher.fit(reference_structure, comparison_structure)
        similarity_metrics['structures_match'] = is_match
        
        if is_match:
            # Get RMS distance if structures match
            rms_dist = matcher.get_rms_dist(reference_structure, comparison_structure)
            similarity_metrics['rms_distance'] = rms_dist[0] if rms_dist else None
        else:
            similarity_metrics['rms_distance'] = None
            
    except Exception as e:
        print(f"Warning: Structure matching failed: {e}")
        similarity_metrics['structures_match'] = False
        similarity_metrics['rms_distance'] = None
    
    return similarity_metrics


def analyze_trajectory_evolution(structures: List[Structure], output_dir: Path) -> Dict[str, Any]:
    """Analyze how structures evolve throughout the trajectory."""
    
    print(f"Analyzing trajectory with {len(structures)} structures...")
    
    # Key timesteps to analyze
    timesteps_of_interest = [0, 200, 400, 600, 800, 900, 950, 990, 995, 999]
    timesteps_of_interest = [t for t in timesteps_of_interest if t < len(structures)]
    
    results = {
        'total_timesteps': len(structures),
        'timesteps_analyzed': timesteps_of_interest,
        'structure_analysis': {},
        'similarity_analysis': {}
    }
    
    # Analyze structures at key timesteps
    for timestep in timesteps_of_interest:
        structure = structures[timestep]
        
        if structure is not None:
            # Basic structure info
            structure_info = {
                'composition': str(structure.composition),
                'lattice_parameters': list(structure.lattice.parameters),
                'volume': structure.lattice.volume,
                'density': structure.density,
                'num_atoms': len(structure)
            }
            
            # Space group analysis
            sg_info = analyze_space_group(structure)
            structure_info.update(sg_info)
            
            results['structure_analysis'][timestep] = structure_info
        else:
            results['structure_analysis'][timestep] = {'error': 'Could not load structure'}
    
    # Compare final structure (timestep 999) with earlier timesteps
    if len(structures) > 999 and structures[999] is not None:
        final_structure = structures[999]
        
        # Compare with timestep 800 (late insertion starting point)
        if len(structures) > 800 and structures[800] is not None:
            similarity_800_vs_999 = calculate_structural_similarity(structures[800], final_structure)
            results['similarity_analysis']['timestep_800_vs_999'] = similarity_800_vs_999
            
            print(f"Structural change from timestep 800 to 999:")
            print(f"  Average lattice difference: {similarity_800_vs_999.get('avg_lattice_rel_diff', 'N/A'):.1%}")
            print(f"  Volume difference: {similarity_800_vs_999.get('volume_rel_diff', 'N/A'):.1%}")
            print(f"  Structures match: {similarity_800_vs_999.get('structures_match', False)}")
        
        # Compare with multiple earlier timesteps to see evolution pattern
        comparison_timesteps = [0, 400, 600, 700, 800, 850, 900, 950]
        comparison_timesteps = [t for t in comparison_timesteps if t < len(structures)]
        
        for timestep in comparison_timesteps:
            if structures[timestep] is not None:
                similarity = calculate_structural_similarity(structures[timestep], final_structure)
                results['similarity_analysis'][f'timestep_{timestep}_vs_999'] = similarity
    
    # Analyze space group evolution
    space_groups = []
    timesteps_with_sg = []
    
    for timestep in timesteps_of_interest:
        if timestep in results['structure_analysis'] and 'space_group_number' in results['structure_analysis'][timestep]:
            sg = results['structure_analysis'][timestep]['space_group_number']
            if sg is not None:
                space_groups.append(sg)
                timesteps_with_sg.append(timestep)
    
    if space_groups:
        results['space_group_evolution'] = {
            'timesteps': timesteps_with_sg,
            'space_groups': space_groups,
            'unique_space_groups': list(set(space_groups)),
            'final_space_group': space_groups[-1] if space_groups else None,
            'space_group_changes': len(set(space_groups)) > 1
        }
        
        print(f"Space group evolution:")
        print(f"  Unique space groups encountered: {results['space_group_evolution']['unique_space_groups']}")
        print(f"  Final space group: {results['space_group_evolution']['final_space_group']}")
        print(f"  Space group changed during trajectory: {results['space_group_evolution']['space_group_changes']}")
    
    return results


def create_trajectory_analysis_plots(analysis_results: Dict[str, Any], output_dir: Path) -> None:
    """Create visualization plots for trajectory analysis."""
    
    plots_dir = output_dir / "trajectory_analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Lattice parameter evolution
    timesteps = analysis_results['timesteps_analyzed']
    structure_analysis = analysis_results['structure_analysis']
    
    valid_timesteps = []
    lattice_params = {'a': [], 'b': [], 'c': [], 'alpha': [], 'beta': [], 'gamma': []}
    volumes = []
    densities = []
    
    for timestep in timesteps:
        if timestep in structure_analysis and 'lattice_parameters' in structure_analysis[timestep]:
            params = structure_analysis[timestep]['lattice_parameters']
            if len(params) == 6:
                valid_timesteps.append(timestep)
                lattice_params['a'].append(params[0])
                lattice_params['b'].append(params[1])
                lattice_params['c'].append(params[2])
                lattice_params['alpha'].append(params[3])
                lattice_params['beta'].append(params[4])
                lattice_params['gamma'].append(params[5])
                volumes.append(structure_analysis[timestep]['volume'])
                densities.append(structure_analysis[timestep]['density'])
    
    if valid_timesteps:
        # Lattice parameters plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        param_units = ['Å', 'Å', 'Å', '°', '°', '°']
        
        for i, (param, unit) in enumerate(zip(param_names, param_units)):
            axes[i].plot(valid_timesteps, lattice_params[param], 'o-', markersize=6)
            axes[i].set_title(f'Lattice Parameter {param.upper()} ({unit})')
            axes[i].set_xlabel('Timestep')
            axes[i].set_ylabel(f'{param} ({unit})')
            axes[i].grid(True, alpha=0.3)
            
            # Highlight the late insertion starting point (timestep 800)
            if 800 in valid_timesteps:
                idx_800 = valid_timesteps.index(800)
                axes[i].axvline(x=800, color='red', linestyle='--', alpha=0.7, label='Late insertion start')
                axes[i].scatter(800, lattice_params[param][idx_800], color='red', s=100, zorder=5)
        
        # Add legend to first subplot
        axes[0].legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "lattice_parameter_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Volume and density evolution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(valid_timesteps, volumes, 'o-', color='blue', markersize=6)
        ax1.set_title('Unit Cell Volume Evolution')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Volume (Ų)')
        ax1.grid(True, alpha=0.3)
        if 800 in valid_timesteps:
            ax1.axvline(x=800, color='red', linestyle='--', alpha=0.7, label='Late insertion start')
            ax1.legend()
        
        ax2.plot(valid_timesteps, densities, 'o-', color='green', markersize=6)
        ax2.set_title('Density Evolution')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Density (g/cm³)')
        ax2.grid(True, alpha=0.3)
        if 800 in valid_timesteps:
            ax2.axvline(x=800, color='red', linestyle='--', alpha=0.7, label='Late insertion start')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "volume_density_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Space group evolution
    if 'space_group_evolution' in analysis_results:
        sg_data = analysis_results['space_group_evolution']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sg_data['timesteps'], sg_data['space_groups'], 'o-', markersize=8, linewidth=2)
        ax.set_title('Space Group Evolution Throughout Trajectory')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Space Group Number')
        ax.grid(True, alpha=0.3)
        
        # Highlight late insertion starting point
        if 800 in sg_data['timesteps']:
            ax.axvline(x=800, color='red', linestyle='--', alpha=0.7, label='Late insertion start')
            ax.legend()
        
        # Add space group values as text labels
        for x, y in zip(sg_data['timesteps'], sg_data['space_groups']):
            ax.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "space_group_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Structural similarity vs timestep
    if 'similarity_analysis' in analysis_results:
        sim_data = analysis_results['similarity_analysis']
        
        # Extract comparison data
        timesteps_compared = []
        lattice_diffs = []
        volume_diffs = []
        
        for key, similarity in sim_data.items():
            if key.startswith('timestep_') and key.endswith('_vs_999'):
                timestep = int(key.split('_')[1])
                timesteps_compared.append(timestep)
                lattice_diffs.append(similarity.get('avg_lattice_rel_diff', 0) * 100)  # Convert to percentage
                volume_diffs.append(similarity.get('volume_rel_diff', 0) * 100)
        
        if timesteps_compared:
            # Sort by timestep
            sorted_data = sorted(zip(timesteps_compared, lattice_diffs, volume_diffs))
            timesteps_compared, lattice_diffs, volume_diffs = zip(*sorted_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(timesteps_compared, lattice_diffs, 'o-', color='blue', markersize=6)
            ax1.set_title('Lattice Parameter Difference vs Final Structure')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Average Lattice Parameter Difference (%)')
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=800, color='red', linestyle='--', alpha=0.7, label='Late insertion start')
            ax1.legend()
            
            ax2.plot(timesteps_compared, volume_diffs, 'o-', color='green', markersize=6)
            ax2.set_title('Volume Difference vs Final Structure')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Volume Difference (%)')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=800, color='red', linestyle='--', alpha=0.7, label='Late insertion start')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / "structural_similarity_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Trajectory analysis plots saved in: {plots_dir}")


def main():
    """Main function to analyze trajectory structures."""
    
    # Configuration
    trajectory_zip_path = "generated_structures/three_property_study/unconditional/generated_trajectories.zip"
    output_dir = Path("generated_structures/trajectory_analysis")
    trajectory_idx = 0
    
    if not Path(trajectory_zip_path).exists():
        print(f"Error: Trajectory file not found: {trajectory_zip_path}")
        return
    
    print("=" * 60)
    print("Trajectory Structure Analysis")
    print("=" * 60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectory structures
    print(f"Loading structures from trajectory...")
    structures = load_trajectory_structures(trajectory_zip_path, trajectory_idx)
    
    print(f"Loaded {len(structures)} structures from trajectory")
    if structures:
        valid_structures = sum(1 for s in structures if s is not None)
        print(f"Successfully converted {valid_structures}/{len(structures)} structures to pymatgen format")
    
    # Analyze trajectory evolution
    analysis_results = analyze_trajectory_evolution(structures, output_dir)
    
    # Create plots
    create_trajectory_analysis_plots(analysis_results, output_dir)
    
    # Save analysis results
    import json
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    analysis_results_json = convert_for_json(analysis_results)
    
    with open(output_dir / "trajectory_analysis_results.json", 'w') as f:
        json.dump(analysis_results_json, f, indent=2, default=str)
    
    # Create summary CSV
    if 'structure_analysis' in analysis_results:
        summary_data = []
        
        for timestep, analysis in analysis_results['structure_analysis'].items():
            if 'error' not in analysis:
                row = {
                    'timestep': timestep,
                    'composition': analysis.get('composition', ''),
                    'space_group_number': analysis.get('space_group_number', None),
                    'space_group_symbol': analysis.get('space_group_symbol', ''),
                    'crystal_system': analysis.get('crystal_system', ''),
                    'volume': analysis.get('volume', None),
                    'density': analysis.get('density', None),
                    'num_atoms': analysis.get('num_atoms', None)
                }
                
                # Add lattice parameters
                if 'lattice_parameters' in analysis and len(analysis['lattice_parameters']) == 6:
                    params = analysis['lattice_parameters']
                    row.update({
                        'a': params[0], 'b': params[1], 'c': params[2],
                        'alpha': params[3], 'beta': params[4], 'gamma': params[5]
                    })
                
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(output_dir / "trajectory_structure_summary.csv", index=False)
            print(f"Structure summary CSV saved: {output_dir / 'trajectory_structure_summary.csv'}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"Total trajectory length: {analysis_results['total_timesteps']} timesteps")
    print(f"Analyzed timesteps: {analysis_results['timesteps_analyzed']}")
    
    if 'space_group_evolution' in analysis_results:
        sg_data = analysis_results['space_group_evolution']
        print(f"\nSpace group evolution:")
        print(f"  Unique space groups: {sg_data['unique_space_groups']}")
        print(f"  Final space group: {sg_data['final_space_group']}")
        print(f"  Space group changed: {sg_data['space_group_changes']}")
    
    if 'similarity_analysis' in analysis_results:
        sim_800_999 = analysis_results['similarity_analysis'].get('timestep_800_vs_999')
        if sim_800_999:
            print(f"\nStructural changes from timestep 800 to 999 (late insertion window):")
            avg_lattice_diff = sim_800_999.get('avg_lattice_rel_diff')
            if avg_lattice_diff is not None:
                print(f"  Average lattice parameter change: {avg_lattice_diff:.1%}")
            
            volume_diff = sim_800_999.get('volume_rel_diff')
            if volume_diff is not None:
                print(f"  Volume change: {volume_diff:.1%}")
            
            print(f"  Structures considered equivalent: {sim_800_999.get('structures_match', False)}")
            
            rms_dist = sim_800_999.get('rms_distance')
            if rms_dist is not None:
                print(f"  RMS distance: {rms_dist:.3f} Å")
    
    print(f"\nDetailed results saved in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()