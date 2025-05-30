#!/usr/bin/env python3
"""
Analyze late insertion trajectory results.

This script compares the final structures from each conditioning strategy
with the original unconditioned trajectory to assess:
1) Structural similarity (RMSD, lattice differences)  
2) Property differences (space group, energy if available)
3) Overall impact of conditioning on final structures
"""

import os
import json
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
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
# from pymatgen.analysis.lattice_match import lattice_points_in_supercell


def load_structure_from_unconditioned_trajectory(
    zip_path: str, 
    trajectory_idx: int = 0, 
    timestep: int = -1
) -> Structure:
    """Load the final structure from the unconditioned trajectory."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
        
        if trajectory_idx >= len(trajectory_files):
            raise ValueError(f"Trajectory index {trajectory_idx} not available. Found {len(trajectory_files)} trajectories.")
        
        chosen_file = trajectory_files[trajectory_idx]
        print(f"Loading final structure from: {chosen_file}")
        
        with zf.open(chosen_file) as f:
            content = io.StringIO(f.read().decode())
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            
            # Get final structure (timestep=-1 means last structure)
            atoms = atoms_list[timestep]
            structure = AseAtomsAdaptor.get_structure(atoms)
            
            return structure


def load_structures_from_strategy(strategy_dir: Path) -> List[Structure]:
    """Load all final structures from a conditioning strategy."""
    extxyz_path = strategy_dir / "generated_crystals.extxyz"
    
    if not extxyz_path.exists():
        print(f"Warning: No structures found in {strategy_dir}")
        return []
    
    try:
        atoms_list = ase.io.read(extxyz_path, index=":", format="extxyz")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        
        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
        return structures
        
    except Exception as e:
        print(f"Error loading structures from {strategy_dir}: {e}")
        return []


def analyze_space_group(structure: Structure) -> Dict[str, Any]:
    """Analyze space group properties of a structure."""
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5.0)
        return {
            'space_group_number': analyzer.get_space_group_number(),
            'space_group_symbol': analyzer.get_space_group_symbol(),
            'crystal_system': analyzer.get_crystal_system(),
            'lattice_type': analyzer.get_lattice_type(),
            'point_group': analyzer.get_point_group_symbol()
        }
    except Exception as e:
        print(f"Warning: Space group analysis failed: {e}")
        return {
            'space_group_number': 1,
            'space_group_symbol': 'P1',
            'crystal_system': 'triclinic',
            'lattice_type': 'triclinic',
            'point_group': '1'
        }


def calculate_structural_similarity(
    reference_structure: Structure,
    comparison_structure: Structure
) -> Dict[str, float]:
    """Calculate structural similarity metrics between two structures."""
    
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
            similarity_metrics[f'lattice_param_{["a", "b", "c", "alpha", "beta", "gamma"][i]}_rel_diff'] = rel_diff
    
    similarity_metrics['avg_lattice_rel_diff'] = np.mean(lattice_diffs) if lattice_diffs else 0.0
    similarity_metrics['max_lattice_rel_diff'] = np.max(lattice_diffs) if lattice_diffs else 0.0
    
    # Volume difference
    ref_volume = reference_structure.lattice.volume
    comp_volume = comparison_structure.lattice.volume
    similarity_metrics['volume_rel_diff'] = abs(comp_volume - ref_volume) / ref_volume if ref_volume > 0 else 0.0
    
    # Density difference
    ref_density = reference_structure.density
    comp_density = comparison_structure.density
    similarity_metrics['density_rel_diff'] = abs(comp_density - ref_density) / ref_density if ref_density > 0 else 0.0
    
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


def analyze_strategy_results(
    base_results_dir: Path,
    unconditioned_trajectory_path: str,
    trajectory_idx: int = 0
) -> Dict[str, Any]:
    """Analyze all strategy results compared to the unconditioned baseline."""
    
    print("Loading unconditioned reference structure...")
    reference_structure = load_structure_from_unconditioned_trajectory(
        unconditioned_trajectory_path, trajectory_idx, timestep=-1
    )
    
    print(f"Reference structure: {reference_structure.composition}")
    reference_sg = analyze_space_group(reference_structure)
    print(f"Reference space group: {reference_sg['space_group_number']} ({reference_sg['space_group_symbol']})")
    
    # Load experiment configuration
    config_path = base_results_dir / "experiment_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            experiment_config = json.load(f)
    else:
        experiment_config = {}
    
    # Analyze each strategy
    strategy_analyses = {}
    
    # Find all strategy directories
    strategy_dirs = [d for d in base_results_dir.iterdir() 
                    if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\nAnalyzing {len(strategy_dirs)} strategies...")
    
    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        print(f"\nAnalyzing strategy: {strategy_name}")
        
        # Load strategy configuration
        strategy_config_path = strategy_dir / "strategy_config.json"
        if strategy_config_path.exists():
            with open(strategy_config_path, 'r') as f:
                strategy_config = json.load(f)
        else:
            strategy_config = {}
        
        # Load structures from this strategy
        structures = load_structures_from_strategy(strategy_dir)
        
        if not structures:
            strategy_analyses[strategy_name] = {
                'success': False,
                'error': 'No structures found',
                'strategy_config': strategy_config
            }
            continue
        
        # Analyze the first (and typically only) structure
        structure = structures[0]
        
        # Space group analysis
        structure_sg = analyze_space_group(structure)
        
        # Structural similarity analysis
        similarity = calculate_structural_similarity(reference_structure, structure)
        
        # Check if conditioning was successful
        conditioning_success = {}
        if 'strategy' in strategy_config and 'conditions' in strategy_config['strategy']:
            conditions = strategy_config['strategy']['conditions']
            
            if 'space_group' in conditions:
                target_sg = conditions['space_group']
                actual_sg = structure_sg['space_group_number']
                conditioning_success['space_group'] = {
                    'target': target_sg,
                    'actual': actual_sg,
                    'achieved': target_sg == actual_sg
                }
            
            # Note: Energy properties would require DFT calculation
            # For now we just record the target values
            if 'energy_above_hull' in conditions:
                conditioning_success['energy_above_hull'] = {
                    'target': conditions['energy_above_hull'],
                    'actual': None,  # Would need DFT calculation
                    'achieved': None
                }
            
            if 'formation_energy_per_atom' in conditions:
                conditioning_success['formation_energy_per_atom'] = {
                    'target': conditions['formation_energy_per_atom'],
                    'actual': None,  # Would need DFT calculation
                    'achieved': None
                }
        
        strategy_analyses[strategy_name] = {
            'success': True,
            'strategy_config': strategy_config,
            'structure_info': {
                'composition': str(structure.composition),
                'formula': structure.formula,
                'lattice_parameters': list(structure.lattice.parameters),
                'volume': structure.lattice.volume,
                'density': structure.density,
                'num_atoms': len(structure)
            },
            'space_group': structure_sg,
            'similarity_to_reference': similarity,
            'conditioning_success': conditioning_success,
            'property_changes': {
                'space_group_changed': (
                    structure_sg['space_group_number'] != reference_sg['space_group_number']
                ),
                'crystal_system_changed': (
                    structure_sg['crystal_system'] != reference_sg['crystal_system']
                )
            }
        }
    
    return {
        'reference_structure': {
            'composition': str(reference_structure.composition),
            'space_group': reference_sg,
            'lattice_parameters': list(reference_structure.lattice.parameters),
            'volume': reference_structure.lattice.volume,
            'density': reference_structure.density
        },
        'experiment_config': experiment_config,
        'strategy_analyses': strategy_analyses,
        'summary_statistics': calculate_summary_statistics(strategy_analyses, reference_sg)
    }


def calculate_summary_statistics(
    strategy_analyses: Dict[str, Any],
    reference_sg: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate summary statistics across all strategies."""
    
    successful_strategies = {name: analysis for name, analysis in strategy_analyses.items() 
                           if analysis.get('success', False)}
    
    # Always return basic counts even if no successful strategies
    summary = {
        'total_strategies': len(strategy_analyses),
        'successful_strategies': len(successful_strategies),
        'failed_strategies': len(strategy_analyses) - len(successful_strategies)
    }
    
    if not successful_strategies:
        return summary
    
    # Structural similarity statistics
    similarity_metrics = []
    for analysis in successful_strategies.values():
        if 'similarity_to_reference' in analysis:
            similarity_metrics.append(analysis['similarity_to_reference'])
    
    summary = {
        'total_strategies': len(strategy_analyses),
        'successful_strategies': len(successful_strategies),
        'failed_strategies': len(strategy_analyses) - len(successful_strategies)
    }
    
    if similarity_metrics:
        # Average similarity metrics
        summary['avg_similarity_metrics'] = {}
        for key in similarity_metrics[0].keys():
            if isinstance(similarity_metrics[0][key], (int, float)):
                values = [m[key] for m in similarity_metrics if m[key] is not None]
                if values:
                    summary['avg_similarity_metrics'][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
    
    # Space group change statistics
    space_group_changes = []
    crystal_system_changes = []
    
    for analysis in successful_strategies.values():
        if 'property_changes' in analysis:
            space_group_changes.append(analysis['property_changes']['space_group_changed'])
            crystal_system_changes.append(analysis['property_changes']['crystal_system_changed'])
    
    summary['property_change_rates'] = {
        'space_group_changed': np.mean(space_group_changes) if space_group_changes else 0,
        'crystal_system_changed': np.mean(crystal_system_changes) if crystal_system_changes else 0
    }
    
    # Conditioning success rates
    conditioning_types = ['space_group', 'energy_above_hull', 'formation_energy_per_atom']
    conditioning_success_rates = {}
    
    for cond_type in conditioning_types:
        successes = []
        for analysis in successful_strategies.values():
            if ('conditioning_success' in analysis and 
                cond_type in analysis['conditioning_success'] and
                analysis['conditioning_success'][cond_type]['achieved'] is not None):
                successes.append(analysis['conditioning_success'][cond_type]['achieved'])
        
        if successes:
            conditioning_success_rates[cond_type] = np.mean(successes)
    
    summary['conditioning_success_rates'] = conditioning_success_rates
    
    return summary


def create_analysis_plots(analysis_results: Dict[str, Any], output_dir: Path) -> None:
    """Create visualization plots for the analysis results."""
    
    strategy_analyses = analysis_results['strategy_analyses']
    successful_strategies = {name: analysis for name, analysis in strategy_analyses.items() 
                           if analysis.get('success', False)}
    
    if not successful_strategies:
        print("No successful strategies to plot")
        return
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create plots directory
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Lattice parameter changes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    
    for i, param in enumerate(param_names):
        key = f'lattice_param_{param}_rel_diff'
        values = []
        labels = []
        
        for name, analysis in successful_strategies.items():
            if ('similarity_to_reference' in analysis and 
                key in analysis['similarity_to_reference']):
                values.append(analysis['similarity_to_reference'][key] * 100)  # Convert to percentage
                labels.append(name.replace('_', '\n'))
        
        if values:
            axes[i].bar(range(len(values)), values)
            axes[i].set_title(f'Lattice Parameter {param.upper()}\nRelative Difference (%)')
            axes[i].set_xticks(range(len(labels)))
            axes[i].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            axes[i].set_ylabel('Relative Difference (%)')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "lattice_parameter_changes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Volume and density changes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Volume changes
    volume_changes = []
    labels = []
    for name, analysis in successful_strategies.items():
        if ('similarity_to_reference' in analysis and 
            'volume_rel_diff' in analysis['similarity_to_reference']):
            volume_changes.append(analysis['similarity_to_reference']['volume_rel_diff'] * 100)
            labels.append(name.replace('_', '\n'))
    
    if volume_changes:
        ax1.bar(range(len(volume_changes)), volume_changes)
        ax1.set_title('Volume Changes (%)')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Relative Volume Change (%)')
    
    # Density changes
    density_changes = []
    for name, analysis in successful_strategies.items():
        if ('similarity_to_reference' in analysis and 
            'density_rel_diff' in analysis['similarity_to_reference']):
            density_changes.append(analysis['similarity_to_reference']['density_rel_diff'] * 100)
    
    if density_changes:
        ax2.bar(range(len(density_changes)), density_changes)
        ax2.set_title('Density Changes (%)')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Relative Density Change (%)')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "volume_density_changes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Space group analysis
    reference_sg = analysis_results['reference_structure']['space_group']['space_group_number']
    
    space_groups = []
    strategy_names = []
    colors = []
    
    for name, analysis in successful_strategies.items():
        if 'space_group' in analysis:
            sg_num = analysis['space_group']['space_group_number']
            space_groups.append(sg_num)
            strategy_names.append(name.replace('_', '\n'))
            
            # Color code: green if matches reference, blue if matches target, red otherwise
            if sg_num == reference_sg:
                colors.append('green')
            elif ('conditioning_success' in analysis and 
                  'space_group' in analysis['conditioning_success'] and
                  analysis['conditioning_success']['space_group']['achieved']):
                colors.append('blue')
            else:
                colors.append('red')
    
    if space_groups:
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(space_groups)), space_groups, color=colors, alpha=0.7)
        ax.axhline(y=reference_sg, color='green', linestyle='--', linewidth=2, 
                  label=f'Reference SG {reference_sg}')
        ax.set_title('Space Group Numbers by Strategy')
        ax.set_xticks(range(len(strategy_names)))
        ax.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Space Group Number')
        ax.legend()
        
        # Add value labels on bars
        for bar, sg in zip(bars, space_groups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{sg}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "space_group_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Analysis plots saved in: {plots_dir}")


def save_analysis_results(
    analysis_results: Dict[str, Any], 
    output_dir: Path
) -> None:
    """Save analysis results to files."""
    
    # Save full analysis as JSON
    with open(output_dir / "full_analysis_results.json", 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Create summary CSV
    strategy_analyses = analysis_results['strategy_analyses']
    successful_strategies = {name: analysis for name, analysis in strategy_analyses.items() 
                           if analysis.get('success', False)}
    
    if successful_strategies:
        summary_data = []
        
        for strategy_name, analysis in successful_strategies.items():
            row = {
                'strategy_name': strategy_name,
                'composition': analysis['structure_info']['composition'],
                'space_group_number': analysis['space_group']['space_group_number'],
                'space_group_symbol': analysis['space_group']['space_group_symbol'],
                'crystal_system': analysis['space_group']['crystal_system'],
                'volume': analysis['structure_info']['volume'],
                'density': analysis['structure_info']['density']
            }
            
            # Add similarity metrics
            if 'similarity_to_reference' in analysis:
                sim = analysis['similarity_to_reference']
                row.update({
                    'avg_lattice_rel_diff': sim.get('avg_lattice_rel_diff', None),
                    'volume_rel_diff': sim.get('volume_rel_diff', None),
                    'density_rel_diff': sim.get('density_rel_diff', None),
                    'structures_match': sim.get('structures_match', None),
                    'rms_distance': sim.get('rms_distance', None)
                })
            
            # Add conditioning success
            if 'conditioning_success' in analysis:
                for cond_type, cond_data in analysis['conditioning_success'].items():
                    row[f'{cond_type}_target'] = cond_data.get('target', None)
                    row[f'{cond_type}_achieved'] = cond_data.get('achieved', None)
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_dir / "analysis_summary.csv", index=False)
        print(f"Summary CSV saved: {output_dir / 'analysis_summary.csv'}")


def main():
    """Main function to run the analysis."""
    
    # Configuration
    base_results_dir = Path("generated_structures/late_insertion_trajectories")
    unconditioned_trajectory_path = "generated_structures/three_property_study/unconditional/generated_trajectories.zip"
    trajectory_idx = 0
    
    if not base_results_dir.exists():
        print(f"Error: Results directory not found: {base_results_dir}")
        return
    
    if not Path(unconditioned_trajectory_path).exists():
        print(f"Error: Unconditioned trajectory not found: {unconditioned_trajectory_path}")
        return
    
    print("=" * 60)
    print("Late Insertion Trajectory Analysis")
    print("=" * 60)
    
    # Run analysis
    analysis_results = analyze_strategy_results(
        base_results_dir=base_results_dir,
        unconditioned_trajectory_path=unconditioned_trajectory_path,
        trajectory_idx=trajectory_idx
    )
    
    # Save results
    save_analysis_results(analysis_results, base_results_dir)
    
    # Create plots
    create_analysis_plots(analysis_results, base_results_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    ref_sg = analysis_results['reference_structure']['space_group']
    print(f"Reference structure:")
    print(f"  Composition: {analysis_results['reference_structure']['composition']}")
    print(f"  Space group: {ref_sg['space_group_number']} ({ref_sg['space_group_symbol']})")
    print(f"  Crystal system: {ref_sg['crystal_system']}")
    
    summary = analysis_results['summary_statistics']
    print(f"\nStrategy results:")
    print(f"  Total strategies: {summary['total_strategies']}")
    print(f"  Successful: {summary['successful_strategies']}")
    print(f"  Failed: {summary['failed_strategies']}")
    
    if 'property_change_rates' in summary:
        print(f"\nProperty change rates:")
        print(f"  Space group changed: {summary['property_change_rates']['space_group_changed']:.1%}")
        print(f"  Crystal system changed: {summary['property_change_rates']['crystal_system_changed']:.1%}")
    
    if 'conditioning_success_rates' in summary:
        print(f"\nConditioning success rates:")
        for cond_type, rate in summary['conditioning_success_rates'].items():
            print(f"  {cond_type}: {rate:.1%}")
    
    if 'avg_similarity_metrics' in summary:
        print(f"\nAverage structural differences:")
        avg_metrics = summary['avg_similarity_metrics']
        if 'avg_lattice_rel_diff' in avg_metrics:
            print(f"  Average lattice parameter difference: {avg_metrics['avg_lattice_rel_diff']['mean']:.1%}")
        if 'volume_rel_diff' in avg_metrics:
            print(f"  Average volume difference: {avg_metrics['volume_rel_diff']['mean']:.1%}")
        if 'density_rel_diff' in avg_metrics:
            print(f"  Average density difference: {avg_metrics['density_rel_diff']['mean']:.1%}")
    
    # Detailed strategy results
    strategy_analyses = analysis_results['strategy_analyses']
    successful_strategies = {name: analysis for name, analysis in strategy_analyses.items() 
                           if analysis.get('success', False)}
    
    if successful_strategies:
        print(f"\nDetailed strategy results:")
        for strategy_name, analysis in successful_strategies.items():
            print(f"\n  {strategy_name}:")
            
            # Space group info
            sg = analysis['space_group']
            print(f"    Space group: {sg['space_group_number']} ({sg['space_group_symbol']})")
            print(f"    Crystal system: {sg['crystal_system']}")
            
            # Structural similarity
            if 'similarity_to_reference' in analysis:
                sim = analysis['similarity_to_reference']
                print(f"    Avg lattice difference: {sim.get('avg_lattice_rel_diff', 0):.1%}")
                print(f"    Volume difference: {sim.get('volume_rel_diff', 0):.1%}")
                print(f"    Structure match: {sim.get('structures_match', False)}")
            
            # Conditioning success
            if 'conditioning_success' in analysis:
                for cond_type, cond_data in analysis['conditioning_success'].items():
                    if cond_data['achieved'] is not None:
                        status = "✓" if cond_data['achieved'] else "✗"
                        print(f"    {cond_type} conditioning: {status} (target: {cond_data['target']})")
    
    failed_strategies = {name: analysis for name, analysis in strategy_analyses.items() 
                        if not analysis.get('success', False)}
    
    if failed_strategies:
        print(f"\nFailed strategies:")
        for strategy_name, analysis in failed_strategies.items():
            print(f"  {strategy_name}: {analysis.get('error', 'Unknown error')}")
    
    print(f"\nDetailed results saved in: {base_results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()