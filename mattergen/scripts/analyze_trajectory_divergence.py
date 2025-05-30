#!/usr/bin/env python3
"""
Analyze trajectory divergence between original and late-insertion conditioned trajectories.

This script compares the evolution of structures from timestep 800-1000 in the original
unconditioned trajectory vs the conditioned late-insertion trajectories to visualize:
1. How quickly the trajectories diverge from the common starting point
2. What structural properties change most dramatically 
3. The effectiveness of late-stage conditioning on different properties
"""

import zipfile
import io
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
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


def load_trajectory_segment(
    zip_path: str, 
    trajectory_idx: int = 0,
    start_timestep: int = 800,
    end_timestep: int = 1000
) -> List[Structure]:
    """Load a segment of the original trajectory (timesteps 800-1000)."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
        
        if trajectory_idx >= len(trajectory_files):
            raise ValueError(f"Trajectory index {trajectory_idx} not available. Found {len(trajectory_files)} trajectories.")
        
        chosen_file = trajectory_files[trajectory_idx]
        print(f"Loading original trajectory segment: {chosen_file}")
        print(f"  Timesteps: {start_timestep} to {end_timestep}")
        
        with zf.open(chosen_file) as f:
            content = io.StringIO(f.read().decode())
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            
            # Remember: trajectory has predictor+corrector steps, so 2x structures per timestep
            # timestep 800 = index 1600, timestep 1000 = index 2000 (but index 1999 is last)
            start_idx = start_timestep * 2  # Convert timestep to trajectory index
            end_idx = min(end_timestep * 2, len(atoms_list))  # Don't exceed trajectory length
            
            print(f"  Loading indices {start_idx} to {end_idx-1} from {len(atoms_list)} total structures")
            
            structures = []
            for i in range(start_idx, end_idx, 2):  # Take every 2nd structure (skip corrector steps)
                try:
                    atoms = atoms_list[i]
                    structure = AseAtomsAdaptor.get_structure(atoms)
                    structures.append(structure)
                except Exception as e:
                    print(f"Warning: Could not convert structure at index {i}: {e}")
                    structures.append(None)
            
            print(f"  Successfully loaded {len([s for s in structures if s is not None])}/{len(structures)} structures")
            return structures


def load_late_insertion_trajectory(strategy_dir: Path) -> List[Structure]:
    """Load the full late insertion trajectory (200 steps from timestep 800-1000)."""
    trajectory_zip = strategy_dir / "generated_trajectories.zip"
    
    if not trajectory_zip.exists():
        print(f"Warning: No trajectory found for {strategy_dir.name}")
        return []
    
    print(f"Loading late insertion trajectory: {strategy_dir.name}")
    
    with zipfile.ZipFile(trajectory_zip, 'r') as zf:
        trajectory_files = [f for f in zf.namelist() if f.endswith('.extxyz')]
        
        if not trajectory_files:
            print(f"Warning: No trajectory files in {trajectory_zip}")
            return []
        
        # Use the first trajectory file
        chosen_file = trajectory_files[0]
        print(f"  Loading from: {chosen_file}")
        
        with zf.open(chosen_file) as f:
            content = io.StringIO(f.read().decode())
            atoms_list = ase.io.read(content, index=":", format="extxyz")
            
            print(f"  Found {len(atoms_list)} trajectory structures")
            
            structures = []
            # Take every 2nd structure to skip corrector steps, just like original
            for i in range(0, len(atoms_list), 2):
                try:
                    atoms = atoms_list[i]
                    structure = AseAtomsAdaptor.get_structure(atoms)
                    structures.append(structure)
                except Exception as e:
                    print(f"Warning: Could not convert structure at index {i}: {e}")
                    structures.append(None)
            
            print(f"  Successfully loaded {len([s for s in structures if s is not None])}/{len(structures)} structures")
            return structures


def analyze_structure_properties(structure: Structure) -> Dict[str, Any]:
    """Extract comprehensive structural properties."""
    if structure is None:
        return {'error': 'Structure is None'}
    
    try:
        # Basic properties
        properties = {
            'composition': str(structure.composition),
            'formula': structure.formula,
            'num_atoms': len(structure),
            'volume': structure.lattice.volume,
            'density': structure.density
        }
        
        # Lattice parameters
        lattice_params = structure.lattice.parameters
        properties.update({
            'a': lattice_params[0],
            'b': lattice_params[1], 
            'c': lattice_params[2],
            'alpha': lattice_params[3],
            'beta': lattice_params[4],
            'gamma': lattice_params[5]
        })
        
        # Space group analysis
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5.0)
            properties.update({
                'space_group_number': analyzer.get_space_group_number(),
                'space_group_symbol': analyzer.get_space_group_symbol(),
                'crystal_system': analyzer.get_crystal_system(),
                'lattice_type': analyzer.get_lattice_type(),
                'point_group': analyzer.get_point_group_symbol()
            })
        except Exception as e:
            print(f"Warning: Space group analysis failed: {e}")
            properties.update({
                'space_group_number': 1,
                'space_group_symbol': 'P1',
                'crystal_system': 'triclinic',
                'lattice_type': 'triclinic',
                'point_group': '1'
            })
        
        return properties
        
    except Exception as e:
        return {'error': f'Analysis failed: {e}'}


def calculate_trajectory_similarity(
    original_structures: List[Structure],
    conditioned_structures: List[Structure]
) -> Dict[str, List[float]]:
    """Calculate similarity metrics between two trajectories over time."""
    
    min_length = min(len(original_structures), len(conditioned_structures))
    
    similarity_metrics = {
        'timestep': [],
        'lattice_similarity': [],
        'volume_similarity': [],
        'density_similarity': [],
        'structural_match': [],
        'rms_distance': []
    }
    
    matcher = StructureMatcher(
        ltol=0.2, stol=0.3, angle_tol=5,
        primitive_cell=True, scale=True, attempt_supercell=True
    )
    
    for i in range(min_length):
        timestep = 800 + i  # Convert index to actual timestep
        similarity_metrics['timestep'].append(timestep)
        
        orig_struct = original_structures[i]
        cond_struct = conditioned_structures[i]
        
        if orig_struct is None or cond_struct is None:
            similarity_metrics['lattice_similarity'].append(None)
            similarity_metrics['volume_similarity'].append(None)
            similarity_metrics['density_similarity'].append(None)
            similarity_metrics['structural_match'].append(False)
            similarity_metrics['rms_distance'].append(None)
            continue
        
        # Lattice parameter similarity (average relative difference)
        orig_params = np.array(orig_struct.lattice.parameters[:3])  # a, b, c
        cond_params = np.array(cond_struct.lattice.parameters[:3])
        lattice_rel_diff = np.mean(np.abs(cond_params - orig_params) / orig_params)
        similarity_metrics['lattice_similarity'].append(1.0 - lattice_rel_diff)  # Convert to similarity
        
        # Volume similarity
        orig_vol = orig_struct.lattice.volume
        cond_vol = cond_struct.lattice.volume
        vol_rel_diff = abs(cond_vol - orig_vol) / orig_vol
        similarity_metrics['volume_similarity'].append(1.0 - vol_rel_diff)
        
        # Density similarity
        orig_dens = orig_struct.density
        cond_dens = cond_struct.density
        dens_rel_diff = abs(cond_dens - orig_dens) / orig_dens
        similarity_metrics['density_similarity'].append(1.0 - dens_rel_diff)
        
        # Structure matching
        try:
            is_match = matcher.fit(orig_struct, cond_struct)
            similarity_metrics['structural_match'].append(is_match)
            
            if is_match:
                rms_dist = matcher.get_rms_dist(orig_struct, cond_struct)
                similarity_metrics['rms_distance'].append(rms_dist[0] if rms_dist else None)
            else:
                similarity_metrics['rms_distance'].append(None)
        except Exception:
            similarity_metrics['structural_match'].append(False)
            similarity_metrics['rms_distance'].append(None)
    
    return similarity_metrics


def analyze_strategy_divergence(
    original_trajectory: List[Structure],
    strategy_dir: Path,
    strategy_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze how one conditioning strategy diverges from the original trajectory."""
    
    print(f"\nAnalyzing strategy: {strategy_dir.name}")
    
    # Load late insertion trajectory
    conditioned_trajectory = load_late_insertion_trajectory(strategy_dir)
    
    if not conditioned_trajectory:
        return {
            'strategy_name': strategy_dir.name,
            'success': False,
            'error': 'No trajectory found'
        }
    
    # Analyze trajectory similarity
    similarity_metrics = calculate_trajectory_similarity(original_trajectory, conditioned_trajectory)
    
    # Analyze final structures
    original_final = original_trajectory[-1] if original_trajectory else None
    conditioned_final = conditioned_trajectory[-1] if conditioned_trajectory else None
    
    final_comparison = {}
    if original_final and conditioned_final:
        orig_props = analyze_structure_properties(original_final)
        cond_props = analyze_structure_properties(conditioned_final)
        
        final_comparison = {
            'original_final': orig_props,
            'conditioned_final': cond_props,
            'property_changes': {
                'space_group_changed': (
                    orig_props.get('space_group_number') != cond_props.get('space_group_number')
                ),
                'crystal_system_changed': (
                    orig_props.get('crystal_system') != cond_props.get('crystal_system')
                ),
                'volume_change_pct': (
                    100 * (cond_props.get('volume', 0) - orig_props.get('volume', 0)) / orig_props.get('volume', 1)
                ) if orig_props.get('volume') else None
            }
        }
    
    # Check conditioning success
    conditioning_success = {}
    if 'strategy' in strategy_config and 'conditions' in strategy_config['strategy']:
        conditions = strategy_config['strategy']['conditions']
        
        if 'space_group' in conditions and conditioned_final:
            target_sg = conditions['space_group']
            actual_sg = cond_props.get('space_group_number')
            conditioning_success['space_group'] = {
                'target': target_sg,
                'actual': actual_sg,
                'achieved': target_sg == actual_sg
            }
    
    return {
        'strategy_name': strategy_dir.name,
        'success': True,
        'strategy_config': strategy_config,
        'trajectory_length': len(conditioned_trajectory),
        'similarity_metrics': similarity_metrics,
        'final_comparison': final_comparison,
        'conditioning_success': conditioning_success
    }


def create_divergence_plots(results: Dict[str, Any], output_dir: Path) -> None:
    """Create comprehensive trajectory divergence visualization."""
    
    plots_dir = output_dir / "divergence_plots"
    plots_dir.mkdir(exist_ok=True)
    
    successful_strategies = {name: result for name, result in results['strategy_analyses'].items() 
                           if result.get('success', False)}
    
    if not successful_strategies:
        print("No successful strategies to plot")
        return
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Trajectory similarity over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    similarity_types = ['lattice_similarity', 'volume_similarity', 'density_similarity']
    plot_configs = [
        ('lattice_similarity', 'Lattice Parameter Similarity', axes[0, 0]),
        ('volume_similarity', 'Volume Similarity', axes[0, 1]),
        ('density_similarity', 'Density Similarity', axes[1, 0])
    ]
    
    for sim_type, title, ax in plot_configs:
        for strategy_name, result in successful_strategies.items():
            if 'similarity_metrics' in result:
                metrics = result['similarity_metrics']
                timesteps = metrics.get('timestep', [])
                similarities = metrics.get(sim_type, [])
                
                # Filter out None values
                valid_data = [(t, s) for t, s in zip(timesteps, similarities) if s is not None]
                if valid_data:
                    t_vals, s_vals = zip(*valid_data)
                    ax.plot(t_vals, s_vals, 'o-', label=strategy_name.replace('_', ' '), markersize=3)
        
        ax.set_title(title)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Similarity (1.0 = identical)')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1.1)
    
    # 4th subplot: Structure matching success rate
    ax = axes[1, 1]
    for strategy_name, result in successful_strategies.items():
        if 'similarity_metrics' in result:
            metrics = result['similarity_metrics']
            timesteps = metrics.get('timestep', [])
            matches = metrics.get('structural_match', [])
            
            # Convert boolean to float for plotting
            match_rate = [float(m) if m is not None else 0.0 for m in matches]
            if timesteps and match_rate:
                ax.plot(timesteps, match_rate, 'o-', label=strategy_name.replace('_', ' '), markersize=3)
    
    ax.set_title('Structure Matching Success')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Match Success (1.0 = match, 0.0 = no match)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "trajectory_similarity_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Property evolution comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    properties = ['volume', 'density', 'a', 'b', 'c', 'space_group_number']
    property_titles = ['Volume (Ų)', 'Density (g/cm³)', 'Lattice a (Å)', 'Lattice b (Å)', 'Lattice c (Å)', 'Space Group']
    
    # First plot original trajectory
    original_props = results.get('original_trajectory_properties', {})
    
    for i, (prop, title) in enumerate(zip(properties, property_titles)):
        ax = axes[i]
        
        # Plot original trajectory
        if prop in original_props:
            timesteps = original_props['timestep']
            values = original_props[prop]
            valid_data = [(t, v) for t, v in zip(timesteps, values) if v is not None]
            if valid_data:
                t_vals, v_vals = zip(*valid_data)
                ax.plot(t_vals, v_vals, 'k-', linewidth=3, label='Original (Unconditioned)', alpha=0.7)
        
        # Plot conditioned trajectories  
        for strategy_name, result in successful_strategies.items():
            if 'trajectory_properties' in result:
                props = result['trajectory_properties']
                if prop in props:
                    timesteps = props['timestep']
                    values = props[prop]
                    valid_data = [(t, v) for t, v in zip(timesteps, values) if v is not None]
                    if valid_data:
                        t_vals, v_vals = zip(*valid_data)
                        ax.plot(t_vals, v_vals, 'o-', label=strategy_name.replace('_', ' '), markersize=2, alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlabel('Timestep')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "property_evolution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Divergence plots saved in: {plots_dir}")


def main():
    """Main analysis function."""
    
    # Configuration
    original_trajectory_path = "generated_structures/three_property_study/unconditional/generated_trajectories.zip"
    late_insertion_dir = Path("generated_structures/late_insertion_trajectories")
    output_dir = Path("generated_structures/trajectory_divergence_analysis")
    
    if not Path(original_trajectory_path).exists():
        print(f"Error: Original trajectory not found: {original_trajectory_path}")
        return
    
    if not late_insertion_dir.exists():
        print(f"Error: Late insertion directory not found: {late_insertion_dir}")
        return
    
    print("=" * 60)
    print("Trajectory Divergence Analysis")
    print("=" * 60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original trajectory segment (timesteps 800-1000)
    print("Loading original trajectory segment...")
    original_trajectory = load_trajectory_segment(
        original_trajectory_path, 
        trajectory_idx=0,
        start_timestep=800,
        end_timestep=1000
    )
    
    # Analyze original trajectory properties over time
    print("Analyzing original trajectory properties...")
    original_trajectory_properties = {'timestep': [], 'volume': [], 'density': [], 'a': [], 'b': [], 'c': [], 'space_group_number': []}
    
    for i, structure in enumerate(original_trajectory):
        timestep = 800 + i
        original_trajectory_properties['timestep'].append(timestep)
        
        if structure:
            props = analyze_structure_properties(structure)
            original_trajectory_properties['volume'].append(props.get('volume'))
            original_trajectory_properties['density'].append(props.get('density'))
            original_trajectory_properties['a'].append(props.get('a'))
            original_trajectory_properties['b'].append(props.get('b'))
            original_trajectory_properties['c'].append(props.get('c'))
            original_trajectory_properties['space_group_number'].append(props.get('space_group_number'))
        else:
            for key in ['volume', 'density', 'a', 'b', 'c', 'space_group_number']:
                original_trajectory_properties[key].append(None)
    
    # Find all strategy directories
    strategy_dirs = [d for d in late_insertion_dir.iterdir() 
                    if d.is_dir() and not d.name.startswith('.') and d.name != 'analysis_plots']
    
    print(f"\nFound {len(strategy_dirs)} strategy directories")
    
    # Analyze each strategy
    strategy_analyses = {}
    
    for strategy_dir in strategy_dirs:
        # Load strategy configuration
        strategy_config = {}
        config_path = strategy_dir / "strategy_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                strategy_config = json.load(f)
        
        # Analyze divergence
        result = analyze_strategy_divergence(original_trajectory, strategy_dir, strategy_config)
        strategy_analyses[strategy_dir.name] = result
        
        # Add trajectory properties analysis for successful strategies
        if result.get('success') and 'similarity_metrics' in result:
            conditioned_trajectory = load_late_insertion_trajectory(strategy_dir)
            if conditioned_trajectory:
                trajectory_props = {'timestep': [], 'volume': [], 'density': [], 'a': [], 'b': [], 'c': [], 'space_group_number': []}
                
                for i, structure in enumerate(conditioned_trajectory):
                    timestep = 800 + i
                    trajectory_props['timestep'].append(timestep)
                    
                    if structure:
                        props = analyze_structure_properties(structure)
                        trajectory_props['volume'].append(props.get('volume'))
                        trajectory_props['density'].append(props.get('density'))
                        trajectory_props['a'].append(props.get('a'))
                        trajectory_props['b'].append(props.get('b'))
                        trajectory_props['c'].append(props.get('c'))
                        trajectory_props['space_group_number'].append(props.get('space_group_number'))
                    else:
                        for key in ['volume', 'density', 'a', 'b', 'c', 'space_group_number']:
                            trajectory_props[key].append(None)
                
                result['trajectory_properties'] = trajectory_props
    
    # Compile results
    results = {
        'original_trajectory_length': len(original_trajectory),
        'original_trajectory_properties': original_trajectory_properties,
        'strategy_analyses': strategy_analyses,
        'summary_statistics': {
            'total_strategies': len(strategy_analyses),
            'successful_strategies': len([r for r in strategy_analyses.values() if r.get('success', False)]),
            'failed_strategies': len([r for r in strategy_analyses.values() if not r.get('success', False)])
        }
    }
    
    # Save results
    with open(output_dir / "trajectory_divergence_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create plots
    create_divergence_plots(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY DIVERGENCE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"Original trajectory segment: timesteps 800-1000 ({len(original_trajectory)} structures)")
    print(f"Strategies analyzed: {results['summary_statistics']['total_strategies']}")
    print(f"Successful: {results['summary_statistics']['successful_strategies']}")
    print(f"Failed: {results['summary_statistics']['failed_strategies']}")
    
    successful_strategies = {name: result for name, result in strategy_analyses.items() 
                           if result.get('success', False)}
    
    if successful_strategies:
        print(f"\nSuccessful strategies:")
        for strategy_name, result in successful_strategies.items():
            print(f"  - {strategy_name}: {result.get('trajectory_length', 0)} trajectory points")
            
            # Show conditioning success
            if 'conditioning_success' in result:
                for cond_type, cond_data in result['conditioning_success'].items():
                    status = "✓" if cond_data.get('achieved') else "✗"
                    print(f"    {cond_type}: {status} (target: {cond_data.get('target')}, actual: {cond_data.get('actual')})")
            
            # Show final similarity
            if 'similarity_metrics' in result:
                metrics = result['similarity_metrics']
                if metrics['lattice_similarity']:
                    final_sim = metrics['lattice_similarity'][-1]
                    if final_sim is not None:
                        print(f"    Final lattice similarity: {final_sim:.3f}")
    
    failed_strategies = {name: result for name, result in strategy_analyses.items() 
                        if not result.get('success', False)}
    
    if failed_strategies:
        print(f"\nFailed strategies:")
        for strategy_name, result in failed_strategies.items():
            print(f"  - {strategy_name}: {result.get('error', 'Unknown error')}")
    
    print(f"\nDetailed results saved in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()