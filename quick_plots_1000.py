#!/usr/bin/env python3
"""
Quick plotting script for 1000 structure analysis - skips bond analysis
"""

import sys
sys.path.append('/mnt/polished-lake/home/cdrobinson/mattergen-interp')

from pathlib import Path
from mattergen.scripts.diffusion_step_analysis.unpack_results import (
    load_trajectories, analyze_final_structures, print_lattice_analysis,
    plot_finalization_curves, plot_lattice_analysis, plot_num_atoms_distribution,
    plot_individual_trajectories, analyze_species_changes, 
    analyze_element_finalization_timing, print_element_timing_analysis
)

def main():
    output_dir = Path("generated_structures/run003")
    max_trajectories = 100  # Use smaller subset for faster plotting
    
    print(f"\nQuick plotting analysis in: {output_dir}")
    print(f"Loading {max_trajectories} trajectories for plotting...")
    
    trajectories = load_trajectories(output_dir, max_trajectories)
    print(f"Loaded {len(trajectories)} trajectories")
    
    print("\nAnalyzing final structures...")
    final_structures = [traj[-1] for traj in trajectories]
    lattice_analyses = analyze_final_structures(trajectories)
    print_lattice_analysis(lattice_analyses)
    
    print("\nCreating analysis plots...")
    plot_finalization_curves(trajectories, output_dir)
    plot_lattice_analysis(lattice_analyses, output_dir)
    plot_num_atoms_distribution(output_dir)
    
    print("\nCreating individual trajectory plots (subset)...")
    # Just plot first 10 trajectories to save time
    plot_individual_trajectories(trajectories[:10], output_dir)
    
    print("\nAnalyzing species changes...")
    analyze_species_changes(trajectories, output_dir)
    
    print("\nAnalyzing element finalization timing...")
    timing_analysis = analyze_element_finalization_timing(trajectories)
    print_element_timing_analysis(timing_analysis)
    
    print(f"\nPlots saved in {output_dir}/plots/")
    print("Quick analysis complete!")

if __name__ == "__main__":
    main()