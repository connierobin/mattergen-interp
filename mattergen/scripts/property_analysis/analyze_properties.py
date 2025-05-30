#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from mattergen.evaluation.reference.presets import get_reference_dataset
from .space_group_analyzer import SpaceGroupAnalyzer
from .energy_analyzer import EnergyAboveHullAnalyzer
from .structure_loader import (
    load_structures_auto, 
    load_structures_from_directory,
    discover_structure_files,
    create_structure_summary
)


def main():
    parser = argparse.ArgumentParser(description="Analyze properties of generated crystal structures")
    parser.add_argument("input_path", type=str, help="Path to structure file(s) or directory")
    parser.add_argument("--output_dir", type=str, default="property_analysis_results", 
                       help="Output directory for results")
    parser.add_argument("--reference_dataset", type=str, default="mp_20", 
                       help="Reference dataset for energy calculations (mp_20, alex_mp)")
    parser.add_argument("--analyze_space_group", action="store_true", default=True,
                       help="Analyze space group properties")
    parser.add_argument("--analyze_energy", action="store_true", default=False,
                       help="Analyze energy above hull (requires DFT energies)")
    parser.add_argument("--energy_file", type=str, 
                       help="JSON file with energy data (structure_id -> energy_per_atom)")
    parser.add_argument("--file_pattern", type=str, default="*.cif",
                       help="File pattern for directory search")
    parser.add_argument("--symprec", type=float, default=0.01,
                       help="Symmetry precision for space group analysis")
    parser.add_argument("--angle_tolerance", type=float, default=5.0,
                       help="Angle tolerance for space group analysis")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load structures
    input_path = Path(args.input_path)
    if input_path.is_file():
        structures = load_structures_auto(input_path)
        print(f"Loaded {len(structures)} structures from {input_path}")
    elif input_path.is_dir():
        structures = load_structures_from_directory(input_path, args.file_pattern)
        print(f"Loaded {len(structures)} structures from {input_path}")
        
        # Also discover what files are available
        discovered = discover_structure_files(input_path)
        print("Discovered files:")
        for file_type, files in discovered.items():
            print(f"  {file_type}: {len(files)} files")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    if not structures:
        print("No structures found!")
        return
    
    # Create structure summary
    summary_df = create_structure_summary(structures)
    summary_df.to_csv(output_dir / "structure_summary.csv", index=False)
    print(f"Structure summary saved to {output_dir / 'structure_summary.csv'}")
    
    # Load energy data if provided
    energy_data = {}
    if args.energy_file:
        with open(args.energy_file, 'r') as f:
            energy_data = json.load(f)
        print(f"Loaded energy data for {len(energy_data)} structures")
    
    all_results = []
    
    # Analyze space groups
    if args.analyze_space_group:
        print("\nAnalyzing space groups...")
        space_group_analyzer = SpaceGroupAnalyzer(
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance
        )
        
        structures_list = [s[0] for s in structures]
        structure_ids = [s[1] for s in structures]
        
        sg_results = space_group_analyzer.analyze_structures(structures_list, structure_ids)
        sg_results.to_csv(output_dir / "space_group_analysis.csv", index=False)
        print(f"Space group analysis saved to {output_dir / 'space_group_analysis.csv'}")
        
        all_results.append(sg_results)
    
    # Analyze energy above hull
    if args.analyze_energy:
        print("\nAnalyzing energy above hull...")
        try:
            reference_dataset = get_reference_dataset(args.reference_dataset)
            energy_analyzer = EnergyAboveHullAnalyzer(reference_dataset)
            
            energy_results = []
            for structure, structure_id in structures:
                energy_per_atom = energy_data.get(structure_id, None)
                if energy_per_atom is None:
                    print(f"Warning: No energy data for {structure_id}, skipping energy analysis")
                    continue
                
                result = energy_analyzer.analyze_structure(structure, structure_id, energy_per_atom)
                result['structure_id'] = structure_id
                result['analyzer'] = 'energy_above_hull'
                energy_results.append(result)
            
            if energy_results:
                energy_df = pd.DataFrame(energy_results)
                energy_df.to_csv(output_dir / "energy_analysis.csv", index=False)
                print(f"Energy analysis saved to {output_dir / 'energy_analysis.csv'}")
                all_results.append(energy_df)
            else:
                print("No energy analyses completed - check energy data file")
                
        except Exception as e:
            print(f"Error in energy analysis: {e}")
    
    # Combine all results if multiple analyses were run
    if len(all_results) > 1:
        # Merge on structure_id
        combined_df = all_results[0]
        for df in all_results[1:]:
            combined_df = combined_df.merge(df, on='structure_id', how='outer', suffixes=('', '_y'))
        
        combined_df.to_csv(output_dir / "combined_analysis.csv", index=False)
        print(f"Combined analysis saved to {output_dir / 'combined_analysis.csv'}")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()