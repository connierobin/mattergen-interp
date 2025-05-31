#!/usr/bin/env python3
"""
Analyze DFT Band Gap Results

Collects and analyzes all DFT band gap calculation results.
"""

import json
import glob
from pathlib import Path
import pandas as pd

def collect_results():
    """Collect all DFT results into a summary."""
    results = []
    
    for result_file in glob.glob("dft_results/*/*/*/*/band_gap_results.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            for structure_result in data['results']:
                if structure_result.get('converged', False):
                    results.append(structure_result)
                    
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    return results

def main():
    print("📊 Analyzing DFT Band Gap Results")
    print("=" * 40)
    
    results = collect_results()
    
    if not results:
        print("No results found yet. Run DFT calculations first.")
        return
    
    print(f"Found {len(results)} successful calculations")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    print("\nSummary by Source → Target:")
    summary = df.groupby(['source_condition', 'target_condition']).agg({
        'estimated_band_gap_ev': ['count', 'mean', 'std'],
        'is_metal': 'sum'
    }).round(3)
    
    print(summary)
    
    # Save detailed results
    df.to_csv('dft_band_gap_analysis.csv', index=False)
    print(f"\n📁 Detailed results saved to: dft_band_gap_analysis.csv")

if __name__ == "__main__":
    main()
