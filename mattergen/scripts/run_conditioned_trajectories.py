#!/usr/bin/env python3
"""
Generate crystal structures conditioned on space_group, energy_above_hull, formation_energy_per_atom, and dft_band_gap.

This script creates runs with different conditioning values for the 4 properties of interest:
1. space_group (categorical) - Different crystal symmetries  
2. energy_above_hull (continuous) - Thermodynamic stability
3. formation_energy_per_atom (continuous) - Formation energetics
4. dft_band_gap (continuous) - Electronic band gap

Results are organized in generated_structures/ with descriptive folder names for easy analysis.
"""

from pathlib import Path
import json
from typing import Dict, Any, List
from mattergen.generator import CrystalGenerator
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.data_classes import PRETRAINED_MODEL_NAME
from typing import cast

def get_next_run_number(base_dir: Path) -> int:
    """Find the next available run number by checking existing directories."""
    existing_runs = [d for d in base_dir.glob("run*") if d.is_dir()]
    if not existing_runs:
        return 1
    
    # Extract run numbers and find max
    run_numbers = []
    for run_dir in existing_runs:
        try:
            num = int(run_dir.name[3:])  # Extract number from "runXXX"
            run_numbers.append(num)
        except ValueError:
            continue
    
    return max(run_numbers, default=0) + 1

def get_available_conditioning_strategies() -> List[Dict[str, Any]]:
    """
    Get conditioning strategies for the 4 properties of interest:
    - space_group: Crystallographic space groups (categorical)
    - energy_above_hull: Thermodynamic stability (continuous)
    - formation_energy_per_atom: Formation energetics (continuous)
    - dft_band_gap: Electronic band gap (continuous)
    """
    
    strategies = [
        # Baseline unconditional
        {
            "name": "unconditional",
            "checkpoint_path": "checkpoints/mattergen_base",
            "conditions": {},
            "description": "Unconditional baseline generation"
        },
        
        # Space group conditioning - diverse crystal systems
        {
            "name": "space_group_cubic_fcc",
            "checkpoint_path": "checkpoints/space_group", 
            "conditions": {"space_group": 225},  # Fm-3m (FCC)
            "description": "Space group 225 (Fm-3m, FCC cubic)"
        },
        {
            "name": "space_group_cubic_primitive",
            "checkpoint_path": "checkpoints/space_group",
            "conditions": {"space_group": 221},  # Pm-3m
            "description": "Space group 221 (Pm-3m, primitive cubic)"
        },
        {
            "name": "space_group_hexagonal",
            "checkpoint_path": "checkpoints/space_group",
            "conditions": {"space_group": 194},  # P63/mmc
            "description": "Space group 194 (P63/mmc, hexagonal)"
        },
        {
            "name": "space_group_tetragonal",
            "checkpoint_path": "checkpoints/space_group",
            "conditions": {"space_group": 139},  # I4/mmm
            "description": "Space group 139 (I4/mmm, tetragonal)"
        },
        {
            "name": "space_group_orthorhombic",
            "checkpoint_path": "checkpoints/space_group",
            "conditions": {"space_group": 62},   # Pnma
            "description": "Space group 62 (Pnma, orthorhombic)"
        },
        
        # Energy above hull conditioning (thermodynamic stability)
        {
            "name": "energy_above_hull_stable",
            "checkpoint_path": "checkpoints/chemical_system_energy_above_hull",
            "conditions": {"energy_above_hull": 0.0},  # Perfectly stable
            "description": "Perfectly stable (0.0 eV/atom above hull)"
        },
        {
            "name": "energy_above_hull_metastable",
            "checkpoint_path": "checkpoints/chemical_system_energy_above_hull",
            "conditions": {"energy_above_hull": 0.1},  # Stability threshold
            "description": "Marginally stable (0.1 eV/atom above hull)"
        },
        {
            "name": "energy_above_hull_unstable",
            "checkpoint_path": "checkpoints/chemical_system_energy_above_hull",
            "conditions": {"energy_above_hull": 0.3},  # Unstable
            "description": "Unstable (0.3 eV/atom above hull)"
        },
        
        # Formation energy per atom conditioning (uses base model as fallback)
        # Note: This uses base model since no specific formation_energy_per_atom model exists
        {
            "name": "formation_energy_favorable",
            "checkpoint_path": "checkpoints/mattergen_base",  # Fallback to base model
            "conditions": {"formation_energy_per_atom": -1.0},  # Favorable formation
            "description": "Favorable formation (-1.0 eV/atom)"
        },
        {
            "name": "formation_energy_neutral",
            "checkpoint_path": "checkpoints/mattergen_base",  # Fallback to base model  
            "conditions": {"formation_energy_per_atom": 0.0},  # Neutral
            "description": "Neutral formation energy (0.0 eV/atom)"
        },
        {
            "name": "formation_energy_unfavorable",
            "checkpoint_path": "checkpoints/mattergen_base",  # Fallback to base model
            "conditions": {"formation_energy_per_atom": 0.5},  # Unfavorable
            "description": "Unfavorable formation (+0.5 eV/atom)"
        },
        
        # DFT band gap conditioning (electronic properties)
        {
            "name": "dft_band_gap_metallic",
            "checkpoint_path": "checkpoints/dft_band_gap",
            "conditions": {"dft_band_gap": 0.0},  # Metallic (zero band gap)
            "description": "Metallic (0.0 eV band gap)"
        },
        {
            "name": "dft_band_gap_narrow_semiconductor",
            "checkpoint_path": "checkpoints/dft_band_gap",
            "conditions": {"dft_band_gap": 1.0},  # Narrow gap semiconductor
            "description": "Narrow gap semiconductor (1.0 eV band gap)"
        },
        {
            "name": "dft_band_gap_wide_semiconductor",
            "checkpoint_path": "checkpoints/dft_band_gap",
            "conditions": {"dft_band_gap": 3.0},  # Wide gap semiconductor
            "description": "Wide gap semiconductor (3.0 eV band gap)"
        },
        {
            "name": "dft_band_gap_insulator",
            "checkpoint_path": "checkpoints/dft_band_gap",
            "conditions": {"dft_band_gap": 6.0},  # Wide band gap insulator
            "description": "Wide band gap insulator (6.0 eV band gap)"
        },
    ]
    
    return strategies

def generate_conditioned_crystals(
    output_base_dir: str = "generated_structures",
    experiment_name: str = "conditioned_generation",
    batch_size: int = 16,
    num_batches: int = 3,
    record_trajectories: bool = True
):
    """
    Generate crystal structures with various property conditions using local checkpoints.
    
    Args:
        output_base_dir: Base directory for all generated structures
        experiment_name: Name of this specific experiment  
        batch_size: Number of structures to generate in parallel per condition
        num_batches: How many batches to generate per condition
        record_trajectories: Whether to save the generation trajectories
    """
    
    # Create organized output directory structure
    base_path = Path(output_base_dir)
    experiment_path = base_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # Get all conditioning strategies
    strategies = get_available_conditioning_strategies()
    
    print(f"Conditioned Crystal Generation Experiment")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Output directory: {experiment_path}")
    print(f"Strategies to test: {len(strategies)}")
    print(f"Structures per strategy: {batch_size * num_batches}")
    print(f"Total structures: {len(strategies) * batch_size * num_batches}")
    print(f"Record trajectories: {record_trajectories}")
    print(f"{'='*60}")
    
    # Track results
    results = {
        "experiment_name": experiment_name,
        "total_strategies": len(strategies),
        "batch_size": batch_size,
        "num_batches": num_batches,
        "record_trajectories": record_trajectories,
        "results": []
    }
    
    # Generate for each strategy
    for i, strategy in enumerate(strategies):
        strategy_name = strategy["name"]
        output_path = experiment_path / strategy_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{i+1}/{len(strategies)}] {strategy['description']}")
        print(f"Strategy: {strategy_name}")
        print(f"Checkpoint: {strategy['checkpoint_path']}")
        print(f"Conditions: {strategy['conditions']}")
        print(f"Output: {output_path}")
        
        # Save strategy configuration
        strategy_config = {
            "strategy": strategy,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "record_trajectories": record_trajectories
        }
        with open(output_path / "strategy_config.json", "w") as f:
            json.dump(strategy_config, f, indent=2)
        
        try:
            # Use Hugging Face Hub models instead of local checkpoints
            model_name_mapping = {
                "checkpoints/mattergen_base": "mattergen_base",
                "checkpoints/chemical_system": "chemical_system", 
                "checkpoints/space_group": "space_group",
                "checkpoints/chemical_system_energy_above_hull": "chemical_system_energy_above_hull",
                "checkpoints/dft_mag_density": "dft_mag_density",
                "checkpoints/dft_band_gap": "dft_band_gap",
                "checkpoints/ml_bulk_modulus": "ml_bulk_modulus",
                "checkpoints/dft_mag_density_hhi_score": "dft_mag_density_hhi_score",
            }
            
            checkpoint_path = strategy["checkpoint_path"]
            if checkpoint_path in model_name_mapping:
                # Use Hugging Face Hub
                hf_model_name = model_name_mapping[checkpoint_path]
                checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(cast(PRETRAINED_MODEL_NAME, hf_model_name))
                print(f"Using Hugging Face model: {hf_model_name}")
            else:
                # Fallback to local path (will likely fail)
                checkpoint_info = MatterGenCheckpointInfo(
                    model_path=str(Path(checkpoint_path).resolve()),
                    load_epoch="last"
                )
                print(f"Using local checkpoint: {checkpoint_path}")
            
            # Initialize generator
            generator = CrystalGenerator(
                checkpoint_info=checkpoint_info,
                properties_to_condition_on=strategy["conditions"],
                batch_size=batch_size,
                num_batches=num_batches,
                record_trajectories=record_trajectories
            )
            
            # Generate structures
            generated_structures = generator.generate(output_dir=str(output_path))
            
            result = {
                "strategy_name": strategy_name,
                "success": True,
                "num_generated": len(generated_structures),
                "conditions": strategy["conditions"],
                "checkpoint_path": strategy["checkpoint_path"]
            }
            
            print(f"✓ Successfully generated {len(generated_structures)} structures")
            
        except Exception as e:
            result = {
                "strategy_name": strategy_name,
                "success": False,
                "error": str(e),
                "conditions": strategy["conditions"],
                "checkpoint_path": strategy["checkpoint_path"]
            }
            
            print(f"✗ Error: {str(e)}")
            
            # Save error details
            with open(output_path / "error.txt", "w") as f:
                f.write(f"Error: {str(e)}\n")
                f.write(f"Strategy: {strategy}\n")
        
        results["results"].append(result)
    
    # Save overall experiment results
    with open(experiment_path / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results["results"] if r["success"])
    failed = len(results["results"]) - successful
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete!")
    print(f"Successful strategies: {successful}/{len(strategies)}")
    print(f"Failed strategies: {failed}")
    if successful > 0:
        total_structures = sum(r.get("num_generated", 0) for r in results["results"] if r["success"])
        print(f"Total structures generated: {total_structures}")
    print(f"Results saved in: {experiment_path}")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    # Run the full experiment
    results = generate_conditioned_crystals(
        output_base_dir="generated_structures",
        experiment_name="three_property_study",
        batch_size=8,     # Moderate size for good statistics
        num_batches=2,    # 2 batches = 16 structures per condition
        record_trajectories=True
    )
    
    # Print final summary
    successful_strategies = [r for r in results["results"] if r["success"]]
    if successful_strategies:
        print(f"\nSuccessful conditioning strategies:")
        for result in successful_strategies:
            print(f"  - {result['strategy_name']}: {result['num_generated']} structures")
    
    failed_strategies = [r for r in results["results"] if not r["success"]]
    if failed_strategies:
        print(f"\nFailed strategies:")
        for result in failed_strategies:
            print(f"  - {result['strategy_name']}: {result['error']}")