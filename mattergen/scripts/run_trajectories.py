from pathlib import Path
import json
from typing import Optional
from mattergen.generator import CrystalGenerator
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo

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

def generate_crystals(
    output_dir: str = "generated_structures",
    batch_size: int = 32,
    num_batches: int = 1,
    record_trajectories: bool = True,
    save_chemgraphs: bool = False,
    run_dir: Optional[str] = None
):
    """
    Generate crystal structures using the base MatterGen model.
    
    Args:
        output_dir: Base directory where generated structures will be saved
        batch_size: Number of structures to generate in parallel
        num_batches: How many batches to generate
        record_trajectories: Whether to save the generation trajectories
        save_chemgraphs: Whether to save ChemGraphs throughout trajectories
        run_dir: Optional specific run directory name (e.g., "run001")
                If None, will automatically create next numbered directory
    """
    # Create base output directory
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Determine run directory
    if run_dir is None:
        run_number = get_next_run_number(base_path)
        run_dir = f"run{run_number:03d}"
    
    # Create run-specific output directory
    output_path = base_path / run_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration
    config = {
        "batch_size": batch_size,
        "num_batches": num_batches,
        "record_trajectories": record_trajectories,
        "save_chemgraphs": save_chemgraphs
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize generator with base model
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo.from_hf_hub("mattergen_base"),
        batch_size=batch_size,
        num_batches=num_batches,
        record_trajectories=record_trajectories
    )
    
    # Generate structures
    generated_structures = generator.generate(output_dir=output_path)
    print(f"\nGenerated {len(generated_structures)} structures")
    print(f"Results saved in: {output_path}")
    return generated_structures

if __name__ == "__main__":
    generate_crystals(record_trajectories=True, num_batches=100, save_chemgraphs=True)