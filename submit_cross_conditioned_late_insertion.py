#!/usr/bin/env python3
"""
Generate and submit job scripts for cross-conditioned late insertion trajectories.

This script creates SLURM job files for all combinations of source->target conditions
and timesteps, allowing parallel execution on the cluster.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any


def get_within_category_conditioning_pairs() -> Dict[str, List[str]]:
    """Get source->target pairs within each property category."""
    return {
        # DFT Band Gap category: insulator -> other electronic properties
        "dft_band_gap_insulator": [
            "dft_band_gap_metallic",
            "dft_band_gap_narrow_semiconductor", 
            "dft_band_gap_wide_semiconductor"
        ],
        
        # Energy Above Hull category: metastable -> other stability levels
        "energy_above_hull_metastable": [
            "energy_above_hull_stable",
            "energy_above_hull_unstable"
        ],
        
        # Space Group category: fcc -> other crystal structures
        "space_group_cubic_fcc": [
            "space_group_cubic_primitive",
            "space_group_hexagonal", 
            "space_group_tetragonal",
            "space_group_orthorhombic"
        ]
    }


def create_job_script(source_condition: str, target_condition: str, timestep: int, 
                     trajectory_idx: int = 0, job_dir: Path = None) -> str:
    """Create a SLURM job script for a specific cross-conditioning run."""
    
    if job_dir is None:
        job_dir = Path("jobs/cross_conditioned_late_insertion")
    
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive job name
    job_name = f"cross_cond_{source_condition}_to_{target_condition}_step{timestep}_traj{trajectory_idx}"
    
    # Create SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_dir}/{job_name}.out
#SBATCH --error={job_dir}/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=88G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=connietherobin@gmail.com

# Activate your Python environment
source .venv/bin/activate

# Set environment variables for your authentication tokens
export WANDB_API_KEY="[REMOVED]"
export HF_TOKEN="[REMOVED]"

# Print job info
echo "========================================"
echo "Job: {job_name}"
echo "Source condition: {source_condition}"
echo "Target condition: {target_condition}"
echo "Start timestep: {timestep}"
echo "Trajectory index: {trajectory_idx}"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"

# Run the cross-conditioning script
python mattergen/scripts/run_cross_conditioned_late_insertion.py \\
    --source_condition {source_condition} \\
    --target_condition {target_condition} \\
    --start_timestep {timestep} \\
    --trajectory_idx {trajectory_idx} \\
    --total_timesteps 1000 \\
    --output_base_dir generated_structures/cross_conditioned_late_insertion

# Print completion info
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
"""
    
    # Write script to file
    script_path = job_dir / f"{job_name}.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return str(script_path)


def generate_all_cross_conditioning_jobs(timesteps: List[int] = [600, 800, 900, 950],
                                       trajectory_indices: List[int] = [0],
                                       job_dir: Path = None) -> List[Dict[str, Any]]:
    """Generate job scripts for within-category cross-conditioning combinations."""
    
    if job_dir is None:
        job_dir = Path("jobs/cross_conditioned_late_insertion")
    
    conditioning_pairs = get_within_category_conditioning_pairs()
    job_info = []
    
    # Calculate total jobs
    total_jobs = 0
    for source, targets in conditioning_pairs.items():
        total_jobs += len(targets) * len(timesteps) * len(trajectory_indices)
    
    print(f"Generating {total_jobs} job scripts for within-category cross-conditioning...")
    print(f"Timesteps: {timesteps}")
    print(f"Trajectory indices: {trajectory_indices}")
    print("Categories:")
    for source, targets in conditioning_pairs.items():
        print(f"  {source} -> {len(targets)} targets")
    
    job_count = 0
    for source, targets in conditioning_pairs.items():
        for target in targets:
            for timestep in timesteps:
                for traj_idx in trajectory_indices:
                    job_count += 1
                    
                    print(f"[{job_count}/{total_jobs}] Creating job: {source} -> {target}, step {timestep}, traj {traj_idx}")
                    
                    script_path = create_job_script(
                        source_condition=source,
                        target_condition=target,
                        timestep=timestep,
                        trajectory_idx=traj_idx,
                        job_dir=job_dir
                    )
                    
                    job_info.append({
                        "job_id": job_count,
                        "source_condition": source,
                        "target_condition": target,
                        "timestep": timestep,
                        "trajectory_idx": traj_idx,
                        "script_path": script_path,
                        "job_name": f"cross_cond_{source}_to_{target}_step{timestep}_traj{traj_idx}"
                    })
    
    return job_info


def create_submission_script(job_info: List[Dict[str, Any]], 
                           batch_size: int = 10,
                           job_dir: Path = None) -> str:
    """Create a script to submit all jobs, optionally in batches."""
    
    if job_dir is None:
        job_dir = Path("jobs/cross_conditioned_late_insertion")
    
    submission_script_path = job_dir / "submit_all_cross_conditioning_jobs.sh"
    
    script_content = """#!/bin/bash
# Submit all cross-conditioned late insertion trajectory jobs

echo "Submitting cross-conditioned late insertion trajectory jobs..."
echo "Total jobs to submit: """ + str(len(job_info)) + f"""
echo "Batch size: {batch_size}"
echo ""

# Function to submit a batch of jobs
submit_batch() {{
    local batch_start=$1
    local batch_end=$2
    echo "Submitting batch: jobs $batch_start to $batch_end"
    
"""
    
    # Add job submissions in batches
    for i, job in enumerate(job_info):
        if i % batch_size == 0 and i > 0:
            script_content += """
    # Wait a bit between batches to avoid overwhelming the scheduler
    echo "Waiting 30 seconds before next batch..."
    sleep 30
"""
        
        script_content += f"""    echo "Submitting job {i+1}/{len(job_info)}: {job['job_name']}"
    sbatch {job['script_path']}
"""
    
    script_content += """
}

# Submit all jobs
submit_batch 1 """ + str(len(job_info)) + """

echo ""
echo "All jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "Check job outputs in: """ + str(job_dir) + """
"""
    
    with open(submission_script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(submission_script_path, 0o755)
    
    return str(submission_script_path)


def create_monitoring_script(job_info: List[Dict[str, Any]], job_dir: Path = None) -> str:
    """Create a script to monitor job progress."""
    
    if job_dir is None:
        job_dir = Path("jobs/cross_conditioned_late_insertion")
    
    monitoring_script_path = job_dir / "monitor_jobs.py"
    
    script_content = f"""#!/usr/bin/env python3
'''
Monitor cross-conditioned late insertion trajectory jobs.
'''

import os
import json
from pathlib import Path
from collections import defaultdict

def check_job_status():
    '''Check status of all cross-conditioning jobs.'''
    
    job_dir = Path("{job_dir}")
    output_dir = Path("generated_structures/cross_conditioned_late_insertion")
    
    total_jobs = {len(job_info)}
    completed = 0
    failed = 0
    running = 0
    
    status_by_timestep = defaultdict(lambda: {{"completed": 0, "failed": 0, "running": 0}})
    status_by_source = defaultdict(lambda: {{"completed": 0, "failed": 0, "running": 0}})
    
    print(f"Checking status of {{total_jobs}} cross-conditioning jobs...")
    print("="*60)
    
    for job_info_item in {job_info}:
        job_name = job_info_item["job_name"]
        source = job_info_item["source_condition"]
        target = job_info_item["target_condition"]
        timestep = job_info_item["timestep"]
        traj_idx = job_info_item["trajectory_idx"]
        
        # Check if result file exists
        result_dir = output_dir / f"step_{{timestep}}" / source / target / f"trajectory_{{traj_idx}}"
        result_file = result_dir / "result_summary.json"
        
        if result_file.exists():
            try:
                with open(result_file) as f:
                    result = json.load(f)
                if result.get("success", False):
                    completed += 1
                    status_by_timestep[timestep]["completed"] += 1
                    status_by_source[source]["completed"] += 1
                else:
                    failed += 1
                    status_by_timestep[timestep]["failed"] += 1
                    status_by_source[source]["failed"] += 1
            except:
                failed += 1
                status_by_timestep[timestep]["failed"] += 1
                status_by_source[source]["failed"] += 1
        else:
            running += 1
            status_by_timestep[timestep]["running"] += 1
            status_by_source[source]["running"] += 1
    
    print(f"Overall Status:")
    print(f"  Completed: {{completed}}/{{total_jobs}} ({{completed/total_jobs*100:.1f}}%)")
    print(f"  Failed:    {{failed}}/{{total_jobs}} ({{failed/total_jobs*100:.1f}}%)")
    print(f"  Running:   {{running}}/{{total_jobs}} ({{running/total_jobs*100:.1f}}%)")
    print()
    
    print("Status by timestep:")
    for timestep in sorted(status_by_timestep.keys()):
        stats = status_by_timestep[timestep]
        total = sum(stats.values())
        print(f"  Step {{timestep}}: {{stats['completed']}}/{{total}} completed, {{stats['failed']}}/{{total}} failed, {{stats['running']}}/{{total}} running")
    print()
    
    print("Status by source condition:")
    for source in sorted(status_by_source.keys()):
        stats = status_by_source[source]
        total = sum(stats.values())
        print(f"  {{source}}: {{stats['completed']}}/{{total}} completed, {{stats['failed']}}/{{total}} failed, {{stats['running']}}/{{total}} running")

if __name__ == "__main__":
    check_job_status()
"""
    
    with open(monitoring_script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(monitoring_script_path, 0o755)
    
    return str(monitoring_script_path)


def main():
    """Main function to generate all job scripts."""
    
    # Configuration
    timesteps = [600, 800, 900, 950]  # As requested
    trajectory_indices = [0]  # Start with first trajectory from each condition
    job_dir = Path("jobs/cross_conditioned_late_insertion")
    
    print("Within-Category Cross-Conditioned Late Insertion Job Generator")
    print("=" * 60)
    print(f"Timesteps to test: {timesteps}")
    print(f"Trajectory indices: {trajectory_indices}")
    print(f"Job directory: {job_dir}")
    print()
    
    # Show the conditioning pairs
    conditioning_pairs = get_within_category_conditioning_pairs()
    print("Within-category conditioning pairs:")
    for source, targets in conditioning_pairs.items():
        print(f"  {source}:")
        for target in targets:
            print(f"    -> {target}")
    print()
    
    # Generate all job scripts
    job_info = generate_all_cross_conditioning_jobs(
        timesteps=timesteps,
        trajectory_indices=trajectory_indices,
        job_dir=job_dir
    )
    
    print(f"\\nGenerated {len(job_info)} job scripts in {job_dir}")
    
    # Save job manifest
    job_manifest_path = job_dir / "job_manifest.json"
    with open(job_manifest_path, "w") as f:
        json.dump(job_info, f, indent=2)
    print(f"Job manifest saved to: {job_manifest_path}")
    
    # Create submission script
    submission_script = create_submission_script(job_info, batch_size=10, job_dir=job_dir)
    print(f"Submission script created: {submission_script}")
    
    # Create monitoring script
    monitoring_script = create_monitoring_script(job_info, job_dir=job_dir)
    print(f"Monitoring script created: {monitoring_script}")
    
    print()
    print("Next steps:")
    print(f"1. Review job scripts in: {job_dir}")
    print(f"2. Submit all jobs: {submission_script}")
    print(f"3. Monitor progress: python {monitoring_script}")
    print()
    print("File organization:")
    print("Results will be saved in:")
    print("  generated_structures/cross_conditioned_late_insertion/")
    print("    step_<timestep>/")
    print("      <source_condition>/")
    print("        <target_condition>/")
    print("          trajectory_<idx>/")
    print("            - experiment_config.json")
    print("            - run_config.json") 
    print("            - result_summary.json")
    print("            - generated_structures/")
    print("            - generated_trajectories.zip")


if __name__ == "__main__":
    main()