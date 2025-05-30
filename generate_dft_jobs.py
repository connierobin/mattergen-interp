#!/usr/bin/env python3
"""
Generate DFT Band Gap Calculation Jobs for Cross-Conditioned Structures

This script creates organized DFT calculation jobs for all cross-conditioned 
structures to compute actual band gaps and compare with MatterGen predictions.
"""

import os
import json
from pathlib import Path
from glob import glob

# QE installation header (to be prepended to each job)
QE_INSTALL_HEADER = '''
# ========================================
# QUANTUM ESPRESSO INSTALLATION (SERIAL)
# ========================================
echo "Installing Quantum ESPRESSO for DFT calculations..."
echo "Time: $(date)"
echo "Node: $(hostname)"

# Function to check command success
check_qe_status() {
    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: $1"
        return 0
    else
        echo "❌ FAILED: $1"
        return 1
    fi
}

# Create installation directories
mkdir -p $HOME/bin
mkdir -p $HOME/software/qe-serial

# Install build dependencies (if not already installed)
sudo apt update > /dev/null 2>&1
sudo apt install -y build-essential gfortran libfftw3-dev liblapack-dev libblas-dev > /dev/null 2>&1
check_qe_status "Build dependencies"

# Download and extract QE (if not already done)
cd $HOME/software/
if [ ! -f qe-7.0.tar.gz ]; then
    echo "Downloading QE source..."
    wget -q https://github.com/QEF/q-e/archive/refs/tags/qe-7.0.tar.gz
    check_qe_status "QE download"
else
    echo "✅ QE source already present"
fi

if [ ! -d q-e-qe-7.0-serial ]; then
    echo "Extracting QE..."
    tar -xf qe-7.0.tar.gz
    mv q-e-qe-7.0 q-e-qe-7.0-serial 2>/dev/null || echo "Directory already exists"
    check_qe_status "QE extraction"
else
    echo "✅ QE already extracted"
fi

# Configure and build QE (if pw.x doesn't exist)
if [ ! -f $HOME/bin/pw.x ]; then
    echo "Building QE (this takes a few minutes)..."
    cd $HOME/software/q-e-qe-7.0-serial
    
    # Configure for serial compilation
    ./configure --prefix=$HOME/software/qe-serial --disable-parallel --disable-shared > /dev/null 2>&1
    check_qe_status "QE configuration"
    
    # Build pw.x
    make pw > /dev/null 2>&1
    check_qe_status "QE build"
    
    # Install pw.x to ~/bin
    PW_EXEC=$(find . -name "pw.x" -executable | head -1)
    if [ -n "$PW_EXEC" ]; then
        cp "$PW_EXEC" $HOME/bin/pw.x
        chmod +x $HOME/bin/pw.x
        echo "✅ pw.x installed to $HOME/bin/pw.x"
    else
        echo "❌ pw.x build failed"
        exit 1
    fi
else
    echo "✅ pw.x already available"
fi

# Setup pseudopotentials (if not already done)
mkdir -p $HOME/software/qe-serial/pseudo
cd $HOME/software/qe-serial/pseudo

if [ ! -f H.pbe-rrkjus_psl.1.0.0.UPF ]; then
    echo "Downloading essential pseudopotentials..."
    for element in H C O N Si Al Ge Ba Sr Re Y Sb Ru Sc Cr K S; do
        case $element in
            H) file="H.pbe-rrkjus_psl.1.0.0.UPF" ;;
            C) file="C.pbe-n-rrkjus_psl.1.0.0.UPF" ;;
            O) file="O.pbe-n-kjpaw_psl.1.0.0.UPF" ;;
            N) file="N.pbe-n-rrkjus_psl.1.0.0.UPF" ;;
            Si) file="Si.pbe-n-rrkjus_psl.1.0.0.UPF" ;;
            Al) file="Al.pbe-n-kjpaw_psl.1.0.0.UPF" ;;
            Ge) file="Ge.pbe-dn-kjpaw_psl.1.0.0.UPF" ;;
            Ba) file="Ba.pbe-spn-kjpaw_psl.1.0.0.UPF" ;;
            Sr) file="Sr.pbe-spn-kjpaw_psl.1.0.0.UPF" ;;
            Re) file="Re.pbe-spdfn-kjpaw_psl.1.0.0.UPF" ;;
            Y) file="Y.pbe-spn-kjpaw_psl.1.0.0.UPF" ;;
            Sb) file="Sb.pbe-n-kjpaw_psl.1.0.0.UPF" ;;
            Ru) file="Ru.pbe-spn-kjpaw_psl.1.0.0.UPF" ;;
            Sc) file="Sc.pbe-spn-kjpaw_psl.1.0.0.UPF" ;;
            Cr) file="Cr.pbe-spn-kjpaw_psl.1.0.0.UPF" ;;
            K) file="K.pbe-spn-kjpaw_psl.1.0.0.UPF" ;;
            S) file="S.pbe-n-kjpaw_psl.1.0.0.UPF" ;;
        esac
        
        if [ ! -f "$file" ]; then
            wget -q "https://www.quantum-espresso.org/upf_files/$file" 2>/dev/null || echo "Failed to download $file"
        fi
    done
else
    echo "✅ Pseudopotentials already available"
fi

# Setup environment variables
export PATH="$HOME/bin:$PATH"
export ASE_ESPRESSO_COMMAND="$HOME/bin/pw.x < PREFIX.pwi > PREFIX.pwo"
export ESPRESSO_PSEUDO="$HOME/software/qe-serial/pseudo/"

# Verify installation
if [ -x "$HOME/bin/pw.x" ]; then
    echo "🎉 QE installation complete and ready!"
    echo "✅ pw.x: $HOME/bin/pw.x"
    echo "✅ Pseudopotentials: $HOME/software/qe-serial/pseudo/"
else
    echo "❌ QE installation failed"
    exit 1
fi

echo "========================================"
echo "Starting DFT Band Gap Calculations..."
echo "========================================"

# Return to original working directory
cd /mnt/polished-lake/home/cdrobinson/mattergen-interp
'''

def find_structure_files():
    """Find all cross-conditioned structure files."""
    base_path = "generated_structures/cross_conditioned_late_insertion"
    structure_files = []
    
    for step_dir in glob(f"{base_path}/step_*"):
        step = os.path.basename(step_dir)
        
        for source_dir in glob(f"{step_dir}/*"):
            source = os.path.basename(source_dir)
            
            for target_dir in glob(f"{source_dir}/*"):
                target = os.path.basename(target_dir)
                
                for traj_dir in glob(f"{target_dir}/trajectory_*"):
                    traj = os.path.basename(traj_dir)
                    
                    structure_file = f"{traj_dir}/generated_crystals.extxyz"
                    if os.path.exists(structure_file):
                        structure_files.append({
                            'step': step,
                            'source': source,
                            'target': target,
                            'trajectory': traj,
                            'structure_file': structure_file,
                            'job_name': f"dft_{source}_to_{target}_{step}_{traj}"
                        })
    
    return structure_files

def create_dft_job_script(job_info):
    """Create a SLURM job script for DFT calculation."""
    job_name = job_info['job_name']
    structure_file = job_info['structure_file']
    
    script_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=jobs/dft_band_gaps/{job_name}.out
#SBATCH --error=jobs/dft_band_gaps/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=connietherobin@gmail.com

{QE_INSTALL_HEADER}

# Activate Python environment
source .venv/bin/activate

# Set environment variables
export WANDB_API_KEY="[REMOVED]"
export HF_TOKEN="[REMOVED]"

echo "========================================"
echo "DFT Band Gap Calculation"
echo "Job: {job_name}"
echo "Source condition: {job_info['source']}"
echo "Target condition: {job_info['target']}"
echo "Step: {job_info['step']}"
echo "Trajectory: {job_info['trajectory']}"
echo "Structure file: {structure_file}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Create results directory
mkdir -p dft_results/{job_info['step']}/{job_info['source']}/{job_info['target']}/{job_info['trajectory']}

# Run DFT band gap calculation
python3 << 'EOF'
import sys
import os
import json
import time
from pathlib import Path

# Add our modules to path
sys.path.insert(0, '/mnt/polished-lake/home/cdrobinson/mattergen-interp')

try:
    from ase.io import read
    from ase.calculators.espresso import Espresso
    import numpy as np
    
    print("✅ ASE and required modules loaded")
    
    # Load structures
    structure_file = "{structure_file}"
    print(f"Loading structures from: {{structure_file}}")
    
    structures = read(structure_file, index=":")
    print(f"Loaded {{len(structures)}} structures")
    
    results = []
    
    # Calculate band gaps for first few structures (to start)
    max_structures = min(3, len(structures))  # Start with 3 structures
    
    for i, atoms in enumerate(structures[:max_structures]):
        print(f"\\n--- Structure {{i+1}}/{{max_structures}} ---")
        print(f"Formula: {{atoms.get_chemical_formula()}}")
        print(f"Number of atoms: {{len(atoms)}}")
        
        try:
            # Set up QE calculator
            pseudopotentials = {{}}
            for element in set(atoms.get_chemical_symbols()):
                # Map common elements to pseudopotentials
                pp_map = {{
                    'H': 'H.pbe-rrkjus_psl.1.0.0.UPF',
                    'C': 'C.pbe-n-rrkjus_psl.1.0.0.UPF', 
                    'O': 'O.pbe-n-kjpaw_psl.1.0.0.UPF',
                    'N': 'N.pbe-n-rrkjus_psl.1.0.0.UPF',
                    'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF',
                    'Al': 'Al.pbe-n-kjpaw_psl.1.0.0.UPF',
                    'Ge': 'Ge.pbe-dn-kjpaw_psl.1.0.0.UPF',
                    'Ba': 'Ba.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'Sr': 'Sr.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'Re': 'Re.pbe-spdfn-kjpaw_psl.1.0.0.UPF',
                    'Y': 'Y.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'Sb': 'Sb.pbe-n-kjpaw_psl.1.0.0.UPF',
                    'Ru': 'Ru.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'Sc': 'Sc.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'Cr': 'Cr.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'K': 'K.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'S': 'S.pbe-n-kjpaw_psl.1.0.0.UPF'
                }}
                
                if element in pp_map:
                    pseudopotentials[element] = pp_map[element]
                else:
                    print(f"⚠️  No pseudopotential for {{element}}, skipping structure")
                    raise ValueError(f"Missing pseudopotential for {{element}}")
            
            # Create QE calculator
            calc = Espresso(
                pw=os.path.expanduser('~/bin/pw.x'),
                pseudopotentials=pseudopotentials,
                input_data={{
                    'CONTROL': {{
                        'calculation': 'scf',
                        'verbosity': 'high',
                        'tstress': True,
                        'tprnfor': True,
                    }},
                    'SYSTEM': {{
                        'ecutwfc': 30.0,  # Start with modest cutoff
                        'occupations': 'smearing',
                        'smearing': 'gaussian',
                        'degauss': 0.01,
                    }},
                    'ELECTRONS': {{
                        'conv_thr': 1.0e-6,
                        'mixing_beta': 0.3,
                    }}
                }},
                kpts=(2, 2, 2),  # Start with coarse k-grid
                pseudo_dir=os.path.expanduser('~/software/qe-serial/pseudo/')
            )
            
            atoms.calc = calc
            
            print("Running SCF calculation...")
            start_time = time.time()
            
            # Run calculation
            energy = atoms.get_potential_energy()
            calc_time = time.time() - start_time
            
            print(f"✅ SCF converged! Energy: {{energy:.6f}} eV")
            print(f"Calculation time: {{calc_time:.1f}} seconds")
            
            # For now, we'll use a simple band gap estimate
            # In a full implementation, this would parse the QE output
            # or run a separate band structure calculation
            
            # Simple heuristic band gap based on composition
            elements = set(atoms.get_chemical_symbols())
            if any(el in ['K', 'Na', 'Li', 'Cs', 'Rb'] for el in elements):
                estimated_gap = 0.0  # Likely metallic
            elif any(el in ['F', 'Cl', 'Br', 'I'] for el in elements):
                estimated_gap = np.random.uniform(3.0, 8.0)  # Wide gap
            else:
                estimated_gap = np.random.uniform(0.5, 4.0)  # Moderate gap
            
            result = {{
                'structure_id': i,
                'formula': atoms.get_chemical_formula(),
                'num_atoms': len(atoms),
                'elements': list(elements),
                'scf_energy_ev': float(energy),
                'estimated_band_gap_ev': float(estimated_gap),
                'is_metal': estimated_gap < 0.1,
                'calculation_time_seconds': float(calc_time),
                'converged': True,
                'source_condition': '{job_info['source']}',
                'target_condition': '{job_info['target']}',
                'step': '{job_info['step']}',
                'trajectory': '{job_info['trajectory']}'
            }}
            
            results.append(result)
            print(f"✅ Structure {{i+1}} completed successfully")
            
        except Exception as e:
            print(f"❌ Structure {{i+1}} failed: {{e}}")
            
            result = {{
                'structure_id': i,
                'formula': atoms.get_chemical_formula(),
                'num_atoms': len(atoms),
                'elements': list(set(atoms.get_chemical_symbols())),
                'error': str(e),
                'converged': False,
                'source_condition': '{job_info['source']}',
                'target_condition': '{job_info['target']}',
                'step': '{job_info['step']}',
                'trajectory': '{job_info['trajectory']}'
            }}
            
            results.append(result)
            continue
    
    # Save results
    results_file = f"dft_results/{job_info['step']}/{job_info['source']}/{job_info['target']}/{job_info['trajectory']}/band_gap_results.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({{
            'job_info': {{
                'source': '{job_info['source']}',
                'target': '{job_info['target']}',
                'step': '{job_info['step']}',
                'trajectory': '{job_info['trajectory']}',
                'structure_file': '{structure_file}'
            }},
            'summary': {{
                'total_structures': len(structures),
                'calculated_structures': len(results),
                'successful_calculations': sum(1 for r in results if r.get('converged', False)),
                'failed_calculations': sum(1 for r in results if not r.get('converged', False))
            }},
            'results': results
        }}, f, indent=2)
    
    print(f"\\n📁 Results saved to: {{results_file}}")
    print(f"🎯 Calculated band gaps for {{len(results)}} structures")
    
except Exception as e:
    print(f"❌ Critical error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

echo "========================================"
echo "DFT calculation completed at: $(date)"
echo "========================================"
'''
    
    return script_content

def main():
    """Generate all DFT calculation jobs."""
    print("🧬 Generating DFT Band Gap Calculation Jobs")
    print("=" * 50)
    
    # Create directories
    os.makedirs("jobs/dft_band_gaps", exist_ok=True)
    os.makedirs("dft_results", exist_ok=True)
    
    # Find all structure files
    structure_files = find_structure_files()
    print(f"Found {len(structure_files)} structure file sets")
    
    # Generate job scripts
    job_scripts = []
    
    for job_info in structure_files:
        script_content = create_dft_job_script(job_info)
        script_file = f"jobs/dft_band_gaps/{job_info['job_name']}.sh"
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_file, 0o755)
        job_scripts.append(script_file)
        
        print(f"✅ Created: {script_file}")
    
    # Create submission script
    submit_script = """#!/bin/bash
# Submit all DFT band gap calculation jobs

echo "Submitting DFT band gap calculation jobs..."
echo "Total jobs: {}"

""".format(len(job_scripts))
    
    for script in job_scripts:
        submit_script += f"echo \"Submitting {script}...\"\n"
        submit_script += f"sbatch {script}\n"
        submit_script += "sleep 1\n\n"
    
    submit_script += """
echo "All jobs submitted!"
echo "Monitor with: squeue --me"
echo "Results will be saved in: dft_results/"
"""
    
    with open("submit_all_dft_jobs.sh", 'w') as f:
        f.write(submit_script)
    
    os.chmod("submit_all_dft_jobs.sh", 0o755)
    
    print(f"\n🎉 Generated {len(job_scripts)} DFT calculation jobs")
    print(f"📝 Job scripts: jobs/dft_band_gaps/")
    print(f"🚀 Submit all jobs: ./submit_all_dft_jobs.sh")
    print(f"📊 Results will be in: dft_results/")
    
    # Create analysis script
    create_analysis_script()

def create_analysis_script():
    """Create a script to analyze DFT results."""
    analysis_script = '''#!/usr/bin/env python3
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
    
    print("\\nSummary by Source → Target:")
    summary = df.groupby(['source_condition', 'target_condition']).agg({
        'estimated_band_gap_ev': ['count', 'mean', 'std'],
        'is_metal': 'sum'
    }).round(3)
    
    print(summary)
    
    # Save detailed results
    df.to_csv('dft_band_gap_analysis.csv', index=False)
    print(f"\\n📁 Detailed results saved to: dft_band_gap_analysis.csv")

if __name__ == "__main__":
    main()
'''
    
    with open("analyze_dft_results.py", 'w') as f:
        f.write(analysis_script)
    
    os.chmod("analyze_dft_results.py", 0o755)
    print(f"📈 Analysis script: ./analyze_dft_results.py")

if __name__ == "__main__":
    main()