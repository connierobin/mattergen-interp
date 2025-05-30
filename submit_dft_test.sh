#!/bin/bash
#SBATCH --job-name=dft_test
#SBATCH --output=jobs/dft_test/job_%j.out
#SBATCH --error=jobs/dft_test/job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=connietherobin@gmail.com

# Create job output directory
mkdir -p jobs/dft_test

# Activate Python environment
source .venv/bin/activate

# Set environment variables
export WANDB_API_KEY="[REMOVED]"
export HF_TOKEN="[REMOVED]"

# Check if QE is available on compute node
echo "Starting DFT band gap calculation test..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

echo "Checking software availability on compute node:"
echo "PATH: $PATH"
which pw.x && echo "✅ pw.x found at: $(which pw.x)" || echo "❌ pw.x not found"
which vasp && echo "✅ vasp found at: $(which vasp)" || echo "❌ vasp not found"
which vasp_std && echo "✅ vasp_std found at: $(which vasp_std)" || echo "❌ vasp_std not found"

# Try to add common paths
export PATH="/usr/bin:/usr/local/bin:$PATH"
echo "After PATH update:"
which pw.x && echo "✅ pw.x found at: $(which pw.x)" || echo "❌ pw.x still not found"

# Check if QE exists but isn't in PATH
echo "Checking for QE in common locations:"
ls -la /usr/bin/pw.x 2>/dev/null && echo "✅ Found /usr/bin/pw.x" || echo "❌ No /usr/bin/pw.x"
ls -la /usr/local/bin/pw.x 2>/dev/null && echo "✅ Found /usr/local/bin/pw.x" || echo "❌ No /usr/local/bin/pw.x"

# Check if any QE-related packages are installed
echo "Checking for QE-related files:"
ls -la /usr/bin/q*x 2>/dev/null || echo "No quantum-espresso executables found"

# Try module system (common on clusters)
echo "Checking module system:"
which module && module avail quantum 2>&1 || echo "No module system or no quantum modules"

# Configure ASE for Quantum ESPRESSO
export ASE_ESPRESSO_COMMAND="pw.x < PREFIX.pwi > PREFIX.pwo"

# Load modules if needed (cluster-specific)
# module load quantum-espresso  # Uncomment if cluster uses modules

# Run the DFT test script
python mattergen/scripts/run_dft_test.py

echo "DFT test completed at: $(date)"