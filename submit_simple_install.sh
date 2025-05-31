#!/bin/bash
#SBATCH --job-name=simple_install
#SBATCH --output=jobs/simple_install/job_%j.out
#SBATCH --error=jobs/simple_install/job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --mem=2G

mkdir -p jobs/simple_install

echo "=== SIMPLE DEPENDENCY INSTALL ==="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo

echo "Attempting to install gfortran and cmake..."
echo "Current user: $(whoami)"
echo "Sudo available: $(which sudo || echo 'No sudo')"

# Try installing the missing packages
apt install gfortran cmake -y 2>&1 || echo "Standard install failed - trying with different approach"

echo
echo "=== POST-INSTALL CHECK ==="
which gfortran && echo "✅ gfortran: $(gfortran --version | head -1)" || echo "❌ gfortran still missing"
which cmake && echo "✅ cmake: $(cmake --version | head -1)" || echo "❌ cmake still missing"

echo
echo "Completed at: $(date)"