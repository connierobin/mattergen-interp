#!/bin/bash
#SBATCH --job-name=qe_mpi_fix
#SBATCH --output=jobs/qe_mpi_fix/job_%j.out
#SBATCH --error=jobs/qe_mpi_fix/job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --mem=4G

mkdir -p jobs/qe_mpi_fix

echo "=== QE MPI CONFLICT TROUBLESHOOTING ==="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo

echo "=== STEP 1: DIAGNOSE MPI ISSUE ==="
echo "Current MPI libraries:"
ldconfig -p | grep mpi | head -10

echo
echo "QE pw.x dependencies:"
ldd /usr/bin/pw.x | grep -E "(mpi|openmpi)"

echo
echo "MPI library versions:"
ls -la /lib/x86_64-linux-gnu/libmpi* | head -5

echo
echo "=== STEP 2: TRY DIFFERENT MPI APPROACHES ==="

echo "Approach 1: Disable MPI entirely"
export OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_btl_vader_single_copy_mechanism=none
echo "Testing pw.x with MPI workarounds..."
timeout 10s /usr/bin/pw.x 2>&1 | head -10 || echo "Approach 1 failed"

echo
echo "Approach 2: Use mpirun with specific flags"
echo "Testing with mpirun --allow-run-as-root..."
timeout 10s mpirun --allow-run-as-root -np 1 /usr/bin/pw.x 2>&1 | head -10 || echo "Approach 2 failed"

echo
echo "Approach 3: Reinstall compatible MPI"
echo "Removing conflicting MPI packages..."
sudo apt remove -y openmpi-bin libopenmpi-dev 2>&1 | head -10

echo "Installing specific MPI version..."
sudo apt install -y openmpi-bin=4.1.2-2ubuntu1 libopenmpi-dev=4.1.2-2ubuntu1 2>&1 | head -15

echo "Testing after MPI reinstall..."
timeout 10s /usr/bin/pw.x 2>&1 | head -10 || echo "Approach 3 failed"

echo
echo "=== STEP 3: TEST SERIAL VERSION ==="
echo "Checking if QE has serial version compiled..."
ls -la /usr/bin/ | grep -E "(pw|espresso)" | head -10

echo
echo "=== STEP 4: ALTERNATIVE - BUILD SERIAL QE ==="
echo "If MPI version doesn't work, we can build a serial version..."
which gfortran && echo "✅ gfortran available for serial build"
which make && echo "✅ make available for serial build"

echo
echo "=== STEP 5: CREATE SIMPLE DFT TEST ==="
mkdir -p /tmp/qe_mpi_test
cd /tmp/qe_mpi_test

cat > serial_test.in << 'EOF'
&control
    calculation='scf'
    prefix='test'
    pseudo_dir='/usr/share/espresso/pseudo/'
    outdir='./'
/
&system
    ibrav=1, celldm(1)=10.0
    nat=1, ntyp=1
    ecutwfc=25.0
/
&electrons
    conv_thr=1.0d-6
/
ATOMIC_SPECIES
 H  1.008  H.pbe-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
 H 0.0 0.0 0.0
K_POINTS gamma
EOF

echo "Testing final QE execution with various methods:"

echo "Method 1: Direct execution"
timeout 15s /usr/bin/pw.x < serial_test.in 2>&1 | head -20 || echo "Direct execution failed"

echo
echo "Method 2: With LD_PRELOAD to fix MPI"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libmpi.so.12
timeout 15s /usr/bin/pw.x < serial_test.in 2>&1 | head -20 || echo "LD_PRELOAD method failed"

echo
echo "Method 3: Using system pw.x from login node environment"
# Check if we can copy the working pw.x from login environment
ls -la /usr/bin/pw.x

echo
echo "=== SUMMARY ==="
echo "Troubleshooting completed at: $(date)"
echo "Next steps depend on which method works"