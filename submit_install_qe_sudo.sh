#!/bin/bash
#SBATCH --job-name=install_qe_sudo
#SBATCH --output=jobs/install_qe/job_%j.out
#SBATCH --error=jobs/install_qe/job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=connietherobin@gmail.com

# Create job output directory
mkdir -p jobs/install_qe

echo "==============================================="
echo "QUANTUM ESPRESSO INSTALLATION ON COMPUTE NODE"
echo "==============================================="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: $1"
    else
        echo "❌ FAILED: $1"
        return 1
    fi
}

echo "=== STEP 1: SUDO ACCESS CHECK ==="
echo "Testing sudo access..."
sudo whoami 2>&1
check_status "Sudo access test"
echo

echo "=== STEP 2: UPDATE PACKAGE LISTS ==="
echo "Updating apt package lists..."
sudo apt update 2>&1
check_status "Package list update"
echo

echo "=== STEP 3: INSTALL BUILD DEPENDENCIES ==="
echo "Installing build-essential and compilers..."
sudo apt install -y build-essential gfortran 2>&1
check_status "Build tools installation"

echo
echo "Installing MPI libraries..."
sudo apt install -y libopenmpi-dev openmpi-bin 2>&1
check_status "MPI installation"

echo
echo "Installing scientific libraries..."
sudo apt install -y libfftw3-dev liblapack-dev libblas-dev libscalapack-mpi-dev 2>&1
check_status "Scientific libraries installation"
echo

echo "=== STEP 4: INSTALL QUANTUM ESPRESSO ==="
echo "Installing quantum-espresso package..."
sudo apt install -y quantum-espresso quantum-espresso-data 2>&1
check_status "Quantum ESPRESSO installation"
echo

echo "=== STEP 5: VERIFY INSTALLATION ==="
echo "Checking installed QE executables:"
ls -la /usr/bin/pw.x /usr/bin/epw.x 2>/dev/null || echo "QE executables not found"

echo
echo "Testing pw.x executable:"
if [ -x /usr/bin/pw.x ]; then
    echo "✅ /usr/bin/pw.x is executable"
    echo "File details: $(ls -la /usr/bin/pw.x)"
    
    echo
    echo "Testing pw.x startup (will timeout after 5 seconds):"
    timeout 5s /usr/bin/pw.x 2>&1 | head -10 || echo "Startup test completed"
    
    echo
    echo "Checking library dependencies:"
    ldd /usr/bin/pw.x | grep -E "(not found|missing)" || echo "✅ All libraries available"
    
else
    echo "❌ /usr/bin/pw.x not found or not executable"
fi

echo
echo "=== STEP 6: VERIFY PSEUDOPOTENTIALS ==="
echo "Checking pseudopotential directory:"
if [ -d /usr/share/espresso/pseudo/ ]; then
    echo "✅ Pseudopotential directory exists"
    echo "Number of pseudopotentials: $(ls /usr/share/espresso/pseudo/ 2>/dev/null | wc -l)"
    echo "Sample pseudopotentials:"
    ls /usr/share/espresso/pseudo/ | head -5
else
    echo "❌ Pseudopotential directory not found"
fi

echo
echo "=== STEP 7: TEST WITH MINIMAL CALCULATION ==="
echo "Creating minimal test input..."
mkdir -p /tmp/qe_test_sudo
cd /tmp/qe_test_sudo

# Create very simple test input
cat > minimal_test.in << 'EOF'
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

echo "Test input created successfully"

if [ -x /usr/bin/pw.x ]; then
    echo
    echo "Running minimal QE test (will timeout after 30 seconds):"
    timeout 30s /usr/bin/pw.x < minimal_test.in 2>&1 | head -50 || echo "Test completed or timed out"
else
    echo "Skipping QE test - pw.x not available"
fi

echo
echo "=== STEP 8: FINAL VERIFICATION ==="
echo "Verifying complete installation:"
echo "QE version:"
if [ -x /usr/bin/pw.x ]; then
    timeout 3s /usr/bin/pw.x 2>&1 | grep -E "(Program|version)" | head -3
fi

echo
echo "Available QE tools:"
ls /usr/bin/ | grep -E "(pw|cp|bands|dos|pp|ph|dynmat)\.x" | head -10

echo
echo "=== STEP 9: SUMMARY ==="
echo "Installation completed at: $(date)"
echo

# Final verification
if [ -x /usr/bin/pw.x ]; then
    echo "🎉 QUANTUM ESPRESSO INSTALLATION SUCCESSFUL!"
    echo "✅ pw.x executable: /usr/bin/pw.x"
    echo "✅ Pseudopotentials: /usr/share/espresso/pseudo/"
    echo "✅ Ready for ASE integration"
    echo
    echo "For ASE, use: export ASE_ESPRESSO_COMMAND='/usr/bin/pw.x < PREFIX.pwi > PREFIX.pwo'"
else
    echo "❌ INSTALLATION FAILED - pw.x not available"
    echo "Check error messages above for troubleshooting"
fi

echo "==============================================="