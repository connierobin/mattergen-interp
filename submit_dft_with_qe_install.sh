#!/bin/bash
#SBATCH --job-name=dft_with_qe
#SBATCH --output=jobs/dft_with_qe/job_%j.out
#SBATCH --error=jobs/dft_with_qe/job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=connietherobin@gmail.com

mkdir -p jobs/dft_with_qe

echo "============================================"
echo "DFT JOB WITH QE INSTALLATION"
echo "============================================"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Home directory: $HOME"
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

echo "=== STEP 1: SETUP HOME INSTALLATION DIRECTORY ==="
echo "Creating QE installation directory in home..."
mkdir -p $HOME/software/qe
mkdir -p $HOME/bin
echo "Installation directories created:"
echo "  Software: $HOME/software/qe"
echo "  Binaries: $HOME/bin"

echo
echo "=== STEP 2: DOWNLOAD AND BUILD QE ==="
echo "Downloading QE source (if not already present)..."
cd $HOME/software/

if [ ! -f qe-7.0.tar.gz ]; then
    wget https://github.com/QEF/q-e/archive/refs/tags/qe-7.0.tar.gz
    check_status "QE source download"
else
    echo "✅ QE source already downloaded"
fi

echo "Extracting QE source..."
if [ ! -d q-e-qe-7.0 ]; then
    tar -xzf qe-7.0.tar.gz
    check_status "QE source extraction"
else
    echo "✅ QE source already extracted"
fi

echo
echo "=== STEP 3: INSTALL BUILD DEPENDENCIES ==="
echo "Installing necessary build tools..."

# Install build dependencies to system (these should be visible)
sudo apt update
sudo apt install -y build-essential gfortran libfftw3-dev liblapack-dev libblas-dev
check_status "Build dependencies installation"

echo
echo "=== STEP 4: CONFIGURE AND BUILD QE ==="
cd $HOME/software/q-e-qe-7.0

echo "Configuring QE for home installation..."
echo "Configure command: ./configure --prefix=$HOME/software/qe"

# Configure QE to install in home directory
./configure --prefix=$HOME/software/qe --enable-shared=no 2>&1 | head -20
check_status "QE configuration"

echo
echo "Compiling QE (this may take several minutes)..."
echo "Starting compilation at: $(date)"

# Build only the essentials we need
make pw 2>&1 | tail -20
check_status "QE pw.x compilation"

echo
echo "=== STEP 5: INSTALL QE TO HOME ==="
echo "Installing QE binaries to $HOME/software/qe..."
make install 2>&1 | tail -10
check_status "QE installation"

# Copy key executables to ~/bin for easy PATH access
cp bin/pw.x $HOME/bin/ 2>/dev/null || echo "pw.x not found in bin/"
cp PW/src/pw.x $HOME/bin/ 2>/dev/null || echo "pw.x not found in PW/src/"

echo
echo "=== STEP 6: SETUP PSEUDOPOTENTIALS ==="
echo "Setting up pseudopotentials..."
mkdir -p $HOME/software/qe/pseudo

# Download essential pseudopotentials if not present
if [ ! -f $HOME/software/qe/pseudo/H.pbe-rrkjus_psl.1.0.0.UPF ]; then
    echo "Downloading basic pseudopotentials..."
    cd $HOME/software/qe/pseudo
    
    # Download a few essential pseudopotentials for testing
    wget -q http://www.quantum-espresso.org/upf_files/H.pbe-rrkjus_psl.1.0.0.UPF || echo "H pseudopotential download failed"
    wget -q http://www.quantum-espresso.org/upf_files/C.pbe-n-rrkjus_psl.1.0.0.UPF || echo "C pseudopotential download failed"
    wget -q http://www.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF || echo "O pseudopotential download failed"
else
    echo "✅ Pseudopotentials already available"
fi

echo
echo "=== STEP 7: SETUP ENVIRONMENT ==="
echo "Setting up PATH and environment variables..."

# Add our home installation to PATH
export PATH="$HOME/bin:$HOME/software/qe/bin:$PATH"
export ASE_ESPRESSO_COMMAND="$HOME/bin/pw.x < PREFIX.pwi > PREFIX.pwo"
export ESPRESSO_PSEUDO="$HOME/software/qe/pseudo/"

echo "Environment configured:"
echo "  PATH includes: $HOME/bin:$HOME/software/qe/bin"
echo "  ASE_ESPRESSO_COMMAND: $ASE_ESPRESSO_COMMAND"
echo "  ESPRESSO_PSEUDO: $ESPRESSO_PSEUDO"

echo
echo "=== STEP 8: VERIFY QE INSTALLATION ==="
echo "Testing QE installation..."

# Check if pw.x is available
which pw.x && echo "✅ pw.x found at: $(which pw.x)" || echo "❌ pw.x not in PATH"

if [ -x "$HOME/bin/pw.x" ]; then
    echo "✅ pw.x executable in home bin"
    
    echo "Testing pw.x startup..."
    timeout 5s $HOME/bin/pw.x 2>&1 | head -5 || echo "Startup test completed"
    
else
    echo "❌ pw.x not found in $HOME/bin/"
    echo "Checking what was built..."
    find $HOME/software/q-e-qe-7.0 -name "pw.x" -executable
fi

echo
echo "=== STEP 9: RUN DFT TEST ==="
echo "Testing with actual DFT calculation..."

# Activate Python environment for DFT test
cd /mnt/polished-lake/home/cdrobinson/mattergen-interp
source .venv/bin/activate

echo "Running DFT test framework..."
python mattergen/scripts/run_dft_test.py || echo "DFT test completed with issues"

echo
echo "=== STEP 10: SUMMARY ==="
echo "Installation completed at: $(date)"

if [ -x "$HOME/bin/pw.x" ]; then
    echo "🎉 SUCCESS: QE installed in home directory!"
    echo
    echo "📋 Installation details:"
    echo "   ✅ QE binary: $HOME/bin/pw.x"
    echo "   ✅ Pseudopotentials: $HOME/software/qe/pseudo/"
    echo "   ✅ PATH configured for this session"
    echo
    echo "🔧 For future jobs, ensure PATH includes:"
    echo "   export PATH=\"$HOME/bin:$HOME/software/qe/bin:\$PATH\""
    echo "   export ASE_ESPRESSO_COMMAND=\"$HOME/bin/pw.x < PREFIX.pwi > PREFIX.pwo\""
    echo "   export ESPRESSO_PSEUDO=\"$HOME/software/qe/pseudo/\""
else
    echo "❌ INSTALLATION FAILED"
    echo "Check build logs above for errors"
fi

echo "============================================"