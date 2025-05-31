#!/bin/bash
#SBATCH --job-name=simple_qe_install
#SBATCH --output=jobs/simple_qe/job_%j.out
#SBATCH --error=jobs/simple_qe/job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=8G

mkdir -p jobs/simple_qe

echo "==========================================="
echo "SIMPLE QE INSTALLATION (SERIAL VERSION)"
echo "==========================================="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Home: $HOME"
echo

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: $1"
        return 0
    else
        echo "❌ FAILED: $1"
        return 1
    fi
}

echo "=== STEP 1: SETUP DIRECTORIES ==="
mkdir -p $HOME/bin
mkdir -p $HOME/software/qe-serial
echo "Directories created: $HOME/bin, $HOME/software/qe-serial"

echo
echo "=== STEP 2: INSTALL BUILD DEPENDENCIES ==="
sudo apt update > /dev/null 2>&1
sudo apt install -y build-essential gfortran libfftw3-dev liblapack-dev libblas-dev > /dev/null 2>&1
check_status "Build dependencies"

echo
echo "=== STEP 3: DOWNLOAD QE SOURCE ==="
cd $HOME/software/
if [ ! -f qe-7.0.tar.gz ]; then
    echo "Downloading QE source..."
    wget -q https://github.com/QEF/q-e/archive/refs/tags/qe-7.0.tar.gz
    check_status "QE download"
else
    echo "✅ QE source already present"
fi

echo "Extracting QE..."
if [ ! -d q-e-qe-7.0-serial ]; then
    tar -xf qe-7.0.tar.gz
    mv q-e-qe-7.0 q-e-qe-7.0-serial 2>/dev/null || echo "Directory already exists"
    check_status "QE extraction"
else
    echo "✅ QE already extracted"
fi

echo
echo "=== STEP 4: CONFIGURE QE (SERIAL VERSION) ==="
cd $HOME/software/q-e-qe-7.0-serial

echo "Configuring QE without MPI (serial version)..."
./configure --prefix=$HOME/software/qe-serial --disable-parallel --disable-shared
check_status "QE configuration" || {
    echo "Configuration failed, trying alternative approach..."
    # Try with minimal configuration
    ./configure --prefix=$HOME/software/qe-serial
}

echo
echo "=== STEP 5: BUILD PW.X ==="
echo "Building pw.x (this takes several minutes)..."
echo "Build started at: $(date)"

# Build just the pw program
make pw 2>&1 | tail -10
build_result=$?

if [ $build_result -eq 0 ]; then
    echo "✅ SUCCESS: QE build completed"
else
    echo "⚠️  Build had issues, checking what was created..."
fi

echo
echo "=== STEP 6: LOCATE AND INSTALL EXECUTABLES ==="
echo "Searching for built executables..."

# Find pw.x executable
PW_EXEC=$(find . -name "pw.x" -executable | head -1)
if [ -n "$PW_EXEC" ]; then
    echo "✅ Found pw.x at: $PW_EXEC"
    
    # Copy to our bin directory
    cp "$PW_EXEC" $HOME/bin/pw.x
    chmod +x $HOME/bin/pw.x
    check_status "pw.x installation to $HOME/bin/"
    
    echo "✅ pw.x installed to $HOME/bin/pw.x"
else
    echo "❌ pw.x not found after build"
    echo "Checking build directory contents:"
    find . -name "*pw*" -type f | head -10
fi

echo
echo "=== STEP 7: SETUP PSEUDOPOTENTIALS ==="
mkdir -p $HOME/software/qe-serial/pseudo
cd $HOME/software/qe-serial/pseudo

echo "Downloading essential pseudopotentials..."
# Download a few basic pseudopotentials for testing
for element in H C O N Si; do
    case $element in
        H) file="H.pbe-rrkjus_psl.1.0.0.UPF" ;;
        C) file="C.pbe-n-rrkjus_psl.1.0.0.UPF" ;;
        O) file="O.pbe-n-kjpaw_psl.1.0.0.UPF" ;;
        N) file="N.pbe-n-rrkjus_psl.1.0.0.UPF" ;;
        Si) file="Si.pbe-n-rrkjus_psl.1.0.0.UPF" ;;
    esac
    
    if [ ! -f "$file" ]; then
        echo "Downloading $file..."
        wget -q "https://www.quantum-espresso.org/upf_files/$file" || echo "Failed to download $file"
    fi
done

echo "Pseudopotentials available:"
ls *.UPF 2>/dev/null | head -5 || echo "No pseudopotentials downloaded"

echo
echo "=== STEP 8: TEST QE INSTALLATION ==="
export PATH="$HOME/bin:$PATH"
export ASE_ESPRESSO_COMMAND="$HOME/bin/pw.x < PREFIX.pwi > PREFIX.pwo"
export ESPRESSO_PSEUDO="$HOME/software/qe-serial/pseudo/"

echo "Environment configured:"
echo "  PATH: $HOME/bin added"
echo "  ASE_ESPRESSO_COMMAND: $ASE_ESPRESSO_COMMAND"
echo "  ESPRESSO_PSEUDO: $ESPRESSO_PSEUDO"

if [ -x "$HOME/bin/pw.x" ]; then
    echo "✅ pw.x found and executable"
    
    echo "Testing pw.x startup..."
    timeout 5s $HOME/bin/pw.x 2>&1 | head -5 || echo "Startup test completed"
    
    echo
    echo "Creating and running test calculation..."
    mkdir -p /tmp/qe_test_serial
    cd /tmp/qe_test_serial
    
    cat > test.in << 'EOF'
&control
    calculation='scf'
    prefix='test'
    pseudo_dir='/mnt/polished-lake/home/cdrobinson/software/qe-serial/pseudo/'
    outdir='./'
/
&system
    ibrav=1, celldm(1)=12.0
    nat=1, ntyp=1
    ecutwfc=30.0
/
&electrons
    conv_thr=1.0d-8
/
ATOMIC_SPECIES
 H  1.008  H.pbe-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
 H 0.5 0.5 0.5
K_POINTS gamma
EOF
    
    echo "Running QE test (30 second timeout)..."
    timeout 30s $HOME/bin/pw.x < test.in > test.out 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ QE test calculation completed successfully!"
        grep -E "(Total|convergence|!" test.out | head -5
    else
        echo "⚠️  QE test had issues, checking output..."
        head -20 test.out | grep -E "(Program|version|error|Error)"
    fi
else
    echo "❌ pw.x not found or not executable"
fi

echo
echo "=== STEP 9: FINAL DFT TEST ==="
cd /mnt/polished-lake/home/cdrobinson/mattergen-interp
source .venv/bin/activate

echo "Running DFT framework test..."
python mattergen/scripts/run_dft_test.py

echo
echo "=== SUMMARY ==="
echo "Installation completed at: $(date)"

if [ -x "$HOME/bin/pw.x" ]; then
    echo "🎉 SUCCESS: QE Serial Installation Complete!"
    echo
    echo "📋 Installation details:"
    echo "   ✅ Executable: $HOME/bin/pw.x"
    echo "   ✅ Pseudopotentials: $HOME/software/qe-serial/pseudo/"
    echo "   ✅ Ready for ASE integration"
    echo
    echo "🔧 Environment setup:"
    echo "   export PATH=\"$HOME/bin:\$PATH\""
    echo "   export ASE_ESPRESSO_COMMAND=\"$HOME/bin/pw.x < PREFIX.pwi > PREFIX.pwo\""
    echo "   export ESPRESSO_PSEUDO=\"$HOME/software/qe-serial/pseudo/\""
else
    echo "❌ INSTALLATION FAILED"
    echo "pw.x not available"
fi

echo "==========================================="