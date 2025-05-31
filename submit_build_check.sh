#!/bin/bash
#SBATCH --job-name=build_check
#SBATCH --output=jobs/build_check/job_%j.out
#SBATCH --error=jobs/build_check/job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=4G

# Create job output directory
mkdir -p jobs/build_check

echo "=== BUILD TOOLS AVAILABILITY CHECK ==="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo

echo "=== COMPILERS ==="
which gcc && echo "✅ gcc: $(gcc --version | head -1)" || echo "❌ gcc not found"
which g++ && echo "✅ g++: $(g++ --version | head -1)" || echo "❌ g++ not found" 
which gfortran && echo "✅ gfortran: $(gfortran --version | head -1)" || echo "❌ gfortran not found"
which ifort && echo "✅ ifort: $(ifort --version 2>/dev/null | head -1)" || echo "❌ ifort not found"
which icc && echo "✅ icc: $(icc --version 2>/dev/null | head -1)" || echo "❌ icc not found"

echo
echo "=== BUILD TOOLS ==="
which make && echo "✅ make: $(make --version | head -1)" || echo "❌ make not found"
which cmake && echo "✅ cmake: $(cmake --version | head -1)" || echo "❌ cmake not found"
which autoconf && echo "✅ autoconf: $(autoconf --version | head -1)" || echo "❌ autoconf not found"

echo
echo "=== MPI ==="
which mpirun && echo "✅ mpirun: $(mpirun --version 2>&1 | head -1)" || echo "❌ mpirun not found"
which mpicc && echo "✅ mpicc: $(mpicc --version 2>/dev/null | head -1)" || echo "❌ mpicc not found"
which mpif90 && echo "✅ mpif90: $(mpif90 --version 2>/dev/null | head -1)" || echo "❌ mpif90 not found"

echo
echo "=== SCIENTIFIC LIBRARIES ==="
# Check for LAPACK/BLAS
find /usr/lib* -name "*lapack*" 2>/dev/null | head -3 && echo "✅ LAPACK libraries found" || echo "❌ LAPACK not found"
find /usr/lib* -name "*blas*" 2>/dev/null | head -3 && echo "✅ BLAS libraries found" || echo "❌ BLAS not found"
find /usr/lib* -name "*fftw*" 2>/dev/null | head -3 && echo "✅ FFTW libraries found" || echo "❌ FFTW not found"

echo
echo "=== PACKAGE MANAGERS ==="
which apt && echo "✅ apt available" || echo "❌ apt not found"
which yum && echo "✅ yum available" || echo "❌ yum not found"
which conda && echo "✅ conda available" || echo "❌ conda not found"
which pip && echo "✅ pip available" || echo "❌ pip not found"

echo
echo "=== DEVELOPMENT PACKAGES CHECK ==="
# Check if packages are installed
dpkg -l | grep -E "(build-essential|gfortran|libopenmpi|libfftw|liblapack|libblas)" 2>/dev/null || echo "No relevant packages found via dpkg"

echo
echo "=== ENVIRONMENT VARIABLES ==="
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CC: $CC"
echo "FC: $FC"
echo "CXX: $CXX"

echo
echo "=== /OPT DIRECTORY ==="
ls -la /opt/ 2>/dev/null | head -10

echo
echo "=== MODULE SYSTEM ==="
which module && module list 2>&1 || echo "No module system"

echo
echo "Build check completed at: $(date)"