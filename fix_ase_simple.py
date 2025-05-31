#!/usr/bin/env python3
"""
Simple ASE fix that works with older ASE versions.
Instead of using EspressoProfile, we'll set the environment correctly.
"""

import re
from glob import glob

def fix_ase_simple():
    """Fix ASE with simple environment configuration."""
    
    job_files = glob("jobs/dft_band_gaps/*.sh")
    
    for job_file in job_files:
        print(f"Applying simple ASE fix to {job_file}...")
        
        with open(job_file, 'r') as f:
            content = f.read()
        
        # Replace the complex ASE section with a simple one
        old_complex_section = re.compile(
            r"try:\s+from ase\.io import read.*?print\(f\"✅ ASE configured to use QE at.*?\"\)",
            re.DOTALL
        )
        
        simple_section = '''try:
    from ase.io import read
    from ase.calculators.espresso import Espresso
    import numpy as np
    import os
    
    print("✅ ASE and required modules loaded")
    
    # Set environment for QE
    os.environ['ASE_ESPRESSO_COMMAND'] = os.path.expanduser('~/bin/pw.x') + ' < PREFIX.pwi > PREFIX.pwo'
    print(f"✅ ASE configured to use QE at: {os.path.expanduser('~/bin/pw.x')}")'''
        
        content = old_complex_section.sub(simple_section, content)
        
        # Simplify the calculator creation
        old_calc = re.compile(
            r"calc = Espresso\(\s+profile=profile,\s+pseudopotentials=pseudopotentials,",
            re.DOTALL
        )
        
        new_calc = '''calc = Espresso(
                pw=os.path.expanduser('~/bin/pw.x'),
                pseudopotentials=pseudopotentials,
                pseudo_dir=os.path.expanduser('~/software/qe-serial/pseudo/'),'''
        
        content = old_calc.sub(new_calc, content)
        
        with open(job_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Applied simple fix to {job_file}")

def main():
    print("🔧 Applying simple ASE fix (compatible with older ASE versions)")
    print("=" * 60)
    
    fix_ase_simple()
    
    print(f"\n🎉 Applied simple ASE fix to all job scripts")
    print(f"🚀 Should work with any ASE version!")

if __name__ == "__main__":
    main()