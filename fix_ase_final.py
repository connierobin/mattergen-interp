#!/usr/bin/env python3
"""
Final ASE fix that should work across all ASE versions.
Create an ASE configuration file and use simple Espresso initialization.
"""

import re
from glob import glob

def fix_ase_final():
    """Apply the final ASE fix with configuration file."""
    
    job_files = glob("jobs/dft_band_gaps/*.sh")
    
    for job_file in job_files:
        print(f"Applying final ASE fix to {job_file}...")
        
        with open(job_file, 'r') as f:
            content = f.read()
        
        # Replace the complex ASE section with a robust one using config
        old_ase_section = re.compile(
            r"try:\\s+from ase\\.io import read.*?print\\(f\\\"✅ ASE configured to use QE at.*?\\\"\\)",
            re.DOTALL
        )
        
        new_ase_section = '''try:
    from ase.io import read
    from ase.calculators.espresso import Espresso
    import numpy as np
    import os
    import tempfile
    
    print("✅ ASE and required modules loaded")
    
    # Create a simple ASE config for QE
    config_dir = os.path.expanduser('~/.config/ase')
    os.makedirs(config_dir, exist_ok=True)
    
    config_file = os.path.join(config_dir, 'config.ini')
    with open(config_file, 'w') as f:
        f.write(f"""[espresso]
command = {os.path.expanduser('~/bin/pw.x')} < PREFIX.pwi > PREFIX.pwo
pseudo_dir = {os.path.expanduser('~/software/qe-serial/pseudo/')}
""")
    
    print(f"✅ ASE configured to use QE at: {os.path.expanduser('~/bin/pw.x')}")'''
        
        content = old_ase_section.sub(new_ase_section, content)
        
        # Simplify the calculator creation to use the config
        old_calc = re.compile(
            r"calc = Espresso\\(\\s+pw=os\\.path\\.expanduser\\('~/bin/pw\\.x'\\),\\s+pseudopotentials=pseudopotentials,",
            re.DOTALL
        )
        
        new_calc = '''calc = Espresso(
                pseudopotentials=pseudopotentials,'''
        
        content = old_calc.sub(new_calc, content)
        
        # Remove the redundant pseudo_dir parameter
        content = re.sub(r",\\s+pseudo_dir=os\\.path\\.expanduser\\('~/software/qe-serial/pseudo/'\\)", "", content)
        
        with open(job_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Applied final ASE fix to {job_file}")

def main():
    print("🔧 Applying final ASE configuration fix")
    print("=" * 50)
    
    fix_ase_final()
    
    print(f"\n🎉 Applied final ASE fix to all job scripts")
    print(f"🚀 Uses ASE config file - should work with any version!")

if __name__ == "__main__":
    main()