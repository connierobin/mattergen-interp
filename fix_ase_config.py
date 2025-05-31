#!/usr/bin/env python3
"""
Fix ASE configuration issue in DFT job scripts.

The issue is that ASE can't find the QE configuration on compute nodes.
We need to properly set up ASE to use our home-installed QE.
"""

import re
from glob import glob

def fix_ase_configuration():
    """Fix ASE configuration in all job scripts."""
    
    job_files = glob("jobs/dft_band_gaps/*.sh")
    
    for job_file in job_files:
        print(f"Fixing ASE config in {job_file}...")
        
        with open(job_file, 'r') as f:
            content = f.read()
        
        # Find the Python section where we import ASE
        old_ase_section = r"try:\s+from ase\.io import read\s+from ase\.calculators\.espresso import Espresso\s+import numpy as np\s+print\(\"✅ ASE and required modules loaded\"\)"
        
        new_ase_section = '''try:
    from ase.io import read
    from ase.calculators.espresso import Espresso
    from ase.calculators.espresso.espresso import EspressoProfile
    import numpy as np
    import os
    
    print("✅ ASE and required modules loaded")
    
    # Configure ASE to use our home-installed QE
    os.environ['ASE_ESPRESSO_COMMAND'] = os.path.expanduser('~/bin/pw.x') + ' < PREFIX.pwi > PREFIX.pwo'
    
    # Create Espresso profile for our installation
    profile = EspressoProfile(
        argv=[os.path.expanduser('~/bin/pw.x')],
        pseudo_dir=os.path.expanduser('~/software/qe-serial/pseudo/')
    )
    
    print(f"✅ ASE configured to use QE at: {os.path.expanduser('~/bin/pw.x')}")'''
        
        content = re.sub(old_ase_section, new_ase_section, content, flags=re.DOTALL)
        
        # Update the Espresso calculator creation to use the profile
        old_calc_creation = r"calc = Espresso\(\s+pw=os\.path\.expanduser\('~/bin/pw\.x'\),\s+pseudopotentials=pseudopotentials,"
        
        new_calc_creation = '''calc = Espresso(
                profile=profile,
                pseudopotentials=pseudopotentials,'''
        
        content = re.sub(old_calc_creation, new_calc_creation, content, flags=re.DOTALL | re.MULTILINE)
        
        # Also remove the redundant pseudo_dir parameter since it's in the profile
        content = re.sub(r",\s+pseudo_dir=os\.path\.expanduser\('~/software/qe-serial/pseudo/'\)", "", content)
        
        with open(job_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed ASE config in {job_file}")

def main():
    print("🔧 Fixing ASE configuration in DFT job scripts")
    print("=" * 50)
    
    fix_ase_configuration()
    
    print(f"\n🎉 Fixed ASE configuration in all job scripts")
    print(f"🚀 Ready to test and submit jobs!")

if __name__ == "__main__":
    main()