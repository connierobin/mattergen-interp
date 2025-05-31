#!/usr/bin/env python3
"""
Fix ASE to use EspressoProfile as suggested by the error message.
"""

from glob import glob

def espresso_profile_fix():
    """Use EspressoProfile as required by modern ASE."""
    
    job_files = glob("jobs/dft_band_gaps/*.sh")
    
    for job_file in job_files:
        print(f"Updating to EspressoProfile in {job_file}...")
        
        with open(job_file, 'r') as f:
            content = f.read()
        
        # Update the import section
        content = content.replace(
            "    from ase.io import read\n    from ase.calculators.espresso import Espresso\n    import numpy as np",
            "    from ase.io import read\n    from ase.calculators.espresso import Espresso\n    from ase.calculators.espresso.espresso import EspressoProfile\n    import numpy as np"
        )
        
        # Replace the calculator creation with EspressoProfile
        content = content.replace(
            """            # Create QE calculator
            # Create QE calculator
            calc = Espresso(
                command=os.path.expanduser('~/bin/pw.x') + ' < PREFIX.pwi > PREFIX.pwo',
                pseudopotentials=pseudopotentials,
                pseudo_dir=os.path.expanduser('~/software/qe-serial/pseudo/'),""",
            """            # Create QE calculator with EspressoProfile
            profile = EspressoProfile(
                argv=[os.path.expanduser('~/bin/pw.x')],
                pseudo_dir=os.path.expanduser('~/software/qe-serial/pseudo/')
            )
            
            calc = Espresso(
                profile=profile,
                pseudopotentials=pseudopotentials,"""
        )
        
        with open(job_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated to EspressoProfile in {job_file}")

def main():
    print("🔧 Updating ASE to use EspressoProfile (modern ASE)")
    print("=" * 50)
    
    espresso_profile_fix()
    
    print(f"\n🎉 Updated all job scripts to use EspressoProfile")
    print(f"🚀 Should work with modern ASE versions!")

if __name__ == "__main__":
    main()