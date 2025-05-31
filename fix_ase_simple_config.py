#!/usr/bin/env python3
"""
Simple ASE fix that creates ASE config and uses minimal Espresso setup.
"""

import re
from glob import glob

def fix_ase_simple_config():
    """Apply simple ASE fix with config file."""
    
    job_files = glob("jobs/dft_band_gaps/*.sh")
    
    for job_file in job_files:
        print(f"Applying simple ASE config fix to {job_file}...")
        
        with open(job_file, 'r') as f:
            content = f.read()
        
        # Find and replace the ASE import section
        content = re.sub(
            r'try:\s+from ase\.io import read\s+from ase\.calculators\.espresso import Espresso\s+import numpy as np\s+import os\s+\s+print\("✅ ASE and required modules loaded"\)\s+\s+# Set environment for QE\s+os\.environ\[\'ASE_ESPRESSO_COMMAND\'\] = os\.path\.expanduser\(\'~/bin/pw\.x\'\) \+ \' < PREFIX\.pwi > PREFIX\.pwo\'\s+print\(f"✅ ASE configured to use QE at: \{os\.path\.expanduser\(\'~/bin/pw\.x\'\)\}"\)',
            '''try:
    from ase.io import read
    from ase.calculators.espresso import Espresso
    import numpy as np
    import os
    
    print("✅ ASE and required modules loaded")
    
    # Create simple ASE config
    config_dir = os.path.expanduser('~/.config/ase')
    os.makedirs(config_dir, exist_ok=True)
    
    config_file = os.path.join(config_dir, 'config.ini')
    with open(config_file, 'w') as f:
        f.write(f"""[espresso]
command = {os.path.expanduser('~/bin/pw.x')} < PREFIX.pwi > PREFIX.pwo
pseudo_dir = {os.path.expanduser('~/software/qe-serial/pseudo/')}
""")
    
    print(f"✅ ASE configured with config file")''',
            content,
            flags=re.DOTALL
        )
        
        # Simplify calculator creation
        content = re.sub(
            r'calc = Espresso\(\s+pw=os\.path\.expanduser\(\'~/bin/pw\.x\'\),\s+pseudopotentials=pseudopotentials,\s+input_data=',
            '''calc = Espresso(
                pseudopotentials=pseudopotentials,
                input_data=''',
            content
        )
        
        # Remove the pseudo_dir parameter since it's in config
        content = re.sub(
            r',\s+pseudo_dir=os\.path\.expanduser\(\'~/software/qe-serial/pseudo/\'\)',
            '',
            content
        )
        
        with open(job_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Applied simple ASE config fix to {job_file}")

def main():
    print("🔧 Applying simple ASE configuration fix with config file")
    print("=" * 60)
    
    fix_ase_simple_config()
    
    print(f"\n🎉 Applied simple ASE config fix to all job scripts")
    print(f"🚀 Uses ~/.config/ase/config.ini for configuration!")

if __name__ == "__main__":
    main()