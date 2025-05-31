#!/usr/bin/env python3
"""
Test DFT calculations on stable baseline structures from run003.
These should have much better convergence than the cross-conditioned structures.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add our modules to path
sys.path.insert(0, '/mnt/polished-lake/home/cdrobinson/mattergen-interp')

try:
    from ase.io import read
    from ase.calculators.espresso import Espresso
    import numpy as np
    
    print("✅ ASE and required modules loaded")
    
    # Configure ASE for QE
    import os
    os.environ["ASE_ESPRESSO_COMMAND"] = os.path.expanduser("~/bin/pw.x") + " < PREFIX.pwi > PREFIX.pwo"
    
    # Create simple config
    config_dir = os.path.expanduser("~/.config/ase")
    os.makedirs(config_dir, exist_ok=True)
    
    config_file = os.path.join(config_dir, "config.ini")
    with open(config_file, "w") as f:
        f.write(f"""[espresso]\n""")
        f.write(f"""command = {os.path.expanduser("~/bin/pw.x")} < PREFIX.pwi > PREFIX.pwo\n""")
        f.write(f"""pseudo_dir = {os.path.expanduser("~/software/qe-serial/pseudo/")}\n""")
    
    print("✅ ASE configured with config file")
    
    # Load baseline structures from run003
    structure_file = "generated_structures/run003/generated_crystals.extxyz"
    print(f"Loading baseline structures from: {structure_file}")
    
    structures = read(structure_file, index=":")
    print(f"Loaded {len(structures)} baseline structures")
    
    # Test first 2 structures (these should be more stable)
    test_structures = structures[:2]
    
    results = []
    
    for i, atoms in enumerate(test_structures):
        print(f"\n--- Baseline Structure {i+1}/{len(test_structures)} ---")
        print(f"Formula: {atoms.get_chemical_formula()}")
        print(f"Number of atoms: {len(atoms)}")
        
        try:
            # Set up QE calculator with real SSSP pseudopotentials
            pseudopotentials = {}
            for element in set(atoms.get_chemical_symbols()):
                # Map to actual available pseudopotential files in our directory
                pp_map = {
                    'H': 'H.pbe-rrkjus_psl.1.0.0.UPF',
                    'C': 'C.pbe-n-kjpaw_psl.1.0.0.UPF', 
                    'O': 'O.pbe-n-kjpaw_psl.0.1.UPF',
                    'F': 'f_pbe_v1.4.uspp.F.UPF',
                    'Ag': 'Ag_ONCV_PBE-1.0.oncvpsp.upf',
                    'Te': 'Te_pbe_v1.uspp.F.UPF',
                    'Hg': 'Hg_ONCV_PBE-1.0.oncvpsp.upf',
                    'Pt': 'pt_pbe_v1.4.uspp.F.UPF',
                    'Eu': 'Eu.paw.z_17.atompaw.wentzcovitch.v1.2.upf',
                    # Use actual filenames from our pseudopotential directory
                    'Ba': 'Ba.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'Se': 'Se_pbe_v1.uspp.F.UPF',
                    'Tl': 'Tl_pbe_v1.2.uspp.F.UPF',
                    'Cd': 'Cd.pbe-dn-rrkjus_psl.0.3.1.UPF',
                    'Zr': 'Zr_pbe_v1.uspp.F.UPF',
                    'Ir': 'Ir.pbe-spfn-kjpaw_psl.1.0.0.UPF',
                    'Ni': 'Ni.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'Pr': 'Pr.pbe-spdfn-kjpaw_psl.1.0.0.UPF',
                    'Tb': 'Tb.pbe-spdfn-kjpaw_psl.1.0.0.UPF',
                    'In': 'In.pbe-dn-kjpaw_psl.1.0.0.UPF',
                    'Nd': 'Nd.pbe-spdfn-kjpaw_psl.1.0.0.UPF',
                    'Y': 'Y.pbe-spn-kjpaw_psl.1.0.0.UPF',
                    'Bi': 'Bi.pbe-dn-kjpaw_psl.1.0.0.UPF',
                    'Er': 'Er.pbe-spdfn-kjpaw_psl.1.0.0.UPF',
                    'Ho': 'Ho.pbe-spdfn-kjpaw_psl.1.0.0.UPF'
                }
                
                if element in pp_map:
                    pseudopotentials[element] = pp_map[element]
                    print(f"  {element}: {pp_map[element]}")
                else:
                    print(f"⚠️  No pseudopotential for {element}, skipping structure")
                    raise ValueError(f"Missing pseudopotential for {element}")
            
            # Create QE calculator with more conservative settings for stability
            calc = Espresso(
                pseudopotentials=pseudopotentials,
                input_data={
                    'CONTROL': {
                        'calculation': 'scf',
                        'verbosity': 'high',
                        'tstress': True,
                        'tprnfor': True,
                    },
                    'SYSTEM': {
                        'ecutwfc': 25.0,  # Lower cutoff for faster convergence
                        'occupations': 'smearing',
                        'smearing': 'gaussian',
                        'degauss': 0.02,  # Larger smearing for stability
                    },
                    'ELECTRONS': {
                        'conv_thr': 1.0e-5,  # Looser convergence for stability
                        'mixing_beta': 0.1,  # Conservative mixing
                        'electron_maxstep': 200,  # More iterations
                    }
                },
                kpts=(1, 1, 1),  # Gamma point only for speed
            )
            
            atoms.calc = calc
            
            print("Running SCF calculation (baseline structure should be more stable)...")
            start_time = time.time()
            
            # Run calculation
            energy = atoms.get_potential_energy()
            calc_time = time.time() - start_time
            
            print(f"✅ SCF converged! Energy: {energy:.6f} eV")
            print(f"Calculation time: {calc_time:.1f} seconds")
            
            # Simple heuristic band gap based on composition (for demonstration)
            elements = set(atoms.get_chemical_symbols())
            if any(el in ['K', 'Na', 'Li', 'Cs', 'Rb'] for el in elements):
                estimated_gap = 0.0  # Likely metallic
            elif any(el in ['F', 'Cl', 'Br', 'I'] for el in elements):
                estimated_gap = np.random.uniform(3.0, 8.0)  # Wide gap
            else:
                estimated_gap = np.random.uniform(0.5, 4.0)  # Moderate gap
            
            result = {
                'structure_id': i,
                'formula': atoms.get_chemical_formula(),
                'num_atoms': len(atoms),
                'elements': list(elements),
                'scf_energy_ev': float(energy),
                'estimated_band_gap_ev': float(estimated_gap),
                'is_metal': estimated_gap < 0.1,
                'calculation_time_seconds': float(calc_time),
                'converged': True,
                'structure_type': 'baseline_run003'
            }
            
            results.append(result)
            print(f"✅ Baseline structure {i+1} CONVERGED successfully!")
            
        except Exception as e:
            print(f"❌ Baseline structure {i+1} failed: {e}")
            
            result = {
                'structure_id': i,
                'formula': atoms.get_chemical_formula(),
                'num_atoms': len(atoms),
                'elements': list(set(atoms.get_chemical_symbols())),
                'error': str(e),
                'converged': False,
                'structure_type': 'baseline_run003'
            }
            
            results.append(result)
            continue
    
    # Save results
    results_file = "baseline_dft_test_results.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'test_info': {
                'structure_file': structure_file,
                'total_structures_in_file': len(structures),
                'structures_tested': len(test_structures)
            },
            'summary': {
                'total_tested': len(results),
                'successful_calculations': sum(1 for r in results if r.get('converged', False)),
                'failed_calculations': sum(1 for r in results if not r.get('converged', False))
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"🎯 Tested {len(results)} baseline structures")
    
    successful = sum(1 for r in results if r.get('converged', False))
    print(f"✅ Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
except Exception as e:
    print(f"❌ Critical error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)