from typing import Any, Dict, Optional

from pymatgen.core.structure import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.utils import expand_into_subsystems
from .base import BasePropertyAnalyzer


class EnergyAboveHullAnalyzer(BasePropertyAnalyzer):
    """Analyzer for energy above hull calculations."""
    
    def __init__(self, reference_dataset: ReferenceDataset, default_energy_per_atom: float = 0.0):
        super().__init__("energy_above_hull")
        self.reference_dataset = reference_dataset
        self.default_energy_per_atom = default_energy_per_atom
    
    def analyze_structure(self, structure: Structure, structure_id: str = None, 
                         energy_per_atom: Optional[float] = None) -> Dict[str, Any]:
        """Analyze energy above hull for a structure."""
        if energy_per_atom is None:
            energy_per_atom = self.default_energy_per_atom
        
        try:
            # Create ComputedEntry for the structure
            total_energy = energy_per_atom * len(structure)
            entry = ComputedEntry(structure.composition, total_energy)
            
            # Get chemical system and expand to subsystems
            chemical_system = structure.composition.chemical_system
            subsystems = expand_into_subsystems(chemical_system)
            
            # Collect reference entries for all subsystems
            reference_entries = []
            for subsys in subsystems:
                key = "-".join(sorted(subsys))
                if key in self.reference_dataset.entries_by_chemsys:
                    for ref_entry in self.reference_dataset.entries_by_chemsys[key]:
                        if not hasattr(ref_entry, 'energy') or ref_entry.energy is None:
                            continue
                        reference_entries.append(ref_entry)
            
            if not reference_entries:
                return {
                    'energy_above_hull': None,
                    'chemical_system': chemical_system,
                    'energy_per_atom': energy_per_atom,
                    'total_energy': total_energy,
                    'analysis_error': f"No reference data available for chemical system: {chemical_system}"
                }
            
            # Create phase diagram and compute energy above hull
            phase_diagram = PhaseDiagram(reference_entries)
            energy_above_hull = phase_diagram.get_e_above_hull(entry, allow_negative=True)
            
            # Get stability information
            is_stable = energy_above_hull <= 0.025  # 25 meV/atom threshold
            
            result = {
                'energy_above_hull': energy_above_hull,
                'energy_per_atom': energy_per_atom,
                'total_energy': total_energy,
                'chemical_system': chemical_system,
                'is_stable': is_stable,
                'num_reference_entries': len(reference_entries),
            }
            
        except Exception as e:
            result = {
                'energy_above_hull': None,
                'energy_per_atom': energy_per_atom,
                'total_energy': energy_per_atom * len(structure) if energy_per_atom else None,
                'chemical_system': structure.composition.chemical_system,
                'is_stable': False,
                'num_reference_entries': 0,
                'analysis_error': str(e)
            }
        
        return result