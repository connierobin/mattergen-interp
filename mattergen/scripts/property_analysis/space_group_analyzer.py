from typing import Any, Dict

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .base import BasePropertyAnalyzer


class SpaceGroupAnalyzer(BasePropertyAnalyzer):
    """Analyzer for space group properties of crystal structures."""
    
    def __init__(self, symprec: float = 0.01, angle_tolerance: float = 5.0):
        super().__init__("space_group")
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
    
    def analyze_structure(self, structure: Structure, structure_id: str = None) -> Dict[str, Any]:
        """Analyze space group properties of a structure."""
        try:
            analyzer = SpacegroupAnalyzer(
                structure, 
                symprec=self.symprec, 
                angle_tolerance=self.angle_tolerance
            )
            
            result = {
                'space_group_number': analyzer.get_space_group_number(),
                'space_group_symbol': analyzer.get_space_group_symbol(),
                'crystal_system': analyzer.get_crystal_system(),
                'lattice_type': analyzer.get_lattice_type(),
                'point_group': analyzer.get_point_group_symbol(),
                'hall_number': analyzer.get_hall_number(),
                'international_symbol': analyzer.get_space_group_symbol(),
                'has_inversion': analyzer.is_laue(),
            }
            
        except Exception as e:
            result = {
                'space_group_number': None,
                'space_group_symbol': 'P1',  # Default to triclinic
                'crystal_system': 'triclinic',
                'lattice_type': 'triclinic',
                'point_group': '1',
                'hall_number': None,
                'international_symbol': 'P1',
                'has_inversion': False,
                'analysis_error': str(e)
            }
        
        return result