from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pymatgen.core.structure import Structure


class BasePropertyAnalyzer(ABC):
    """Base class for property analyzers."""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def analyze_structure(self, structure: Structure, structure_id: str = None) -> Dict[str, Any]:
        """Analyze a single structure and return property values."""
        pass
    
    def analyze_structures(self, structures: List[Structure], structure_ids: List[str] = None) -> pd.DataFrame:
        """Analyze multiple structures and return results as DataFrame."""
        if structure_ids is None:
            structure_ids = [f"structure_{i}" for i in range(len(structures))]
        
        results = []
        for structure, structure_id in zip(structures, structure_ids):
            try:
                result = self.analyze_structure(structure, structure_id)
                result['structure_id'] = structure_id
                result['analyzer'] = self.name
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {structure_id}: {e}")
                error_result = {
                    'structure_id': structure_id,
                    'analyzer': self.name,
                    'error': str(e)
                }
                results.append(error_result)
        
        return pd.DataFrame(results)
    
    def save_results(self, results: pd.DataFrame, output_path: Path) -> None:
        """Save analysis results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            results.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            results.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")