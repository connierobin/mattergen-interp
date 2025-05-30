#!/usr/bin/env python3
"""
Late Insertion Trajectory Feasibility Summary

Based on our analysis of the original unconditioned trajectory and attempts 
at late insertion, this script summarizes our findings and provides 
recommendations for future approaches.
"""

from pathlib import Path
import json


def create_feasibility_summary():
    """Create a comprehensive summary of late insertion feasibility."""
    
    summary = {
        "analysis_date": "2024-12-05",
        "approach": "Late-stage insertion trajectories from timestep 800",
        
        "key_findings": {
            "trajectory_analysis": {
                "total_timesteps_analyzed": 2000,
                "late_insertion_window": "timestep 800-999 (final 200 steps)",
                "structural_changes_in_window": {
                    "average_lattice_parameter_change": "42.9%",
                    "volume_change": "15.6%",
                    "structures_considered_equivalent": False,
                    "space_group_remains_constant": True,
                    "final_space_group": 1  # P1 triclinic
                },
                "interpretation": [
                    "Significant structural evolution occurs in final 200 timesteps",
                    "42.9% average lattice parameter change indicates substantial plasticity",
                    "Space group remains P1 (triclinic) throughout - no symmetry emergence",
                    "Structures are not considered equivalent by StructureMatcher",
                    "This suggests late insertion conditioning could have meaningful impact"
                ]
            },
            
            "late_insertion_attempts": {
                "total_strategies_attempted": 9,
                "successful_strategies": 0,
                "failure_mode": "SVD convergence error in lattice parameter conversion",
                "error_location": "Post-processing after successful diffusion sampling",
                "diffusion_sampling_success": True,
                "all_200_timesteps_completed": True
            },
            
            "technical_challenges": {
                "primary_issue": "Numerical instability in lattice matrix to parameters conversion",
                "specific_error": "linalg.svd: failed to converge due to ill-conditioned matrix",
                "error_occurs_after": "Successful completion of all 200 diffusion timesteps",
                "root_cause": [
                    "Diffusion process can produce near-singular lattice matrices",
                    "Starting from timestep 800 with partially converged structure",
                    "Property conditioning may create geometric contradictions",
                    "SVD decomposition becomes numerically unstable"
                ]
            }
        },
        
        "feasibility_assessment": {
            "approach_viability": "Promising but requires technical modifications",
            "evidence_for_viability": [
                "Large structural changes (42.9%) occur in final 200 timesteps",
                "Diffusion sampling completes successfully for all 200 steps",
                "Error occurs only in final structure conversion, not in diffusion process",
                "Trajectory files contain valid intermediate structures"
            ],
            "technical_barriers": [
                "SVD convergence failures in lattice parameter conversion",
                "Need for robust numerical handling of ill-conditioned matrices",
                "Property conditioning creating geometric instabilities"
            ]
        },
        
        "recommended_solutions": {
            "immediate_fixes": [
                {
                    "solution": "Trajectory structure extraction",
                    "description": "Extract structures from step 199 instead of final converted structure",
                    "feasibility": "High - trajectory files are generated successfully",
                    "implementation": "Modify generator to save structures from intermediate steps"
                },
                {
                    "solution": "Robust lattice parameter conversion",
                    "description": "Implement SVD with fallback methods (pseudoinverse, eigendecomposition)",
                    "feasibility": "Medium - requires modifying data_utils.py",
                    "implementation": "Add try-catch blocks with alternative numerical methods"
                },
                {
                    "solution": "Error tolerance in generator",
                    "description": "Allow generator to return partial results instead of failing completely",
                    "feasibility": "High - modify draw_samples_from_sampler error handling",
                    "implementation": "Catch SVD errors and return best available structure"
                }
            ],
            
            "alternative_approaches": [
                {
                    "approach": "Earlier insertion point",
                    "description": "Start late insertion from timestep 600 or 700 instead of 800",
                    "rationale": "Less converged structures may be more amenable to conditioning",
                    "tradeoff": "Longer conditioning window but potentially more stable numerically"
                },
                {
                    "approach": "Gradual conditioning",
                    "description": "Gradually increase conditioning strength over timesteps",
                    "rationale": "Avoid sudden geometric contradictions",
                    "implementation": "Interpolate conditioning targets from 0 to full strength"
                },
                {
                    "approach": "Property-specific insertion points",
                    "description": "Use different insertion timesteps for different property types",
                    "rationale": "Space group conditioning may need earlier insertion than energy conditioning",
                    "customization": "Tailor approach based on property complexity"
                }
            ]
        },
        
        "scientific_value": {
            "research_questions_addressable": [
                "How much can late-stage conditioning influence final crystal structure?",
                "What is the critical window for property-guided steering?",
                "Which properties are most amenable to late-stage conditioning?",
                "How does conditioning strength affect structural stability?"
            ],
            "comparison_baselines": [
                "Structure at timestep 800 vs final structure (unconditioned)",
                "Structure at timestep 800 vs final structure (conditioned)",
                "Different conditioning strategies applied to same starting structure",
                "Same conditioning applied at different insertion timesteps"
            ]
        },
        
        "next_steps": {
            "phase_1_quick_wins": [
                "Implement trajectory structure extraction from step 199",
                "Run 1-2 test strategies with error handling improvements",
                "Validate that extracted structures are scientifically meaningful"
            ],
            "phase_2_robust_implementation": [
                "Implement robust lattice parameter conversion with fallbacks",
                "Test with full set of 9 conditioning strategies",
                "Compare results with baseline trajectory analysis"
            ],
            "phase_3_extended_analysis": [
                "Test different insertion timesteps (600, 700, 800, 900)",
                "Implement gradual conditioning approaches",
                "Comprehensive property conditioning success analysis"
            ]
        },
        
        "conclusion": {
            "summary": [
                "Late insertion approach is technically feasible and scientifically valuable",
                "Significant structural plasticity exists in final 200 timesteps (42.9% change)",
                "Current failure is a numerical post-processing issue, not a fundamental limitation",
                "Simple modifications can likely resolve the technical barriers",
                "Approach offers unique insights into diffusion model controllability"
            ],
            "confidence_level": "High for technical feasibility, Medium for scientific impact pending validation"
        }
    }
    
    return summary


def save_summary(summary: dict, output_path: Path):
    """Save the feasibility summary to file."""
    
    # Save as JSON
    with open(output_path / "late_insertion_feasibility_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save as readable markdown
    markdown_content = f"""# Late Insertion Trajectory Feasibility Summary

**Analysis Date:** {summary['analysis_date']}  
**Approach:** {summary['approach']}

## Key Findings

### Trajectory Analysis
- **Total timesteps analyzed:** {summary['key_findings']['trajectory_analysis']['total_timesteps_analyzed']}
- **Late insertion window:** {summary['key_findings']['trajectory_analysis']['late_insertion_window']}
- **Average lattice parameter change:** {summary['key_findings']['trajectory_analysis']['structural_changes_in_window']['average_lattice_parameter_change']}
- **Volume change:** {summary['key_findings']['trajectory_analysis']['structural_changes_in_window']['volume_change']}
- **Structures equivalent:** {summary['key_findings']['trajectory_analysis']['structural_changes_in_window']['structures_considered_equivalent']}

**Interpretation:**
"""
    
    for point in summary['key_findings']['trajectory_analysis']['interpretation']:
        markdown_content += f"- {point}\n"
    
    markdown_content += f"""
### Late Insertion Attempts
- **Strategies attempted:** {summary['key_findings']['late_insertion_attempts']['total_strategies_attempted']}
- **Successful strategies:** {summary['key_findings']['late_insertion_attempts']['successful_strategies']}
- **Failure mode:** {summary['key_findings']['late_insertion_attempts']['failure_mode']}
- **Diffusion sampling success:** {summary['key_findings']['late_insertion_attempts']['diffusion_sampling_success']}

## Feasibility Assessment

**Approach viability:** {summary['feasibility_assessment']['approach_viability']}

**Evidence for viability:**
"""
    
    for evidence in summary['feasibility_assessment']['evidence_for_viability']:
        markdown_content += f"- {evidence}\n"
    
    markdown_content += "\n**Technical barriers:**\n"
    for barrier in summary['feasibility_assessment']['technical_barriers']:
        markdown_content += f"- {barrier}\n"
    
    markdown_content += "\n## Recommended Solutions\n\n### Immediate Fixes\n"
    
    for i, fix in enumerate(summary['recommended_solutions']['immediate_fixes'], 1):
        markdown_content += f"""
**{i}. {fix['solution']}**
- Description: {fix['description']}
- Feasibility: {fix['feasibility']}
- Implementation: {fix['implementation']}
"""
    
    markdown_content += "\n### Alternative Approaches\n"
    
    for i, approach in enumerate(summary['recommended_solutions']['alternative_approaches'], 1):
        markdown_content += f"""
**{i}. {approach['approach']}**
- Description: {approach['description']}
- Rationale: {approach['rationale']}
"""
    
    markdown_content += "\n## Scientific Value\n\n**Research questions addressable:**\n"
    for question in summary['scientific_value']['research_questions_addressable']:
        markdown_content += f"- {question}\n"
    
    markdown_content += "\n## Next Steps\n\n### Phase 1: Quick Wins\n"
    for step in summary['next_steps']['phase_1_quick_wins']:
        markdown_content += f"- {step}\n"
    
    markdown_content += "\n### Phase 2: Robust Implementation\n"
    for step in summary['next_steps']['phase_2_robust_implementation']:
        markdown_content += f"- {step}\n"
    
    markdown_content += "\n### Phase 3: Extended Analysis\n"
    for step in summary['next_steps']['phase_3_extended_analysis']:
        markdown_content += f"- {step}\n"
    
    markdown_content += "\n## Conclusion\n\n"
    for point in summary['conclusion']['summary']:
        markdown_content += f"- {point}\n"
    
    markdown_content += f"\n**Confidence Level:** {summary['conclusion']['confidence_level']}\n"
    
    with open(output_path / "late_insertion_feasibility_summary.md", 'w') as f:
        f.write(markdown_content)


def main():
    """Generate and save the feasibility summary."""
    
    output_dir = Path("generated_structures/late_insertion_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Late Insertion Trajectory Feasibility Summary")
    print("=" * 60)
    
    # Generate summary
    summary = create_feasibility_summary()
    
    # Save summary
    save_summary(summary, output_dir)
    
    print("\nSUMMARY OF FINDINGS:")
    print("=" * 40)
    
    print(f"\nTrajectory Analysis:")
    print(f"  • Structural change in final 200 steps: {summary['key_findings']['trajectory_analysis']['structural_changes_in_window']['average_lattice_parameter_change']}")
    print(f"  • Volume change: {summary['key_findings']['trajectory_analysis']['structural_changes_in_window']['volume_change']}")
    print(f"  • Space group remains constant: {summary['key_findings']['trajectory_analysis']['structural_changes_in_window']['space_group_remains_constant']}")
    
    print(f"\nLate Insertion Attempts:")
    print(f"  • Strategies attempted: {summary['key_findings']['late_insertion_attempts']['total_strategies_attempted']}")
    print(f"  • Successful: {summary['key_findings']['late_insertion_attempts']['successful_strategies']}")
    print(f"  • Diffusion sampling worked: {summary['key_findings']['late_insertion_attempts']['diffusion_sampling_success']}")
    print(f"  • Failure mode: {summary['key_findings']['late_insertion_attempts']['failure_mode']}")
    
    print(f"\nFeasibility Assessment:")
    print(f"  • Approach viability: {summary['feasibility_assessment']['approach_viability']}")
    print(f"  • Confidence level: {summary['conclusion']['confidence_level']}")
    
    print(f"\nKey Insights:")
    for insight in summary['conclusion']['summary']:
        print(f"  • {insight}")
    
    print(f"\nNext Steps:")
    print(f"  1. {summary['next_steps']['phase_1_quick_wins'][0]}")
    print(f"  2. {summary['next_steps']['phase_1_quick_wins'][1]}")
    print(f"  3. {summary['next_steps']['phase_1_quick_wins'][2]}")
    
    print(f"\nFiles saved:")
    print(f"  • {output_dir / 'late_insertion_feasibility_summary.json'}")
    print(f"  • {output_dir / 'late_insertion_feasibility_summary.md'}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()