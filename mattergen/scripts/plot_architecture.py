#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np

def create_mattergen_architecture_diagram(save_path="mattergen_architecture.png", figsize=(20, 14)):
    """
    Create a comprehensive architecture diagram for MatterGen showing the generation process.
    
    The diagram shows:
    1. Overall generation loop (diffusion sampling)
    2. Score model detail (GemNet processing) 
    3. Multi-modal handling (positions, lattice, atom types)
    4. Property conditioning flow
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'diffusion': '#E8F4FD',
        'gemnet': '#FFE6CC', 
        'conditioning': '#E8F8E8',
        'data': '#F0F0F0',
        'noise': '#FFE6E6',
        'output': '#E6F3FF'
    }
    
    # Title
    ax.text(5, 9.7, 'MatterGen Architecture: Crystal Structure Generation', 
            fontsize=18, fontweight='bold', ha='center')
    
    # =================== MAIN GENERATION LOOP ===================
    # Diffusion sampling loop
    loop_box = FancyBboxPatch((0.2, 7.8), 9.6, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['diffusion'], 
                              edgecolor='blue', linewidth=2)
    ax.add_patch(loop_box)
    ax.text(5, 9, 'Diffusion Sampling Loop (t = T → ε)', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Timeline
    t_positions = np.linspace(1, 9, 5)
    for i, t_pos in enumerate(t_positions):
        t_val = f"t={1.0 - i*0.25:.2f}" if i < 4 else "t=ε"
        ax.text(t_pos, 8.5, t_val, fontsize=10, ha='center')
        if i < len(t_positions) - 1:
            ax.arrow(t_pos + 0.3, 8.5, 1.4, 0, head_width=0.05, 
                    head_length=0.1, fc='blue', ec='blue')
    
    ax.text(5, 8.1, 'Iterative Denoising: Noise → Crystal Structure', 
            fontsize=11, ha='center', style='italic')
    
    # =================== SCORE MODEL DETAIL ===================
    # Main score model box
    score_box = FancyBboxPatch((0.5, 4.5), 9, 2.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['gemnet'], 
                               edgecolor='orange', linewidth=2)
    ax.add_patch(score_box)
    ax.text(5, 7, 'Score Model: GemNet-T + Property Conditioning', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Input processing
    ax.text(1.5, 6.5, 'Inputs:', fontsize=12, fontweight='bold')
    inputs = [
        'Noisy Structure (x_t)',
        'Timestep (t)', 
        'Properties (optional)'
    ]
    for i, inp in enumerate(inputs):
        ax.text(1.5, 6.2 - i*0.2, f'• {inp}', fontsize=10)
    
    # GemNet processing steps
    gemnet_steps = [
        ('Graph Construction', 'Atoms→Nodes, Bonds→Edges'),
        ('Atom Embedding', 'Atomic numbers→Vectors'),
        ('Edge Embedding', 'Distances→Radial basis'),
        ('Message Passing', '4 Interaction blocks'),
        ('Output Heads', 'Predict noise components')
    ]
    
    step_y = 5.7
    for i, (step, desc) in enumerate(gemnet_steps):
        # Step box
        step_box = FancyBboxPatch((3.5 + i*1.1, step_y), 1, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor='white', edgecolor='orange')
        ax.add_patch(step_box)
        ax.text(4 + i*1.1, step_y + 0.4, step, fontsize=9, ha='center', fontweight='bold')
        ax.text(4 + i*1.1, step_y + 0.15, desc, fontsize=8, ha='center')
        
        # Arrow to next step
        if i < len(gemnet_steps) - 1:
            ax.arrow(4.5 + i*1.1, step_y + 0.3, 0.5, 0, 
                    head_width=0.05, head_length=0.05, fc='orange', ec='orange')
    
    # Property conditioning branch
    prop_box = FancyBboxPatch((0.8, 5.0), 2.2, 1.8,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['conditioning'], 
                              edgecolor='green')
    ax.add_patch(prop_box)
    ax.text(1.9, 6.5, 'Property', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.9, 6.3, 'Conditioning', fontsize=11, fontweight='bold', ha='center')
    
    properties = [
        'Chemical System',
        'Space Group', 
        'Energy above Hull',
        'Band Gap'
    ]
    for i, prop in enumerate(properties):
        ax.text(1.9, 6.0 - i*0.2, f'• {prop}', fontsize=9, ha='center')
    
    # Conditioning arrow
    ax.arrow(3.0, 5.9, 0.4, 0, head_width=0.05, head_length=0.05, 
            fc='green', ec='green', linestyle='--')
    
    # =================== MULTI-MODAL OUTPUTS ===================
    # Output section
    output_y = 3.5
    ax.text(5, 4, 'Multi-Modal Noise Predictions', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Three output modalities
    modalities = [
        ('Positions', 'Fractional coordinates\n(Continuous SDE)', colors['output']),
        ('Lattice', 'Cell parameters\na, b, c, α, β, γ\n(Continuous SDE)', colors['output']),
        ('Atom Types', 'Chemical species\n(Discrete D3PM)', colors['output'])
    ]
    
    for i, (mod, desc, color) in enumerate(modalities):
        mod_box = FancyBboxPatch((1.5 + i*2.5, output_y), 2.2, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='navy')
        ax.add_patch(mod_box)
        ax.text(2.6 + i*2.5, output_y + 0.9, mod, fontsize=12, 
                fontweight='bold', ha='center')
        ax.text(2.6 + i*2.5, output_y + 0.4, desc, fontsize=10, 
                ha='center', va='center')
    
    # Arrows from GemNet to outputs
    for i in range(3):
        start_x = 4 + i*1.1 if i < 3 else 7.7
        end_x = 2.6 + i*2.5
        ax.arrow(start_x, 5.1, end_x - start_x, output_y + 1.2 - 5.1,
                head_width=0.05, head_length=0.05, fc='navy', ec='navy')
    
    # =================== PREDICTOR-CORRECTOR ===================
    # PC sampling box
    pc_box = FancyBboxPatch((0.5, 1.5), 9, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['noise'], edgecolor='red')
    ax.add_patch(pc_box)
    ax.text(5, 2.7, 'Predictor-Corrector Sampling', 
            fontsize=14, fontweight='bold', ha='center')
    
    # PC steps
    pc_steps = [
        ('Predictor', 'Euler step\nusing score'),
        ('Corrector', 'Langevin MCMC\nrefinement'),
        ('Update', 'x_{t-dt} ← x_t - noise')
    ]
    
    for i, (step, desc) in enumerate(pc_steps):
        step_x = 2 + i*2.5
        step_box = FancyBboxPatch((step_x, 1.8), 1.8, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor='white', edgecolor='red')
        ax.add_patch(step_box)
        ax.text(step_x + 0.9, 2.4, step, fontsize=11, 
                fontweight='bold', ha='center')
        ax.text(step_x + 0.9, 2.0, desc, fontsize=9, ha='center')
        
        if i < len(pc_steps) - 1:
            ax.arrow(step_x + 1.8, 2.2, 0.6, 0,
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
    
    # =================== FINAL OUTPUT ===================
    # Final structure
    final_box = FancyBboxPatch((3.5, 0.2), 3, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['data'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(final_box)
    ax.text(5, 0.6, 'Final Crystal Structure', 
            fontsize=13, fontweight='bold', ha='center')
    ax.text(5, 0.35, 'Lattice + Fractional Coords + Atom Types', 
            fontsize=10, ha='center')
    
    # Arrow from PC to final
    ax.arrow(5, 1.5, 0, -0.4, head_width=0.1, head_length=0.05, 
            fc='black', ec='black', linewidth=2)
    
    # =================== ANNOTATIONS ===================
    # Add some key annotations
    ax.text(0.2, 3.2, 'Key Features:', fontsize=12, fontweight='bold')
    features = [
        '• Handles continuous & discrete variables',
        '• Property-conditioned generation', 
        '• Periodic boundary conditions',
        '• Classifier-free guidance',
        '• Predictor-corrector refinement'
    ]
    for i, feature in enumerate(features):
        ax.text(0.2, 2.9 - i*0.2, feature, fontsize=10)
    
    # Data flow legend
    ax.text(8.5, 3.2, 'Data Flow:', fontsize=12, fontweight='bold')
    ax.plot([8.5, 9.2], [2.9, 2.9], 'b-', linewidth=2)
    ax.text(9.3, 2.9, 'Forward pass', fontsize=10, va='center')
    ax.plot([8.5, 9.2], [2.7, 2.7], 'g--', linewidth=2)
    ax.text(9.3, 2.7, 'Conditioning', fontsize=10, va='center')
    ax.plot([8.5, 9.2], [2.5, 2.5], 'r-', linewidth=2)
    ax.text(9.3, 2.5, 'Sampling', fontsize=10, va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Architecture diagram saved to: {save_path}")

def create_detailed_gemnet_diagram(save_path="gemnet_detail.png", figsize=(16, 10)):
    """
    Create a detailed diagram of the GemNet architecture within MatterGen.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'GemNet-T Architecture Detail in MatterGen', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input crystal structure
    crystal_box = FancyBboxPatch((0.5, 6), 2, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#F0F0F0', edgecolor='black')
    ax.add_patch(crystal_box)
    ax.text(1.5, 6.5, 'Crystal Structure', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 6.2, 'Lattice + Positions\n+ Atom Types', fontsize=10, ha='center')
    
    # Graph construction
    graph_box = FancyBboxPatch((3.5, 6), 2, 1,
                               boxstyle="round,pad=0.1", 
                               facecolor='#E8F4FD', edgecolor='blue')
    ax.add_patch(graph_box)
    ax.text(4.5, 6.5, 'Graph Construction', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.5, 6.2, 'Periodic boundaries\nCutoff radius', fontsize=10, ha='center')
    
    # Arrow
    ax.arrow(2.5, 6.5, 1, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # GemNet layers
    layers = [
        'Atom Embedding',
        'Edge Embedding', 
        'Interaction Block 1',
        'Interaction Block 2',
        'Interaction Block 3', 
        'Interaction Block 4',
        'Output Heads'
    ]
    
    layer_y = 5
    for i, layer in enumerate(layers):
        color = '#FFE6CC' if 'Interaction' in layer else '#E8F8E8'
        layer_box = FancyBboxPatch((7, layer_y - i*0.6), 2.5, 0.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='orange')
        ax.add_patch(layer_box)
        ax.text(8.25, layer_y - i*0.6 + 0.25, layer, fontsize=10, 
                fontweight='bold', ha='center')
        
        if i < len(layers) - 1:
            ax.arrow(8.25, layer_y - i*0.6, 0, -0.5, 
                    head_width=0.05, head_length=0.02, fc='orange', ec='orange')
    
    # Arrow from graph to layers
    ax.arrow(5.5, 6.5, 1.4, -1.2, head_width=0.05, head_length=0.05, 
            fc='blue', ec='blue')
    
    # Outputs
    outputs = ['Position Noise', 'Lattice Noise', 'Atom Type Logits']
    for i, output in enumerate(outputs):
        out_box = FancyBboxPatch((6 + i*1.2, 0.5), 1.1, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#E6F3FF', edgecolor='navy')
        ax.add_patch(out_box)
        ax.text(6.55 + i*1.2, 0.8, output, fontsize=10, 
                fontweight='bold', ha='center')
    
    # Arrow from output heads to outputs
    ax.arrow(8.25, 1.4, 0, -0.2, head_width=0.05, head_length=0.02, 
            fc='navy', ec='navy')
    
    # Side annotations
    ax.text(0.5, 4.5, 'Key GemNet Features:', fontsize=12, fontweight='bold')
    features = [
        '• Directional message passing',
        '• Triplet interactions (3-body)',
        '• Spherical harmonics features',
        '• Efficient periodic handling',
        '• Shared parameters across blocks'
    ]
    for i, feature in enumerate(features):
        ax.text(0.5, 4.2 - i*0.3, feature, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"GemNet detail diagram saved to: {save_path}")

if __name__ == "__main__":
    print("Creating MatterGen architecture diagrams...")
    
    # Create main architecture diagram
    create_mattergen_architecture_diagram(
        save_path="mattergen_architecture.png",
        figsize=(20, 14)
    )
    
    # Create detailed GemNet diagram
    create_detailed_gemnet_diagram(
        save_path="gemnet_detail.png", 
        figsize=(16, 10)
    )
    
    print("\nDiagrams created successfully!")
    print("- mattergen_architecture.png: Complete generation process")
    print("- gemnet_detail.png: Detailed GemNet architecture")