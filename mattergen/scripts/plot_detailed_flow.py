#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np

def create_detailed_data_flow_diagram(save_path="mattergen_detailed_flow.png", figsize=(24, 16)):
    """
    Create a detailed data flow diagram showing tensor shapes and transformations.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Define colors for different data types
    colors = {
        'input': '#E8F4FD',
        'tensor': '#FFE6CC', 
        'function': '#E8F8E8',
        'output': '#F0E6FF',
        'shape': '#FFF0E6'
    }
    
    # Title
    ax.text(6, 15.5, 'MatterGen: Detailed Information Flow During Generation', 
            fontsize=18, fontweight='bold', ha='center')
    
    # =================== INITIALIZATION ===================
    y_start = 14.5
    
    # Conditioning Data Input
    cond_box = FancyBboxPatch((0.2, y_start), 2.3, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['input'], edgecolor='blue', linewidth=2)
    ax.add_patch(cond_box)
    ax.text(1.35, y_start + 0.9, 'Conditioning Data', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.35, y_start + 0.6, 'num_atoms: [B]', fontsize=10, ha='center', family='monospace')
    ax.text(1.35, y_start + 0.4, 'chemical_system: [B, MAX_Z+1]', fontsize=10, ha='center', family='monospace')
    ax.text(1.35, y_start + 0.2, 'properties: various shapes', fontsize=10, ha='center', family='monospace')
    
    # Prior Sampling
    prior_box = FancyBboxPatch((3, y_start), 2.5, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['function'], edgecolor='green')
    ax.add_patch(prior_box)
    ax.text(4.25, y_start + 0.9, 'Prior Sampling', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.25, y_start + 0.6, 'corruption.prior_sampling()', fontsize=10, ha='center', family='monospace')
    ax.text(4.25, y_start + 0.4, 'Generates initial noise', fontsize=10, ha='center')
    ax.text(4.25, y_start + 0.2, 'for pos, cell, atomic_numbers', fontsize=10, ha='center')
    
    # Initial Tensors
    init_box = FancyBboxPatch((6, y_start), 2.8, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['tensor'], edgecolor='orange')
    ax.add_patch(init_box)
    ax.text(7.4, y_start + 0.9, 'Initial Noisy Tensors', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.4, y_start + 0.6, 'pos: [N_total, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(7.4, y_start + 0.4, 'cell: [B, 3, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(7.4, y_start + 0.2, 'atomic_numbers: [N_total]', fontsize=10, ha='center', family='monospace')
    
    # Arrows
    ax.arrow(2.5, y_start + 0.6, 0.4, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.arrow(5.5, y_start + 0.6, 0.4, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # =================== DENOISING LOOP ===================
    y_loop = 12.8
    
    # Loop header
    loop_box = FancyBboxPatch((0.2, y_loop), 11.6, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor='#E0E0FF', edgecolor='purple', linewidth=2)
    ax.add_patch(loop_box)
    ax.text(6, y_loop + 0.3, 'Denoising Loop: for t in [T_max, T_max-dt, ..., ε]', 
            fontsize=14, fontweight='bold', ha='center')
    
    # =================== SCORE COMPUTATION ===================
    y_score = 11.5
    
    # Time encoding
    time_box = FancyBboxPatch((0.2, y_score), 2, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['function'], edgecolor='green')
    ax.add_patch(time_box)
    ax.text(1.2, y_score + 0.7, 'Time Encoding', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.2, y_score + 0.5, 't: [B] → t_enc: [B, D]', fontsize=10, ha='center', family='monospace')
    ax.text(1.2, y_score + 0.3, 'noise_level_encoding()', fontsize=10, ha='center', family='monospace')
    ax.text(1.2, y_score + 0.1, '+ property embeddings', fontsize=10, ha='center')
    
    # =================== GEMNET PROCESSING ===================
    y_gemnet = 9.8
    
    # Graph Construction
    graph_box = FancyBboxPatch((0.2, y_gemnet), 2.2, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['function'], edgecolor='blue')
    ax.add_patch(graph_box)
    ax.text(1.3, y_gemnet + 0.9, 'Graph Construction', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.3, y_gemnet + 0.7, 'frac_to_cart_coords()', fontsize=10, ha='center', family='monospace')
    ax.text(1.3, y_gemnet + 0.5, 'generate_interaction_graph()', fontsize=10, ha='center', family='monospace')
    ax.text(1.3, y_gemnet + 0.3, 'Creates edges within cutoff', fontsize=10, ha='center')
    ax.text(1.3, y_gemnet + 0.1, 'Handles periodic boundaries', fontsize=10, ha='center')
    
    # Graph Tensors
    graph_tensors = FancyBboxPatch((2.7, y_gemnet), 2.5, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['tensor'], edgecolor='orange')
    ax.add_patch(graph_tensors)
    ax.text(3.95, y_gemnet + 0.9, 'Graph Tensors', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.95, y_gemnet + 0.7, 'edge_index: [2, E]', fontsize=10, ha='center', family='monospace')
    ax.text(3.95, y_gemnet + 0.5, 'distances: [E]', fontsize=10, ha='center', family='monospace')
    ax.text(3.95, y_gemnet + 0.3, 'unit_vectors: [E, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(3.95, y_gemnet + 0.1, 'triplet_indices: [T, 3]', fontsize=10, ha='center', family='monospace')
    
    # Embedding
    embed_box = FancyBboxPatch((5.5, y_gemnet), 2.2, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['function'], edgecolor='green')
    ax.add_patch(embed_box)
    ax.text(6.6, y_gemnet + 0.9, 'Atom Embedding', fontsize=11, fontweight='bold', ha='center')
    ax.text(6.6, y_gemnet + 0.7, 'atom_emb(atomic_numbers)', fontsize=10, ha='center', family='monospace')
    ax.text(6.6, y_gemnet + 0.5, 'z_per_atom = z[batch_idx]', fontsize=10, ha='center', family='monospace')
    ax.text(6.6, y_gemnet + 0.3, 'h = cat([h, z], dim=1)', fontsize=10, ha='center', family='monospace')
    ax.text(6.6, y_gemnet + 0.1, 'h: [N_total, D_atom]', fontsize=10, ha='center', family='monospace')
    
    # Message Passing
    mp_box = FancyBboxPatch((8, y_gemnet), 2.2, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['function'], edgecolor='red')
    ax.add_patch(mp_box)
    ax.text(9.1, y_gemnet + 0.9, 'Message Passing', fontsize=11, fontweight='bold', ha='center')
    ax.text(9.1, y_gemnet + 0.7, '4x Interaction Blocks', fontsize=10, ha='center', family='monospace')
    ax.text(9.1, y_gemnet + 0.5, 'Triplet interactions', fontsize=10, ha='center')
    ax.text(9.1, y_gemnet + 0.3, 'Edge updates', fontsize=10, ha='center')
    ax.text(9.1, y_gemnet + 0.1, 'Node feature refinement', fontsize=10, ha='center')
    
    # GNN Outputs
    gnn_out_box = FancyBboxPatch((10.5, y_gemnet), 1.3, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['output'], edgecolor='purple')
    ax.add_patch(gnn_out_box)
    ax.text(11.15, y_gemnet + 0.9, 'GNN Outputs', fontsize=11, fontweight='bold', ha='center')
    ax.text(11.15, y_gemnet + 0.7, 'forces: [N, 3]', fontsize=9, ha='center', family='monospace')
    ax.text(11.15, y_gemnet + 0.5, 'stress: [B, 3, 3]', fontsize=9, ha='center', family='monospace')
    ax.text(11.15, y_gemnet + 0.3, 'atom_logits:', fontsize=9, ha='center', family='monospace')
    ax.text(11.15, y_gemnet + 0.1, '[N, MAX_Z+1]', fontsize=9, ha='center', family='monospace')
    
    # Arrows between GemNet stages
    ax.arrow(2.4, y_gemnet + 0.6, 0.2, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
    ax.arrow(5.2, y_gemnet + 0.6, 0.2, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
    ax.arrow(7.7, y_gemnet + 0.6, 0.2, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
    ax.arrow(10.2, y_gemnet + 0.6, 0.2, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
    
    # =================== SCORE CONVERSION ===================
    y_conv = 8.2
    
    # Coordinate transformation
    coord_box = FancyBboxPatch((2, y_conv), 3, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['function'], edgecolor='blue')
    ax.add_patch(coord_box)
    ax.text(3.5, y_conv + 0.7, 'Score Conversion', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.5, y_conv + 0.5, 'Cart forces → Frac position scores', fontsize=10, ha='center')
    ax.text(3.5, y_conv + 0.3, 'pos_score = (cell⁻¹)ᵀ @ pred_forces', fontsize=10, ha='center', family='monospace')
    ax.text(3.5, y_conv + 0.1, 'Element masking for atom types', fontsize=10, ha='center')
    
    # Final Scores
    scores_box = FancyBboxPatch((6, y_conv), 2.5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['tensor'], edgecolor='orange')
    ax.add_patch(scores_box)
    ax.text(7.25, y_conv + 0.7, 'Noise Predictions', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.25, y_conv + 0.5, 'pos_score: [N_total, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(7.25, y_conv + 0.3, 'cell_score: [B, 3, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(7.25, y_conv + 0.1, 'atom_score: [N_total, MAX_Z+1]', fontsize=10, ha='center', family='monospace')
    
    # Arrow from GNN to conversion
    ax.arrow(11.15, y_gemnet, -6, y_conv + 0.5 - y_gemnet, head_width=0.05, head_length=0.05, 
            fc='purple', ec='purple', linestyle='--')
    ax.arrow(5, y_conv + 0.5, 0.9, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
    
    # =================== PREDICTOR/CORRECTOR ===================
    y_pc = 6.5
    
    # Corrector
    corr_box = FancyBboxPatch((1, y_pc), 2.5, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['function'], edgecolor='red')
    ax.add_patch(corr_box)
    ax.text(2.25, y_pc + 0.9, 'Corrector Steps', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.25, y_pc + 0.7, 'n_corrector iterations', fontsize=10, ha='center')
    ax.text(2.25, y_pc + 0.5, 'Langevin MCMC refinement', fontsize=10, ha='center')
    ax.text(2.25, y_pc + 0.3, 'dx = score * dt + noise * √dt', fontsize=10, ha='center', family='monospace')
    ax.text(2.25, y_pc + 0.1, 'x ← x + dx', fontsize=10, ha='center', family='monospace')
    
    # Predictor  
    pred_box = FancyBboxPatch((4, y_pc), 2.5, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['function'], edgecolor='green')
    ax.add_patch(pred_box)
    ax.text(5.25, y_pc + 0.9, 'Predictor Step', fontsize=11, fontweight='bold', ha='center')
    ax.text(5.25, y_pc + 0.7, 'Euler-Maruyama step', fontsize=10, ha='center')
    ax.text(5.25, y_pc + 0.5, 'dx = score * dt', fontsize=10, ha='center', family='monospace')
    ax.text(5.25, y_pc + 0.3, 'x_{t-dt} ← x_t + dx', fontsize=10, ha='center', family='monospace')
    ax.text(5.25, y_pc + 0.1, 'Different for discrete vars', fontsize=10, ha='center')
    
    # Updated tensors
    update_box = FancyBboxPatch((7, y_pc), 2.5, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['tensor'], edgecolor='orange')
    ax.add_patch(update_box)
    ax.text(8.25, y_pc + 0.9, 'Updated Tensors', fontsize=11, fontweight='bold', ha='center')
    ax.text(8.25, y_pc + 0.7, 'pos: [N_total, 3] ←', fontsize=10, ha='center', family='monospace')
    ax.text(8.25, y_pc + 0.5, 'cell: [B, 3, 3] ←', fontsize=10, ha='center', family='monospace')
    ax.text(8.25, y_pc + 0.3, 'atomic_numbers: [N_total] ←', fontsize=10, ha='center', family='monospace')
    ax.text(8.25, y_pc + 0.1, 'Less noisy than before', fontsize=10, ha='center')
    
    # Arrows
    ax.arrow(7.25, y_conv, -4, y_pc + 0.6 - y_conv, head_width=0.05, head_length=0.05, 
            fc='orange', ec='orange', linestyle='--')
    ax.arrow(3.5, y_pc + 0.6, 0.4, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
    ax.arrow(6.5, y_pc + 0.6, 0.4, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
    
    # Loop back arrow
    ax.arrow(8.25, y_pc, 3, 0, head_width=0.05, head_length=0.05, fc='purple', ec='purple')
    ax.arrow(11.25, y_pc + 0.6, 0, y_score - y_pc - 0.6, head_width=0.05, head_length=0.05, 
            fc='purple', ec='purple')
    ax.text(11.5, y_score - 0.5, 'Loop back\nfor next\ntimestep', fontsize=10, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='purple'))
    
    # =================== FINAL OUTPUT ===================
    y_final = 4.5
    
    # Structure conversion
    struct_box = FancyBboxPatch((2, y_final), 3, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['function'], edgecolor='blue')
    ax.add_patch(struct_box)
    ax.text(3.5, y_final + 0.9, 'Structure Conversion', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.5, y_final + 0.7, 'lattice_matrix_to_params()', fontsize=10, ha='center', family='monospace')
    ax.text(3.5, y_final + 0.5, 'get_crystals_list()', fontsize=10, ha='center', family='monospace')
    ax.text(3.5, y_final + 0.3, 'make_structure()', fontsize=10, ha='center', family='monospace')
    ax.text(3.5, y_final + 0.1, 'Split by num_atoms per crystal', fontsize=10, ha='center')
    
    # Final structures
    final_box = FancyBboxPatch((6, y_final), 3, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['output'], edgecolor='purple', linewidth=2)
    ax.add_patch(final_box)
    ax.text(7.5, y_final + 0.9, 'Pymatgen Structures', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, y_final + 0.7, 'List[Structure]', fontsize=10, ha='center', family='monospace')
    ax.text(7.5, y_final + 0.5, 'Lattice parameters (a,b,c,α,β,γ)', fontsize=10, ha='center')
    ax.text(7.5, y_final + 0.3, 'Fractional coordinates', fontsize=10, ha='center')
    ax.text(7.5, y_final + 0.1, 'Element symbols', fontsize=10, ha='center')
    
    # Final arrows
    ax.arrow(8.25, y_pc, -5.75, y_final + 0.6 - y_pc, head_width=0.05, head_length=0.05, 
            fc='green', ec='green', linestyle='--')
    ax.arrow(5, y_final + 0.6, 0.9, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
    
    # =================== LEGEND ===================
    y_legend = 3
    ax.text(0.5, y_legend, 'Legend:', fontsize=12, fontweight='bold')
    
    legend_items = [
        (colors['input'], 'Input Data'),
        (colors['function'], 'Operations/Functions'), 
        (colors['tensor'], 'Tensor States'),
        (colors['output'], 'Final Outputs')
    ]
    
    for i, (color, label) in enumerate(legend_items):
        legend_box = Rectangle((0.5, y_legend - 0.5 - i*0.3), 0.3, 0.2, 
                              facecolor=color, edgecolor='black')
        ax.add_patch(legend_box)
        ax.text(0.9, y_legend - 0.4 - i*0.3, label, fontsize=10, va='center')
    
    # Key
    ax.text(0.5, y_legend - 2.2, 'Key Tensor Shape Notation:', fontsize=12, fontweight='bold')
    ax.text(0.5, y_legend - 2.5, 'B = batch_size, N_total = sum(num_atoms), E = num_edges', 
            fontsize=10, family='monospace')
    ax.text(0.5, y_legend - 2.7, 'MAX_Z = maximum atomic number, D = embedding dimension', 
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Detailed flow diagram saved to: {save_path}")

if __name__ == "__main__":
    create_detailed_data_flow_diagram()