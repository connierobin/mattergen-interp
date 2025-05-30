#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np

def create_detailed_data_flow_with_nn_details(save_path="mattergen_detailed_flow_nn.png", figsize=(26, 18)):
    """
    Create a detailed data flow diagram showing tensor shapes, transformations, and neural network details.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Define colors for different data types
    colors = {
        'input': '#E8F4FD',
        'tensor': '#FFE6CC', 
        'function': '#E8F8E8',
        'nn': '#FFE6F0',
        'output': '#F0E6FF',
        'parameters': '#F0F8E8'
    }
    
    # Title
    ax.text(6.5, 17.5, 'MatterGen: Detailed Architecture with Neural Network Specifications', 
            fontsize=18, fontweight='bold', ha='center')
    
    # =================== INITIALIZATION ===================
    y_start = 16.3
    
    # Conditioning Data Input
    cond_box = FancyBboxPatch((0.2, y_start), 2.5, 1.4,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['input'], edgecolor='blue', linewidth=2)
    ax.add_patch(cond_box)
    ax.text(1.45, y_start + 1.1, 'Conditioning Data', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.45, y_start + 0.8, 'num_atoms: [B]', fontsize=10, ha='center', family='monospace')
    ax.text(1.45, y_start + 0.6, 'chemical_system: [B, 119]', fontsize=10, ha='center', family='monospace')
    ax.text(1.45, y_start + 0.4, 'properties: [B, 1]', fontsize=10, ha='center', family='monospace')
    ax.text(1.45, y_start + 0.2, 'Multi-hot or scalar', fontsize=10, ha='center', style='italic')
    
    # Prior Sampling
    prior_box = FancyBboxPatch((3.2, y_start), 2.8, 1.4,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['function'], edgecolor='green')
    ax.add_patch(prior_box)
    ax.text(4.6, y_start + 1.1, 'Prior Sampling', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.6, y_start + 0.8, 'corruption.prior_sampling()', fontsize=10, ha='center', family='monospace')
    ax.text(4.6, y_start + 0.6, 'Generates initial noise', fontsize=10, ha='center')
    ax.text(4.6, y_start + 0.4, 'Continuous: Gaussian N(0,1)', fontsize=10, ha='center')
    ax.text(4.6, y_start + 0.2, 'Discrete: Uniform categorical', fontsize=10, ha='center')
    
    # Initial Tensors
    init_box = FancyBboxPatch((6.5, y_start), 3, 1.4,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['tensor'], edgecolor='orange')
    ax.add_patch(init_box)
    ax.text(8, y_start + 1.1, 'Initial Noisy Tensors', fontsize=12, fontweight='bold', ha='center')
    ax.text(8, y_start + 0.8, 'pos: [N_total, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(8, y_start + 0.6, 'cell: [B, 3, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(8, y_start + 0.4, 'atomic_numbers: [N_total]', fontsize=10, ha='center', family='monospace')
    ax.text(8, y_start + 0.2, 'N_total = sum(num_atoms)', fontsize=10, ha='center', style='italic')
    
    # Arrows
    ax.arrow(2.7, y_start + 0.7, 0.4, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.arrow(6.0, y_start + 0.7, 0.4, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # =================== PROPERTY EMBEDDINGS ===================
    y_prop = 14.5
    
    # Property embedding details
    prop_embed_box = FancyBboxPatch((10, y_start), 2.8, 1.4,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['nn'], edgecolor='purple')
    ax.add_patch(prop_embed_box)
    ax.text(11.4, y_start + 1.1, 'Property Embeddings', fontsize=11, fontweight='bold', ha='center')
    ax.text(11.4, y_start + 0.9, 'Each → 512D vector', fontsize=10, ha='center')
    ax.text(11.4, y_start + 0.7, 'Chemical System:', fontsize=9, ha='center', fontweight='bold')
    ax.text(11.4, y_start + 0.55, 'Embedding(119, 512)', fontsize=9, ha='center', family='monospace')
    ax.text(11.4, y_start + 0.4, 'Numeric Properties:', fontsize=9, ha='center', fontweight='bold')
    ax.text(11.4, y_start + 0.25, 'NoiseLevelEncoding(512)', fontsize=9, ha='center', family='monospace')
    ax.text(11.4, y_start + 0.1, '~60K params each', fontsize=9, ha='center', color='red')
    
    # =================== DENOISING LOOP ===================
    y_loop = 14.3
    
    # Loop header
    loop_box = FancyBboxPatch((0.2, y_loop), 12.6, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor='#E0E0FF', edgecolor='purple', linewidth=2)
    ax.add_patch(loop_box)
    ax.text(6.5, y_loop + 0.3, 'Denoising Loop: for t in [T_max, T_max-dt, ..., ε] (typically 1000 steps)', 
            fontsize=14, fontweight='bold', ha='center')
    
    # =================== TIME ENCODING ===================
    y_time = 13.2
    
    time_box = FancyBboxPatch((0.2, y_time), 2.5, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['nn'], edgecolor='green')
    ax.add_patch(time_box)
    ax.text(1.45, y_time + 0.8, 'Time Encoding', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.45, y_time + 0.6, 't: [B] → t_enc: [B, 512]', fontsize=10, ha='center', family='monospace')
    ax.text(1.45, y_time + 0.4, 'Sinusoidal positional encoding', fontsize=9, ha='center')
    ax.text(1.45, y_time + 0.2, 'No learnable params', fontsize=9, ha='center', color='red')
    
    # =================== GEMNET PROCESSING ===================
    y_gemnet = 11.0
    
    # Graph Construction
    graph_box = FancyBboxPatch((0.2, y_gemnet), 2.5, 1.8,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['function'], edgecolor='blue')
    ax.add_patch(graph_box)
    ax.text(1.45, y_gemnet + 1.5, 'Graph Construction', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.45, y_gemnet + 1.25, 'frac_to_cart_coords()', fontsize=10, ha='center', family='monospace')
    ax.text(1.45, y_gemnet + 1.05, 'generate_interaction_graph()', fontsize=9, ha='center', family='monospace')
    ax.text(1.45, y_gemnet + 0.85, 'Cutoff: 7.0 Å', fontsize=10, ha='center')
    ax.text(1.45, y_gemnet + 0.65, 'Max neighbors: 50', fontsize=10, ha='center')
    ax.text(1.45, y_gemnet + 0.45, 'Max cell images: 5³', fontsize=10, ha='center')
    ax.text(1.45, y_gemnet + 0.25, 'Periodic boundaries', fontsize=10, ha='center')
    ax.text(1.45, y_gemnet + 0.05, 'Non-parametric', fontsize=9, ha='center', color='red')
    
    # Graph Tensors
    graph_tensors = FancyBboxPatch((3.0, y_gemnet), 2.8, 1.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['tensor'], edgecolor='orange')
    ax.add_patch(graph_tensors)
    ax.text(4.4, y_gemnet + 1.5, 'Graph Tensors', fontsize=11, fontweight='bold', ha='center')
    ax.text(4.4, y_gemnet + 1.25, 'edge_index: [2, E]', fontsize=10, ha='center', family='monospace')
    ax.text(4.4, y_gemnet + 1.05, 'distances: [E]', fontsize=10, ha='center', family='monospace')
    ax.text(4.4, y_gemnet + 0.85, 'unit_vectors: [E, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(4.4, y_gemnet + 0.65, 'triplet_indices: [T, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(4.4, y_gemnet + 0.45, 'RBF features: [E, 128]', fontsize=10, ha='center', family='monospace')
    ax.text(4.4, y_gemnet + 0.25, 'CBF features: [T, 16]', fontsize=10, ha='center', family='monospace')
    ax.text(4.4, y_gemnet + 0.05, 'E ≈ 50×N, T ≈ 200×N', fontsize=9, ha='center', style='italic')
    
    # Atom Embedding
    embed_box = FancyBboxPatch((6.1, y_gemnet), 2.5, 1.8,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['nn'], edgecolor='green')
    ax.add_patch(embed_box)
    ax.text(7.35, y_gemnet + 1.5, 'Atom Embedding', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.35, y_gemnet + 1.25, 'Embedding(119, 512)', fontsize=10, ha='center', family='monospace')
    ax.text(7.35, y_gemnet + 1.05, '+ z_global: [N, 512×P]', fontsize=10, ha='center', family='monospace')
    ax.text(7.35, y_gemnet + 0.85, 'Linear(1024, 512)', fontsize=10, ha='center', family='monospace')
    ax.text(7.35, y_gemnet + 0.65, 'h: [N_total, 512]', fontsize=10, ha='center', family='monospace')
    ax.text(7.35, y_gemnet + 0.45, 'Parameters:', fontsize=9, ha='center', fontweight='bold')
    ax.text(7.35, y_gemnet + 0.25, '119×512 = 61K (embed)', fontsize=9, ha='center', color='red')
    ax.text(7.35, y_gemnet + 0.05, '1024×512 = 524K (proj)', fontsize=9, ha='center', color='red')
    
    # Message Passing
    mp_box = FancyBboxPatch((9.0, y_gemnet), 2.5, 1.8,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['nn'], edgecolor='red')
    ax.add_patch(mp_box)
    ax.text(10.25, y_gemnet + 1.5, 'Message Passing', fontsize=11, fontweight='bold', ha='center')
    ax.text(10.25, y_gemnet + 1.25, '4× Interaction Blocks', fontsize=10, ha='center', family='monospace')
    ax.text(10.25, y_gemnet + 1.05, 'Edge embedding: [E, 512]', fontsize=10, ha='center', family='monospace')
    ax.text(10.25, y_gemnet + 0.85, 'Triplet interactions', fontsize=10, ha='center')
    ax.text(10.25, y_gemnet + 0.65, 'Spherical harmonics (L≤7)', fontsize=10, ha='center')
    ax.text(10.25, y_gemnet + 0.45, 'Parameters per block:', fontsize=9, ha='center', fontweight='bold')
    ax.text(10.25, y_gemnet + 0.25, '~2-3M params each', fontsize=9, ha='center', color='red')
    ax.text(10.25, y_gemnet + 0.05, 'Total: ~10-12M params', fontsize=9, ha='center', color='red')
    
    # Output Heads
    output_heads_box = FancyBboxPatch((11.8, y_gemnet), 1.0, 1.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor=colors['nn'], edgecolor='purple')
    ax.add_patch(output_heads_box)
    ax.text(12.3, y_gemnet + 1.5, 'Output', fontsize=10, fontweight='bold', ha='center')
    ax.text(12.3, y_gemnet + 1.3, 'Heads', fontsize=10, fontweight='bold', ha='center')
    ax.text(12.3, y_gemnet + 1.05, '3 residual', fontsize=9, ha='center')
    ax.text(12.3, y_gemnet + 0.85, 'blocks each', fontsize=9, ha='center')
    ax.text(12.3, y_gemnet + 0.65, 'Energy', fontsize=9, ha='center')
    ax.text(12.3, y_gemnet + 0.45, 'Forces', fontsize=9, ha='center')
    ax.text(12.3, y_gemnet + 0.25, 'Stress', fontsize=9, ha='center')
    ax.text(12.3, y_gemnet + 0.05, '~2M params', fontsize=8, ha='center', color='red')
    
    # Arrows between GemNet stages
    ax.arrow(2.7, y_gemnet + 0.9, 0.2, 0, head_width=0.04, head_length=0.04, fc='black', ec='black')
    ax.arrow(5.8, y_gemnet + 0.9, 0.2, 0, head_width=0.04, head_length=0.04, fc='black', ec='black')
    ax.arrow(8.6, y_gemnet + 0.9, 0.2, 0, head_width=0.04, head_length=0.04, fc='black', ec='black')
    ax.arrow(11.5, y_gemnet + 0.9, 0.2, 0, head_width=0.04, head_length=0.04, fc='black', ec='black')
    
    # =================== PARAMETER COUNT SUMMARY ===================
    param_box = FancyBboxPatch((0.2, y_gemnet - 1.5), 12.6, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['parameters'], edgecolor='darkgreen', linewidth=2)
    ax.add_patch(param_box)
    ax.text(6.5, y_gemnet - 0.8, 'GemNet-T Total Parameters: ~15-20M', 
            fontsize=14, fontweight='bold', ha='center')
    ax.text(6.5, y_gemnet - 1.1, 'Atom embed: 61K | Edge embed: 590K | 4× Interaction: 10-12M | Output heads: 2-3M | Basis functions: 500K', 
            fontsize=11, ha='center')
    
    # =================== SCORE CONVERSION ===================
    y_conv = 8.5
    
    # Coordinate transformation
    coord_box = FancyBboxPatch((2, y_conv), 3.5, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['function'], edgecolor='blue')
    ax.add_patch(coord_box)
    ax.text(3.75, y_conv + 0.9, 'Score Conversion', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.75, y_conv + 0.7, 'Cart forces → Frac position scores', fontsize=10, ha='center')
    ax.text(3.75, y_conv + 0.5, 'pos_score = (cell⁻¹)ᵀ @ pred_forces', fontsize=10, ha='center', family='monospace')
    ax.text(3.75, y_conv + 0.3, 'Element masking for atom types', fontsize=10, ha='center')
    ax.text(3.75, y_conv + 0.1, 'No learnable parameters', fontsize=9, ha='center', color='red')
    
    # Final Scores
    scores_box = FancyBboxPatch((6.5, y_conv), 3, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['tensor'], edgecolor='orange')
    ax.add_patch(scores_box)
    ax.text(8, y_conv + 0.9, 'Noise Predictions', fontsize=11, fontweight='bold', ha='center')
    ax.text(8, y_conv + 0.7, 'pos_score: [N_total, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(8, y_conv + 0.5, 'cell_score: [B, 3, 3]', fontsize=10, ha='center', family='monospace')
    ax.text(8, y_conv + 0.3, 'atom_score: [N_total, 119]', fontsize=10, ha='center', family='monospace')
    ax.text(8, y_conv + 0.1, 'All float32 tensors', fontsize=9, ha='center', style='italic')
    
    # Arrow from MP to conversion
    ax.arrow(10.25, y_gemnet, -4.5, y_conv + 0.6 - y_gemnet, head_width=0.05, head_length=0.05, 
            fc='purple', ec='purple', linestyle='--')
    ax.arrow(5.5, y_conv + 0.6, 0.9, 0, head_width=0.04, head_length=0.04, fc='black', ec='black')
    
    # =================== PREDICTOR/CORRECTOR ===================
    y_pc = 6.5
    
    # Corrector
    corr_box = FancyBboxPatch((1, y_pc), 3, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['function'], edgecolor='red')
    ax.add_patch(corr_box)
    ax.text(2.5, y_pc + 1.2, 'Corrector Steps', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.5, y_pc + 1.0, 'n_corrector iterations (typ. 1)', fontsize=10, ha='center')
    ax.text(2.5, y_pc + 0.8, 'Langevin MCMC refinement', fontsize=10, ha='center')
    ax.text(2.5, y_pc + 0.6, 'dx = score × dt + noise × √dt', fontsize=10, ha='center', family='monospace')
    ax.text(2.5, y_pc + 0.4, 'x ← x + dx', fontsize=10, ha='center', family='monospace')
    ax.text(2.5, y_pc + 0.2, 'Different SDEs per modality', fontsize=10, ha='center')
    ax.text(2.5, y_pc + 0.0, 'No learnable parameters', fontsize=9, ha='center', color='red')
    
    # Predictor  
    pred_box = FancyBboxPatch((4.5, y_pc), 3, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['function'], edgecolor='green')
    ax.add_patch(pred_box)
    ax.text(6, y_pc + 1.2, 'Predictor Step', fontsize=11, fontweight='bold', ha='center')
    ax.text(6, y_pc + 1.0, 'Euler-Maruyama step', fontsize=10, ha='center')
    ax.text(6, y_pc + 0.8, 'Continuous: dx = score × dt', fontsize=10, ha='center', family='monospace')
    ax.text(6, y_pc + 0.6, 'Discrete: D3PM transitions', fontsize=10, ha='center', family='monospace')
    ax.text(6, y_pc + 0.4, 'x_{t-dt} ← x_t + dx', fontsize=10, ha='center', family='monospace')
    ax.text(6, y_pc + 0.2, 'Adaptive step size', fontsize=10, ha='center')
    ax.text(6, y_pc + 0.0, 'No learnable parameters', fontsize=9, ha='center', color='red')
    
    # Updated tensors
    update_box = FancyBboxPatch((8.0, y_pc), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['tensor'], edgecolor='orange')
    ax.add_patch(update_box)
    ax.text(9.5, y_pc + 1.2, 'Updated Tensors', fontsize=11, fontweight='bold', ha='center')
    ax.text(9.5, y_pc + 1.0, 'pos: [N_total, 3] ←', fontsize=10, ha='center', family='monospace')
    ax.text(9.5, y_pc + 0.8, 'cell: [B, 3, 3] ←', fontsize=10, ha='center', family='monospace')
    ax.text(9.5, y_pc + 0.6, 'atomic_numbers: [N_total] ←', fontsize=10, ha='center', family='monospace')
    ax.text(9.5, y_pc + 0.4, 'Less noisy than before', fontsize=10, ha='center')
    ax.text(9.5, y_pc + 0.2, 'Gradual denoising process', fontsize=10, ha='center')
    ax.text(9.5, y_pc + 0.0, 'Same shapes, refined values', fontsize=9, ha='center', style='italic')
    
    # Arrows
    ax.arrow(8, y_conv, -5, y_pc + 0.75 - y_conv, head_width=0.05, head_length=0.05, 
            fc='orange', ec='orange', linestyle='--')
    ax.arrow(4, y_pc + 0.75, 0.4, 0, head_width=0.04, head_length=0.04, fc='black', ec='black')
    ax.arrow(7.5, y_pc + 0.75, 0.4, 0, head_width=0.04, head_length=0.04, fc='black', ec='black')
    
    # Loop back arrow
    ax.arrow(9.5, y_pc, 3, 0, head_width=0.05, head_length=0.05, fc='purple', ec='purple')
    ax.arrow(12.5, y_pc + 0.75, 0, y_time - y_pc - 0.75, head_width=0.05, head_length=0.05, 
            fc='purple', ec='purple')
    ax.text(12.8, y_time - 0.5, 'Loop back\nfor next\ntimestep', fontsize=10, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='purple'))
    
    # =================== FINAL OUTPUT ===================
    y_final = 4.0
    
    # Structure conversion
    struct_box = FancyBboxPatch((2.5, y_final), 3.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['function'], edgecolor='blue')
    ax.add_patch(struct_box)
    ax.text(4.25, y_final + 1.2, 'Structure Conversion', fontsize=11, fontweight='bold', ha='center')
    ax.text(4.25, y_final + 1.0, 'lattice_matrix_to_params()', fontsize=10, ha='center', family='monospace')
    ax.text(4.25, y_final + 0.8, 'get_crystals_list()', fontsize=10, ha='center', family='monospace')
    ax.text(4.25, y_final + 0.6, 'make_structure()', fontsize=10, ha='center', family='monospace')
    ax.text(4.25, y_final + 0.4, 'Split by num_atoms per crystal', fontsize=10, ha='center')
    ax.text(4.25, y_final + 0.2, 'Convert to pymatgen objects', fontsize=10, ha='center')
    ax.text(4.25, y_final + 0.0, 'No learnable parameters', fontsize=9, ha='center', color='red')
    
    # Final structures
    final_box = FancyBboxPatch((7, y_final), 3.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['output'], edgecolor='purple', linewidth=2)
    ax.add_patch(final_box)
    ax.text(8.75, y_final + 1.2, 'Pymatgen Structures', fontsize=11, fontweight='bold', ha='center')
    ax.text(8.75, y_final + 1.0, 'List[Structure]', fontsize=10, ha='center', family='monospace')
    ax.text(8.75, y_final + 0.8, 'Lattice: (a,b,c,α,β,γ)', fontsize=10, ha='center')
    ax.text(8.75, y_final + 0.6, 'Fractional coordinates', fontsize=10, ha='center')
    ax.text(8.75, y_final + 0.4, 'Element symbols', fontsize=10, ha='center')
    ax.text(8.75, y_final + 0.2, 'Valid crystal structures', fontsize=10, ha='center')
    ax.text(8.75, y_final + 0.0, 'Ready for property analysis', fontsize=9, ha='center', style='italic')
    
    # Final arrows
    ax.arrow(9.5, y_pc, -6.75, y_final + 0.75 - y_pc, head_width=0.05, head_length=0.05, 
            fc='green', ec='green', linestyle='--')
    ax.arrow(6, y_final + 0.75, 0.9, 0, head_width=0.04, head_length=0.04, fc='black', ec='black')
    
    # =================== LEGEND ===================
    y_legend = 2.5
    ax.text(0.5, y_legend, 'Legend:', fontsize=12, fontweight='bold')
    
    legend_items = [
        (colors['input'], 'Input Data'),
        (colors['function'], 'Operations'), 
        (colors['nn'], 'Neural Networks'),
        (colors['tensor'], 'Tensor States'),
        (colors['output'], 'Final Outputs'),
        (colors['parameters'], 'Parameter Counts')
    ]
    
    for i, (color, label) in enumerate(legend_items):
        legend_box = Rectangle((0.5, y_legend - 0.5 - i*0.25), 0.3, 0.15, 
                              facecolor=color, edgecolor='black')
        ax.add_patch(legend_box)
        ax.text(0.9, y_legend - 0.425 - i*0.25, label, fontsize=10, va='center')
    
    # Key
    ax.text(7, y_legend, 'Key Notation:', fontsize=12, fontweight='bold')
    ax.text(7, y_legend - 0.3, 'B = batch_size, N = num_atoms_total, E = num_edges', 
            fontsize=10, family='monospace')
    ax.text(7, y_legend - 0.5, 'P = num_properties, MAX_Z = 119 (atomic numbers)', 
            fontsize=10, family='monospace')
    ax.text(7, y_legend - 0.7, 'Red text = parameter counts', 
            fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Detailed flow diagram with NN details saved to: {save_path}")

if __name__ == "__main__":
    create_detailed_data_flow_with_nn_details()