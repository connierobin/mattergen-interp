# MatterGen Architecture Diagrams

This directory contains comprehensive visualizations and documentation of the MatterGen architecture, focusing on the detailed information flow during crystal structure generation.

## Files Overview

### 📊 Visual Diagrams
- **`mattergen_architecture.png`** - High-level architecture overview showing the main components
- **`gemnet_detail.png`** - Detailed GemNet-T neural network architecture
- **`mattergen_detailed_flow.png`** - Complete data flow with tensor shapes and transformations
- **`mattergen_detailed_flow_nn.png`** - **[RECOMMENDED]** Most comprehensive diagram with neural network specifications

### 📝 Text Documentation
- **`mattergen_data_flow_ascii.txt`** - ASCII art flowchart of the basic data flow
- **`mattergen_detailed_flow_nn_ascii.txt`** - **[RECOMMENDED]** Complete ASCII documentation with NN details

### 🔧 Generation Scripts
- **`plot_architecture.py`** - Script to generate basic architecture diagrams
- **`plot_detailed_flow.py`** - Script for detailed flow diagrams
- **`plot_detailed_flow_with_nn_details.py`** - **[MOST COMPREHENSIVE]** Script with full NN specifications

## Key Information Captured

### 🧠 Neural Network Architecture Details
- **GemNet-T specifications**: 15-20M total parameters
- **Layer-by-layer breakdown**: Embedding dimensions, interaction blocks, output heads
- **Parameter counts**: Detailed breakdown by component
- **Memory characteristics**: Tensor shapes and scaling properties

### 🔄 Information Flow
- **Complete pipeline**: From conditioning data to final crystal structures
- **Tensor transformations**: Exact shapes at each step
- **Multi-modal handling**: Positions (continuous), lattice (continuous), atom types (discrete)
- **Diffusion process**: 1000-step denoising with predictor-corrector sampling

### ⚙️ Implementation Details
- **Property conditioning**: How chemical system, space group, etc. are embedded
- **Periodic boundaries**: Handled in graph construction and coordinate transformations
- **Score conversion**: Cartesian forces → fractional position scores
- **Batch processing**: Variable-size crystal graphs with careful indexing

## Architecture Highlights

### 🏗️ GemNet-T Core Architecture
```
Atom Embedding (61K params) → Edge Embedding (590K params) → 
4× Interaction Blocks (8-12M params) → Output Heads (900K params)
```

### 📐 Key Tensor Shapes
- **Atoms**: `[N_total, 3]` where `N_total = sum(num_atoms)`
- **Lattice**: `[batch_size, 3, 3]` 
- **Properties**: `[batch_size, 512]` each
- **Edges**: `[num_edges, 512]` where `num_edges ≈ 50 × N_total`

### 🎯 Property Embedding Networks
- **Chemical System**: `Embedding(119, 512)` - 61K parameters
- **Space Group**: `Embedding(230, 512)` - 118K parameters  
- **Numeric Properties**: `NoiseLevelEncoding(512)` - 0 parameters (sinusoidal)

## Usage Recommendations

### For Quick Understanding
Start with **`mattergen_detailed_flow_nn_ascii.txt`** - it's the most complete text-based overview.

### For Presentations
Use **`mattergen_detailed_flow_nn.png`** - comprehensive visual with all NN details.

### For Implementation Reference
Check the parameter breakdown section in the ASCII documentation for exact layer sizes and memory requirements.

## Regenerating Diagrams

To update or modify the diagrams:

```bash
cd /Users/connie/mattergen-interp
python mattergen/scripts/architecture_plots/plot_detailed_flow_with_nn_details.py
```

All scripts are self-contained and will save outputs to the current directory.

## Technical Notes

### Coordinate Systems
- **Fractional coordinates**: Used throughout diffusion (periodic [0,1]³)
- **Cartesian coordinates**: Used in GemNet force prediction
- **Automatic conversion**: Handled by lattice matrix transformations

### Multi-Modal Diffusion
- **Continuous SDEs**: VPSDE for positions and lattice
- **Discrete diffusion**: D3PM for categorical atom types
- **Joint optimization**: Single score model predicts all modalities

### Scaling Properties
- **Memory**: O(N + E + T) where N=atoms, E=edges, T=triplets
- **Compute**: Dominated by 4 interaction blocks with triplet message passing
- **Batch efficiency**: Variable-size graphs processed efficiently

---

*Generated for MatterGen crystal structure generation model analysis*