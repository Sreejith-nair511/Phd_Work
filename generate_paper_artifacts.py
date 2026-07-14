#!/usr/bin/env python
"""Generate publication-ready figures, diagrams, and PDFs for paper artifacts."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

output_dir = Path("Paper_Artifacts/Figures")
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating paper artifacts...")

# ============================================================================
# 1. SPEECH PREPROCESSING PIPELINE DIAGRAM
# ============================================================================
print("1. Generating speech preprocessing pipeline...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'Speech Preprocessing Pipeline', fontsize=16, fontweight='bold',
        ha='center')

# Define box properties
box_width = 2.2
box_height = 0.8
colors = {
    'input': '#E3F2FD',
    'process': '#FFF9C4',
    'output': '#C8E6C9',
    'feature': '#F3E5F5',
}

def draw_box(ax, x, y, text, color, fontsize=9):
    box = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.1", edgecolor='black', 
                         facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.3, mid_y, label, fontsize=8, style='italic')

# Stage 1: Input
draw_box(ax, 5, 10.5, 'Raw Audio File\n(.wav, .mp3)', colors['input'])

# Stage 2: Loading
draw_arrow(ax, 5, 10, 5, 9.2)
draw_box(ax, 5, 8.7, 'load_audio()\n16kHz mono', colors['process'])

# Stage 3: Parallel processing
draw_arrow(ax, 5, 8.2, 2.5, 7.2)
draw_arrow(ax, 5, 8.2, 7.5, 7.2)

# MFCC path
draw_box(ax, 2.5, 6.7, 'extract_mfcc()\n40 coefficients', colors['process'])
draw_arrow(ax, 2.5, 6.2, 2.5, 5.2)
draw_box(ax, 2.5, 4.7, 'Z-score\nNormalization', colors['process'])
draw_arrow(ax, 2.5, 4.2, 2.5, 3.2)
draw_box(ax, 2.5, 2.7, 'MFCC Embedding\n(T_frames, 40)', colors['feature'])

# Prosodic path
draw_box(ax, 7.5, 6.7, 'extract_prosodic_features()', colors['process'])

# Prosodic features
prosodic_y = 5.8
features = ['Speech Rate', 'Pause Duration', 'Response Latency', 'Energy', 'Pitch']
for i, feat in enumerate(features):
    y_pos = prosodic_y - (i * 0.6)
    ax.text(7.5, y_pos, f'  • {feat}', fontsize=8, va='center')

draw_arrow(ax, 7.5, 5.0, 7.5, 3.2)
draw_box(ax, 7.5, 2.7, 'Prosodic Embedding\n(B, 256)', colors['feature'])

# Stage 4: Fusion
draw_arrow(ax, 2.5, 2.2, 4, 1.2)
draw_arrow(ax, 7.5, 2.2, 6, 1.2)
draw_box(ax, 5, 0.7, 'Speech Encoder Fusion\n(B, 256)', colors['output'])

# Add annotations
ax.text(0.2, 10.5, 'File loading & resampling', fontsize=8, style='italic', color='gray')
ax.text(0.2, 8.7, 'Feature extraction', fontsize=8, style='italic', color='gray')
ax.text(0.2, 2.7, 'Final embeddings', fontsize=8, style='italic', color='gray')

# Add legend
legend_y = 11.2
ax.text(0.2, legend_y, 'Legend:', fontsize=9, fontweight='bold')
for i, (label, color) in enumerate(colors.items()):
    y = legend_y - (i+1)*0.35
    rect = plt.Rectangle((0.2, y-0.15), 0.3, 0.3, facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(0.6, y, label.capitalize(), fontsize=8, va='center')

plt.tight_layout()
plt.savefig(output_dir / 'speech_preprocessing_pipeline.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'speech_preprocessing_pipeline.pdf', bbox_inches='tight')
plt.close()
print("✓ speech_preprocessing_pipeline.png/pdf")


# ============================================================================
# 2. MULTIMODAL FUSION STRATEGIES COMPARISON
# ============================================================================
print("2. Generating multimodal fusion strategies...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Multimodal Fusion Strategies', fontsize=16, fontweight='bold', y=0.98)

# Helper function for fusion diagrams
def draw_modality_box(ax, x, y, name, color='#E3F2FD'):
    rect = mpatches.FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold')

modality_colors = {'Speech': '#FFCCCC', 'Text': '#CCFFCC', 'EEG': '#CCCCFF', 'Facial': '#FFFFCC'}

# 1. EARLY FUSION
ax = axes[0, 0]
ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, 4)
ax.axis('off')
ax.set_title('Early Fusion (Concatenation + MLP)', fontsize=12, fontweight='bold')

# Input modalities
y_inputs = 3.5
for i, (mod, color) in enumerate(modality_colors.items()):
    x = 1 + i*1.2
    draw_modality_box(ax, x, y_inputs, mod, color)
    ax.annotate('', xy=(2.5, 2.8), xytext=(x, y_inputs-0.4),
                arrowprops=dict(arrowstyle='->', lw=1.5))

# Concatenation
rect = mpatches.FancyBboxPatch((1.5, 2.3), 2, 0.4,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor='#FFF9C4', linewidth=2)
ax.add_patch(rect)
ax.text(2.5, 2.5, 'Concatenate', ha='center', va='center', fontsize=8, fontweight='bold')

# MLP
ax.annotate('', xy=(2.5, 1.8), xytext=(2.5, 2.3),
            arrowprops=dict(arrowstyle='->', lw=1.5))
rect = mpatches.FancyBboxPatch((1.7, 0.8), 1.6, 0.8,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor='#FFE0B2', linewidth=2)
ax.add_patch(rect)
ax.text(2.5, 1.2, 'MLP\n(256)', ha='center', va='center', fontsize=8, fontweight='bold')

# Output
ax.annotate('', xy=(2.5, 0.3), xytext=(2.5, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
draw_modality_box(ax, 2.5, -0.1, 'Fused\n(256)', '#C8E6C9')

ax.text(2.5, -0.6, 'Complexity: O(4D²)', ha='center', fontsize=8, style='italic')

# 2. LATE FUSION
ax = axes[0, 1]
ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, 4)
ax.axis('off')
ax.set_title('Late Fusion (Per-Head + Weighted Avg)', fontsize=12, fontweight='bold')

# Input modalities
y_inputs = 3.5
for i, (mod, color) in enumerate(modality_colors.items()):
    x = 1 + i*1.2
    draw_modality_box(ax, x, y_inputs, mod, color)
    # Per-modality head
    ax.annotate('', xy=(x, 2.5), xytext=(x, y_inputs-0.4),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    rect = mpatches.FancyBboxPatch((x-0.35, 2.05), 0.7, 0.4,
                                   boxstyle="round,pad=0.02",
                                   edgecolor='black', facecolor='#B2DFDB', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, 2.25, f'H{i+1}', ha='center', va='center', fontsize=7, fontweight='bold')

# Weighted average
ax.annotate('', xy=(2.5, 1.2), xytext=(1.3, 2.05),
            arrowprops=dict(arrowstyle='->', lw=1.2, color='gray'))
ax.annotate('', xy=(2.5, 1.2), xytext=(2.5, 2.05),
            arrowprops=dict(arrowstyle='->', lw=1.2, color='gray'))
ax.annotate('', xy=(2.5, 1.2), xytext=(3.7, 2.05),
            arrowprops=dict(arrowstyle='->', lw=1.2, color='gray'))

rect = mpatches.FancyBboxPatch((1.7, 0.8), 1.6, 0.4,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor='#FFE0B2', linewidth=2)
ax.add_patch(rect)
ax.text(2.5, 1.0, 'Weighted Avg', ha='center', va='center', fontsize=8, fontweight='bold')

# Output
ax.annotate('', xy=(2.5, 0.3), xytext=(2.5, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
draw_modality_box(ax, 2.5, -0.1, 'Fused\n(256)', '#C8E6C9')

ax.text(2.5, -0.6, 'Complexity: O(4D²)', ha='center', fontsize=8, style='italic')

# 3. ATTENTION FUSION
ax = axes[1, 0]
ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, 4)
ax.axis('off')
ax.set_title('Attention Fusion (Transformer)', fontsize=12, fontweight='bold')

# Input modalities
y_inputs = 3.5
for i, (mod, color) in enumerate(modality_colors.items()):
    x = 1 + i*1.2
    draw_modality_box(ax, x, y_inputs, mod, color)
    ax.annotate('', xy=(2.5, 2.7), xytext=(x, y_inputs-0.4),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='gray', alpha=0.6))

# [FUSE] token
draw_modality_box(ax, 2.5, 3.5, '[FUSE]', '#F3E5F5')
ax.annotate('', xy=(2.5, 2.7), xytext=(2.5, 3.1),
            arrowprops=dict(arrowstyle='->', lw=1.5))

# Transformer
rect = mpatches.FancyBboxPatch((1.3, 1.5), 2.4, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#E1BEE7', linewidth=2)
ax.add_patch(rect)
ax.text(2.5, 2.1, 'Transformer', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(2.5, 1.8, '(Multi-Head Attention)', ha='center', va='center', fontsize=7)

# Output [FUSE]
ax.annotate('', xy=(2.5, 0.8), xytext=(2.5, 1.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
draw_modality_box(ax, 2.5, 0.3, 'Fused\n(256)', '#C8E6C9')

ax.text(2.5, -0.4, 'Complexity: O(M²D)', ha='center', fontsize=8, style='italic')

# 4. CROSS-MODAL FUSION
ax = axes[1, 1]
ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, 4)
ax.axis('off')
ax.set_title('Cross-Modal Fusion (Pairwise Attention)', fontsize=12, fontweight='bold')

# Input modalities
y_inputs = 3.5
for i, (mod, color) in enumerate(modality_colors.items()):
    x = 1 + i*1.2
    draw_modality_box(ax, x, y_inputs, mod, color)

# Pairwise interactions
ax.text(2.5, 2.8, 'Pairwise Cross-Attention', ha='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFCCCC', alpha=0.5))

# Show some arrows representing cross-attention
positions = [1, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8]
for i, x in enumerate(positions):
    ax.plot([x, x], [3.1, 2.6], 'k-', alpha=0.3, linewidth=0.8)

# Aggregation
rect = mpatches.FancyBboxPatch((1.3, 1.5), 2.4, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#E1BEE7', linewidth=2)
ax.add_patch(rect)
ax.text(2.5, 1.95, 'Self-Attention Pooling', ha='center', va='center', fontsize=8, fontweight='bold')

# Output
ax.annotate('', xy=(2.5, 0.8), xytext=(2.5, 1.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
draw_modality_box(ax, 2.5, 0.3, 'Fused\n(256)', '#C8E6C9')

ax.text(2.5, -0.4, 'Complexity: O(M³D)', ha='center', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig(output_dir / 'fusion_strategies_comparison.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'fusion_strategies_comparison.pdf', bbox_inches='tight')
plt.close()
print("✓ fusion_strategies_comparison.png/pdf")


# ============================================================================
# 3. SYSTEM ARCHITECTURE DIAGRAM
# ============================================================================
print("3. Generating system architecture...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Multimodal Depression Detection Architecture', 
        fontsize=14, fontweight='bold', ha='center')

# Layer 1: Input Modalities
y_layer1 = 8.5
modalities = [
    ('Speech', '#FFCCCC', 1.5),
    ('Text', '#CCFFCC', 4.5),
    ('EEG', '#CCCCFF', 7.5),
    ('Facial', '#FFFFCC', 10.5),
]

for name, color, x in modalities:
    rect = mpatches.FancyBboxPatch((x-0.7, y_layer1-0.4), 1.4, 0.8,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y_layer1, name, ha='center', va='center', fontsize=10, fontweight='bold')

# Layer 2: Encoders
y_layer2 = 7
encoders = [
    ('Speech\nEncoder', 1.5),
    ('Text\nEncoder', 4.5),
    ('EEG\nEncoder', 7.5),
    ('Facial\nEncoder', 10.5),
]

for name, x in encoders:
    # Arrows from modalities
    for mx, _ in [(1.5, 0), (4.5, 0), (7.5, 0), (10.5, 0)]:
        if x == mx:
            ax.annotate('', xy=(x, y_layer2+0.4), xytext=(x, y_layer1-0.4),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    rect = mpatches.FancyBboxPatch((x-0.7, y_layer2-0.4), 1.4, 0.8,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor='#FFF9C4', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y_layer2, name, ha='center', va='center', fontsize=9, fontweight='bold')

# Layer 3: Embeddings
y_layer3 = 5.5
ax.text(7, y_layer3+0.7, 'Fixed 256-d Embeddings', ha='center', fontsize=9, style='italic')

for x in [1.5, 4.5, 7.5, 10.5]:
    ax.annotate('', xy=(x, y_layer3+0.4), xytext=(x, y_layer2-0.4),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    circle = mpatches.Circle((x, y_layer3), 0.35, edgecolor='black', 
                            facecolor='#E1BEE7', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y_layer3, '256d', ha='center', va='center', fontsize=7, fontweight='bold')

# Layer 4: Fusion (all arrows converge)
y_layer4 = 4
for x in [1.5, 4.5, 7.5, 10.5]:
    ax.annotate('', xy=(7, y_layer4+0.5), xytext=(x, y_layer3-0.35),
               arrowprops=dict(arrowstyle='->', lw=1.2, color='gray', alpha=0.6))

fusion_types = [
    'Early', 'Late',
    'Attention', 'Cross-Modal'
]
ax.text(7, y_layer4+1.2, 'Fusion Layer (4 strategies)', ha='center', fontsize=9, 
        style='italic', fontweight='bold')

rect = mpatches.FancyBboxPatch((5.5, y_layer4-0.4), 3, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#E1BEE7', linewidth=2)
ax.add_patch(rect)
ax.text(7, y_layer4, 'Multimodal Fusion\n(256d)', ha='center', va='center', 
        fontsize=9, fontweight='bold')

# Layer 5: Classification
y_layer5 = 2.5
ax.annotate('', xy=(7, y_layer5+0.5), xytext=(7, y_layer4-0.4),
           arrowprops=dict(arrowstyle='->', lw=2, color='black'))

rect = mpatches.FancyBboxPatch((5.5, y_layer5-0.4), 3, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#FFE0B2', linewidth=2)
ax.add_patch(rect)
ax.text(7, y_layer5, 'MLP Classifier\n(Binary/Regression)', ha='center', va='center',
        fontsize=9, fontweight='bold')

# Layer 6: Output
y_layer6 = 1
ax.annotate('', xy=(6, y_layer6+0.5), xytext=(6.5, y_layer5-0.4),
           arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.annotate('', xy=(8, y_layer6+0.5), xytext=(7.5, y_layer5-0.4),
           arrowprops=dict(arrowstyle='->', lw=2, color='green'))

rect = mpatches.FancyBboxPatch((5.3, y_layer6-0.4), 1.3, 0.8,
                               boxstyle="round,pad=0.05",
                               edgecolor='green', facecolor='#C8E6C9', linewidth=2)
ax.add_patch(rect)
ax.text(6, y_layer6, 'Binary\nLabel', ha='center', va='center', fontsize=8, fontweight='bold')

rect = mpatches.FancyBboxPatch((7.4, y_layer6-0.4), 1.3, 0.8,
                               boxstyle="round,pad=0.05",
                               edgecolor='green', facecolor='#C8E6C9', linewidth=2)
ax.add_patch(rect)
ax.text(8, y_layer6, 'PHQ-8\nScore', ha='center', va='center', fontsize=8, fontweight='bold')

# Add layer labels
ax.text(-0.5, y_layer1, 'Input', fontsize=8, style='italic', color='gray', fontweight='bold')
ax.text(-0.5, y_layer2, 'Encoding', fontsize=8, style='italic', color='gray', fontweight='bold')
ax.text(-0.5, y_layer3, 'Embed', fontsize=8, style='italic', color='gray', fontweight='bold')
ax.text(-0.5, y_layer4, 'Fusion', fontsize=8, style='italic', color='gray', fontweight='bold')
ax.text(-0.5, y_layer5, 'Classify', fontsize=8, style='italic', color='gray', fontweight='bold')
ax.text(-0.5, y_layer6, 'Output', fontsize=8, style='italic', color='gray', fontweight='bold')

# Add example config on the right
config_text = """
Config: speech_text.yaml
─────────────────────
Modalities:
  ✓ Speech (Wav2Vec2)
  ✓ Text (RoBERTa)
  ✗ EEG
  ✗ Facial

Fusion: Attention
Loss: Label Smoothing
Batch Size: 16
Epochs: 50
"""
ax.text(12.5, 5, config_text, fontsize=8, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8),
        verticalalignment='center')

plt.tight_layout()
plt.savefig(output_dir / 'system_architecture.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'system_architecture.pdf', bbox_inches='tight')
plt.close()
print("✓ system_architecture.png/pdf")


# ============================================================================
# 4. SPEECH ENCODER DETAILED ARCHITECTURE
# ============================================================================
print("4. Generating speech encoder architecture...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

fig.suptitle('Speech Encoder Architecture Details', fontsize=14, fontweight='bold')

# Panel 1: MFCC Path
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('MFCC Path', fontsize=11, fontweight='bold')

y = 9
steps_mfcc = [
    ('Raw Audio', '(1, T)', '#FFCCCC'),
    ('MFCC Extraction', '(T_frames, 40)', '#FFF9C4'),
    ('Z-score Norm', '(T_frames, 40)', '#FFF9C4'),
    ('BiLSTM (2L, 512)', '(B, 512)', '#E1BEE7'),
    ('Linear + LayerNorm', '(B, 256)', '#C8E6C9'),
]

for i, (step, shape, color) in enumerate(steps_mfcc):
    y_pos = 9 - i*1.8
    rect = mpatches.FancyBboxPatch((1, y_pos-0.4), 8, 0.8,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='black', facecolor=color, linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(2, y_pos, step, ha='left', va='center', fontsize=9, fontweight='bold')
    ax1.text(8.5, y_pos, shape, ha='right', va='center', fontsize=8, style='italic', color='gray')
    
    if i < len(steps_mfcc) - 1:
        ax1.annotate('', xy=(5, y_pos-0.5), xytext=(5, y_pos-1.3),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Panel 2: Wav2Vec2 Path
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Wav2Vec2 Path', fontsize=11, fontweight='bold')

y = 9
steps_w2v = [
    ('Raw Waveform', '(B, T)', '#FFCCCC'),
    ('Wav2Vec2 Model', '(B, T\', 768)', '#B3E5FC'),
    ('Mean Pooling', '(B, 768)', '#B3E5FC'),
    ('Linear + LayerNorm', '(B, 256)', '#C8E6C9'),
]

for i, (step, shape, color) in enumerate(steps_w2v):
    y_pos = 9 - i*2
    rect = mpatches.FancyBboxPatch((1, y_pos-0.4), 8, 0.8,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='black', facecolor=color, linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(2, y_pos, step, ha='left', va='center', fontsize=9, fontweight='bold')
    ax2.text(8.5, y_pos, shape, ha='right', va='center', fontsize=8, style='italic', color='gray')
    
    if i < len(steps_w2v) - 1:
        ax2.annotate('', xy=(5, y_pos-0.5), xytext=(5, y_pos-1.6),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Panel 3: Prosodic Features
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Prosodic Features Extraction', fontsize=11, fontweight='bold')

prosodic_features = [
    ('Speech Rate', 'voiced_frames / total_frames', 'Rate ↓ in depression'),
    ('Pause Duration', 'mean(silence_segments)', 'Longer pauses ↑'),
    ('Response Latency', 'first_voiced / n_frames', 'Delayed onset ↑'),
    ('Energy', 'mean(RMS[voiced])', 'Lower energy ↓'),
    ('Pitch (F0)', 'ZCR * sr / (2*frame_len)', 'Flat pitch ↓ variation'),
]

y = 9.5
for feat, formula, depression in prosodic_features:
    y_pos = y - len(prosodic_features) * 0.35 + prosodic_features.index((feat, formula, depression)) * 1.8
    
    # Feature name
    ax3.text(1, y_pos+0.4, feat, fontsize=8, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.7))
    
    # Formula
    ax3.text(1, y_pos, f'Formula: {formula}', fontsize=7, style='italic', family='monospace')
    
    # Depression relevance
    ax3.text(1, y_pos-0.4, f'Depression: {depression}', fontsize=7, color='#D32F2F')

# Panel 4: Fusion
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Acoustic + Prosodic Fusion', fontsize=11, fontweight='bold')

# Acoustic embedding
rect1 = mpatches.FancyBboxPatch((1, 7.5), 3.5, 0.8,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='#E1BEE7', linewidth=1.5)
ax4.add_patch(rect1)
ax4.text(2.75, 7.9, 'Acoustic Emb', ha='center', va='center', fontsize=8, fontweight='bold')
ax4.text(2.75, 7.6, '(B, 256)', ha='center', va='center', fontsize=7, style='italic')

# Prosodic embedding
rect2 = mpatches.FancyBboxPatch((5.5, 7.5), 3.5, 0.8,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='#E1BEE7', linewidth=1.5)
ax4.add_patch(rect2)
ax4.text(7.25, 7.9, 'Prosodic Emb', ha='center', va='center', fontsize=8, fontweight='bold')
ax4.text(7.25, 7.6, '(B, 256)', ha='center', va='center', fontsize=7, style='italic')

# Concatenation
ax4.annotate('', xy=(4, 6.5), xytext=(2.75, 7.5),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax4.annotate('', xy=(6, 6.5), xytext=(7.25, 7.5),
            arrowprops=dict(arrowstyle='->', lw=1.5))

rect_concat = mpatches.FancyBboxPatch((3, 5.9), 4, 0.8,
                                      boxstyle="round,pad=0.05",
                                      edgecolor='black', facecolor='#FFE0B2', linewidth=1.5)
ax4.add_patch(rect_concat)
ax4.text(5, 6.3, 'Concatenate [256||256]', ha='center', va='center', fontsize=8, fontweight='bold')
ax4.text(5, 6.0, '(B, 512)', ha='center', va='center', fontsize=7, style='italic')

# MLP Fusion
ax4.annotate('', xy=(5, 5.0), xytext=(5, 5.9),
            arrowprops=dict(arrowstyle='->', lw=1.5))

rect_mlp = mpatches.FancyBboxPatch((3, 3.9), 4, 1,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='black', facecolor='#FFE0B2', linewidth=1.5)
ax4.add_patch(rect_mlp)
ax4.text(5, 4.5, 'MLP Fusion Layer', ha='center', va='center', fontsize=8, fontweight='bold')
ax4.text(5, 4.2, 'Linear(512→256)', ha='center', va='center', fontsize=7)
ax4.text(5, 4.0, 'LayerNorm + GELU + Dropout', ha='center', va='center', fontsize=7)

# Output
ax4.annotate('', xy=(5, 3.0), xytext=(5, 3.9),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))

rect_out = mpatches.FancyBboxPatch((3.5, 2.2), 3, 0.8,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='green', facecolor='#C8E6C9', linewidth=2)
ax4.add_patch(rect_out)
ax4.text(5, 2.6, 'Speech Embedding', ha='center', va='center', fontsize=8, fontweight='bold')
ax4.text(5, 2.3, '(B, 256)', ha='center', va='center', fontsize=7, style='italic')

# Equation
ax4.text(5, 1.3, 'Equation:', fontsize=8, fontweight='bold', ha='center')
ax4.text(5, 0.8, 'out = MLP(concat([acoustic, prosodic]))', fontsize=7, 
        family='monospace', ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

plt.savefig(output_dir / 'speech_encoder_architecture.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'speech_encoder_architecture.pdf', bbox_inches='tight')
plt.close()
print("✓ speech_encoder_architecture.png/pdf")


# ============================================================================
# 5. IMPLEMENTATION STATUS HEATMAP
# ============================================================================
print("5. Generating implementation status heatmap...")

components = [
    'Audio Loading', 'MFCC Extraction', 'Prosodic Features',
    'Speech Encoder', 'Text Encoder', 'EEG Encoder', 'Facial Encoder',
    'Early Fusion', 'Late Fusion', 'Attention Fusion', 'Cross-Modal Fusion',
    'Binary Classification', 'Regression (MSE)', 'Trainer', 'Evaluator',
    'Classification Metrics', 'Regression Metrics', 'Visualization',
    'Utterance Segmentation', 'Participant Aggregation'
]

# Status: 2 = Full, 1 = Partial, 0 = Not Impl
status = [
    2, 2, 2,  # Speech
    2, 2, 2, 2,  # Encoders
    2, 2, 2, 2,  # Fusion
    2, 1, 2, 2,  # Training
    2, 0, 2,  # Evaluation
    0, 0  # Advanced
]

categories = ['Speech', '', '', 'Encoders', '', '', '', 'Fusion', '', '', '', 
              'Training', '', '', '', 'Evaluation', '', '', 'Advanced', '']

fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap data
data = np.array(status).reshape(-1, 1)

im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)

# Set y-axis labels
ax.set_yticks(range(len(components)))
ax.set_yticklabels(components, fontsize=10)
ax.set_xticks([0])
ax.set_xticklabels(['Status'], fontsize=10)

# Add text annotations
for i, (comp, stat) in enumerate(zip(components, status)):
    if stat == 2:
        text = '✓ IMPLEMENTED'
        color = 'white'
    elif stat == 1:
        text = '⚠ PARTIAL'
        color = 'black'
    else:
        text = '✗ NOT IMPL'
        color = 'white'
    
    ax.text(0, i, text, ha='center', va='center', fontsize=9, 
           fontweight='bold', color=color)

# Add category separators
category_indices = [2.5, 6.5, 10.5, 14.5, 17.5, 18.5]
for idx in category_indices:
    ax.axhline(idx, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Title and labels
ax.set_title('Implementation Status by Component', fontsize=14, fontweight='bold', pad=20)

# Add legend
legend_y = len(components) + 0.5
ax.text(-0.3, legend_y-0.5, 'Legend:', fontsize=10, fontweight='bold', transform=ax.transData)
ax.text(-0.3, legend_y-1.5, '✓ Fully Implemented', fontsize=9, color='#2E7D32',
       transform=ax.transData)
ax.text(-0.3, legend_y-2.5, '⚠ Partial/Scaffold', fontsize=9, color='#F57F17',
       transform=ax.transData)
ax.text(-0.3, legend_y-3.5, '✗ Not Implemented', fontsize=9, color='#C62828',
       transform=ax.transData)

# Statistics box
impl_count = sum(1 for s in status if s == 2)
partial_count = sum(1 for s in status if s == 1)
not_impl_count = sum(1 for s in status if s == 0)

stats_text = f"""Implementation Statistics
─────────────────────
Fully Implemented: {impl_count}/{len(status)} ({100*impl_count/len(status):.1f}%)
Partial: {partial_count}/{len(status)} ({100*partial_count/len(status):.1f}%)
Not Implemented: {not_impl_count}/{len(status)} ({100*not_impl_count/len(status):.1f}%)

Overall: {100*impl_count/len(status) + 50*partial_count/len(status):.0f}% Complete
"""

ax.text(0.5, len(components)+4, stats_text, fontsize=9, family='monospace',
       bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9),
       verticalalignment='top', horizontalalignment='center',
       transform=ax.transData)

plt.tight_layout()
plt.savefig(output_dir / 'implementation_status_heatmap.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'implementation_status_heatmap.pdf', bbox_inches='tight')
plt.close()
print("✓ implementation_status_heatmap.png/pdf")

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
print("\nGenerated files:")
print(f"  - {output_dir / 'speech_preprocessing_pipeline.png'}")
print(f"  - {output_dir / 'speech_preprocessing_pipeline.pdf'}")
print(f"  - {output_dir / 'fusion_strategies_comparison.png'}")
print(f"  - {output_dir / 'fusion_strategies_comparison.pdf'}")
print(f"  - {output_dir / 'system_architecture.png'}")
print(f"  - {output_dir / 'system_architecture.pdf'}")
print(f"  - {output_dir / 'speech_encoder_architecture.png'}")
print(f"  - {output_dir / 'speech_encoder_architecture.pdf'}")
print(f"  - {output_dir / 'implementation_status_heatmap.png'}")
print(f"  - {output_dir / 'implementation_status_heatmap.pdf'}")
print("\nTotal: 10 files (5 diagrams × PNG + PDF)")
