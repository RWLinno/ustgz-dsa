"""
Generate framework diagram for CTIS paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Define colors
color_data = '#E8F4F8'
color_model = '#FFF4E6'
color_output = '#F0FFF4'
color_rag = '#FCE7F3'
color_rl = '#EDE9FE'

# Title
ax.text(7, 7.5, 'CTIS: Connected Transportation Information System Framework',
        ha='center', va='center', fontsize=14, fontweight='bold')

# === Data Layer ===
data_box = FancyBboxPatch((0.5, 5.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='#0EA5E9', facecolor=color_data, linewidth=2)
ax.add_patch(data_box)
ax.text(1.75, 6.4, 'Data Layer', ha='center', va='center', fontweight='bold')
ax.text(1.75, 6.1, 'LaDe Dataset', ha='center', va='center', fontsize=9)
ax.text(1.75, 5.85, '• Pickup/Delivery Points', ha='center', va='center', fontsize=8)
ax.text(1.75, 5.65, '• Temporal Features', ha='center', va='center', fontsize=8)

# === Feature Extraction ===
feature_box = FancyBboxPatch((3.5, 5.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                            edgecolor='#0EA5E9', facecolor=color_data, linewidth=2)
ax.add_patch(feature_box)
ax.text(4.75, 6.4, 'Feature Extraction', ha='center', va='center', fontweight='bold')
ax.text(4.75, 6.1, 'Node Embeddings', ha='center', va='center', fontsize=9)
ax.text(4.75, 5.85, '• Location Features', ha='center', va='center', fontsize=8)
ax.text(4.75, 5.65, '• Context Features', ha='center', va='center', fontsize=8)

# Arrow: Data -> Feature
arrow1 = FancyArrowPatch((3, 6.1), (3.5, 6.1), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#0EA5E9')
ax.add_patch(arrow1)

# === Core Model ===
# Spatio-Temporal Module
st_box = FancyBboxPatch((6.5, 5.3), 2, 1.6, boxstyle="round,pad=0.1",
                       edgecolor='#F97316', facecolor=color_model, linewidth=2)
ax.add_patch(st_box)
ax.text(7.5, 6.7, 'ST-Forecasting', ha='center', va='center', fontweight='bold')
ax.text(7.5, 6.4, 'WaveTS + Graph Conv', ha='center', va='center', fontsize=8)
ax.text(7.5, 6.15, 'Temporal Patterns', ha='center', va='center', fontsize=7)
ax.text(7.5, 5.9, '+', ha='center', va='center', fontsize=9)
ax.text(7.5, 5.65, 'Spatial Dependencies', ha='center', va='center', fontsize=7)

# Arrow: Feature -> ST
arrow2 = FancyArrowPatch((6, 6.1), (6.5, 6.1), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#F97316')
ax.add_patch(arrow2)

# === RAG Module ===
rag_box = FancyBboxPatch((6.5, 3.5), 2, 1.3, boxstyle="round,pad=0.1",
                        edgecolor='#EC4899', facecolor=color_rag, linewidth=2)
ax.add_patch(rag_box)
ax.text(7.5, 4.6, 'RAG Module', ha='center', va='center', fontweight='bold')
ax.text(7.5, 4.3, 'RAST Framework', ha='center', va='center', fontsize=8)
ax.text(7.5, 4.05, 'Pattern Retrieval', ha='center', va='center', fontsize=7)
ax.text(7.5, 3.8, '+ Fusion', ha='center', va='center', fontsize=7)

# Arrow: ST <-> RAG (bidirectional)
arrow3 = FancyArrowPatch((7.3, 5.3), (7.3, 4.8), arrowstyle='<->', 
                        mutation_scale=20, linewidth=1.5, color='#EC4899')
ax.add_patch(arrow3)

# === RL Module ===
rl_box = FancyBboxPatch((9, 5.3), 2, 1.6, boxstyle="round,pad=0.1",
                       edgecolor='#8B5CF6', facecolor=color_rl, linewidth=2)
ax.add_patch(rl_box)
ax.text(10, 6.7, 'RL Optimizer', ha='center', va='center', fontweight='bold')
ax.text(10, 6.4, 'DQN-based', ha='center', va='center', fontsize=8)
ax.text(10, 6.15, 'Route Optimization', ha='center', va='center', fontsize=7)
ax.text(10, 5.9, '+', ha='center', va='center', fontsize=9)
ax.text(10, 5.65, 'Resource Allocation', ha='center', va='center', fontsize=7)

# Arrow: ST -> RL
arrow4 = FancyArrowPatch((8.5, 6.3), (9, 6.3), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#8B5CF6')
ax.add_patch(arrow4)

# === LLM Agent ===
llm_box = FancyBboxPatch((9, 3.5), 2, 1.3, boxstyle="round,pad=0.1",
                        edgecolor='#8B5CF6', facecolor=color_rl, linewidth=2)
ax.add_patch(llm_box)
ax.text(10, 4.6, 'LLM Agent', ha='center', va='center', fontweight='bold')
ax.text(10, 4.3, 'Role-based QA', ha='center', va='center', fontsize=8)
ax.text(10, 4.05, 'Interactive', ha='center', va='center', fontsize=7)
ax.text(10, 3.8, 'Explanations', ha='center', va='center', fontsize=7)

# === Output Layer ===
# Predictions
pred_box = FancyBboxPatch((11.5, 5.7), 2, 0.8, boxstyle="round,pad=0.1",
                         edgecolor='#10B981', facecolor=color_output, linewidth=2)
ax.add_patch(pred_box)
ax.text(12.5, 6.3, 'Predictions', ha='center', va='center', fontweight='bold')
ax.text(12.5, 5.95, 'Demand Forecasting', ha='center', va='center', fontsize=8)

# Optimized Routes
routes_box = FancyBboxPatch((11.5, 4.5), 2, 0.8, boxstyle="round,pad=0.1",
                           edgecolor='#10B981', facecolor=color_output, linewidth=2)
ax.add_patch(routes_box)
ax.text(12.5, 5.1, 'Optimized Routes', ha='center', va='center', fontweight='bold')
ax.text(12.5, 4.75, 'Vehicle Routing', ha='center', va='center', fontsize=8)

# Interactive Interface
interface_box = FancyBboxPatch((11.5, 3.3), 2, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='#10B981', facecolor=color_output, linewidth=2)
ax.add_patch(interface_box)
ax.text(12.5, 3.9, 'Q&A Interface', ha='center', va='center', fontweight='bold')
ax.text(12.5, 3.55, 'User Interaction', ha='center', va='center', fontsize=8)

# Arrows to outputs
arrow5 = FancyArrowPatch((11, 6.1), (11.5, 6.1), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#10B981')
ax.add_patch(arrow5)

arrow6 = FancyArrowPatch((11, 4.9), (11.5, 4.9), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#10B981')
ax.add_patch(arrow6)

arrow7 = FancyArrowPatch((11, 3.7), (11.5, 3.7), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#10B981')
ax.add_patch(arrow7)

# === Member Contributions ===
ax.text(7, 2.5, 'Multi-Member Contributions', ha='center', va='center',
        fontsize=11, fontweight='bold')

contributions = [
    ('Weilin', 'RAST (RAG)', '#EC4899'),
    ('Ziyu', 'WaveTS (TS)', '#F97316'),
    ('Songxin', 'RL Routing', '#8B5CF6'),
    ('Yiming', 'LLM Security', '#8B5CF6')
]

x_start = 2
for i, (name, contrib, color) in enumerate(contributions):
    x_pos = x_start + i * 3
    contrib_box = FancyBboxPatch((x_pos-0.7, 1.5), 1.4, 0.6, boxstyle="round,pad=0.05",
                                edgecolor=color, facecolor='white', linewidth=1.5)
    ax.add_patch(contrib_box)
    ax.text(x_pos, 2, name, ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x_pos, 1.7, contrib, ha='center', va='center', fontsize=7)

# === Bottom Label ===
ax.text(7, 0.8, 'CTIS Framework: Integrating ST-Forecasting, RAG, RL, and LLM for Last-mile Delivery',
        ha='center', va='center', fontsize=10, style='italic', color='#666')

plt.tight_layout()
plt.savefig('framework.pdf', dpi=300, bbox_inches='tight')
plt.savefig('framework.png', dpi=300, bbox_inches='tight')
print("Framework diagram saved!")

