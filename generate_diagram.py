import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(8, 7.2), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10.5)
ax.axis('off')

c_dark = '#0F172A'
c_blue = '#1E3A8A'
c_sky = '#0284C7'
c_green = '#059669'
c_red = '#DC2626'
c_box_bg = '#F8FAFC'

# 1. Corpus node
ax.text(5, 10.1, 'Indexed Corpus Chunks (e.g., SEC 10-Ks)', ha='center', va='center',
        fontsize=9.5, fontweight='bold', color=c_dark,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#EFF6FF', edgecolor='#93C5FD', lw=1.2))

ax.annotate('', xy=(5, 9.4), xytext=(5, 9.8),
            arrowprops=dict(arrowstyle='->', lw=1.4, color='#64748B'))

# 2. Embedding node
ax.text(5, 9.1, 'Embedding Extraction (BGE / MiniLM: d-dimensions)', ha='center', va='center',
        fontsize=9, fontweight='bold', color=c_dark,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F1F5F9', edgecolor='#CBD5E1', lw=1.0))

ax.annotate('', xy=(5, 8.45), xytext=(5, 8.8),
            arrowprops=dict(arrowstyle='->', lw=1.4, color='#64748B'))

# 3. Big Box 1: Offline Density Estimation
rect1 = patches.FancyBboxPatch((0.8, 6.8), 8.4, 1.45, boxstyle='round,pad=0.2',
                               facecolor=c_box_bg, edgecolor=c_blue, lw=1.5)
ax.add_patch(rect1)
ax.text(1.1, 7.95, '1. Offline Density Estimation (EM Algorithm)', fontsize=10, fontweight='bold', color=c_blue)
ax.text(1.3, 7.55, '• Fit Mixture of Factor Analyzers (K clusters, low-rank r << d)', fontsize=8.8, color=c_dark)
ax.text(1.3, 7.15, '• Fit EVT Weibull parameters (scale lambda, shape kappa) on validation tail distances', fontsize=8.8, color=c_dark)

# Offline/Online Divider
ax.axhline(6.25, xmin=0.04, xmax=0.96, color='#94A3B8', linestyle='--', lw=1.2)
ax.text(5, 6.25, '   Real-Time Online Inference   ', ha='center', va='center', fontsize=8, fontweight='bold', color='#64748B', backgroundcolor='white')

# 4. Incoming query
ax.annotate('', xy=(5, 5.65), xytext=(5, 6.05),
            arrowprops=dict(arrowstyle='->', lw=1.4, color='#64748B'))

ax.text(5, 5.35, 'Incoming User Query q (Embedding vector)', ha='center', va='center',
        fontsize=9.5, fontweight='bold', color=c_dark,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF3C7', edgecolor='#F59E0B', lw=1.2))

ax.annotate('', xy=(5, 4.65), xytext=(5, 5.05),
            arrowprops=dict(arrowstyle='->', lw=1.4, color='#64748B'))

# 5. Big Box 2: Pre-Retrieval Guardrail
rect2 = patches.FancyBboxPatch((0.8, 3.05), 8.4, 1.4, boxstyle='round,pad=0.2',
                               facecolor=c_box_bg, edgecolor=c_sky, lw=1.5)
ax.add_patch(rect2)
ax.text(1.1, 4.15, '2. Real-Time Pre-Retrieval Statistical Guardrail (< 5ms)', fontsize=10, fontweight='bold', color=c_sky)
ax.text(1.3, 3.75, '• Woodbury low-rank Mahalanobis distance inversion per cluster', fontsize=8.6, color=c_dark)
ax.text(1.3, 3.35, '• Extreme Value Theory (EVT) survival p-value computation: S(q)', fontsize=8.6, color=c_dark)

# Branch arrows
ax.annotate('', xy=(2.7, 2.15), xytext=(3.5, 2.85),
            arrowprops=dict(arrowstyle='->', lw=1.5, color=c_red))
ax.text(2.6, 2.45, 'S(q) < alpha (OOD)', ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_red)

ax.annotate('', xy=(7.3, 2.15), xytext=(6.5, 2.85),
            arrowprops=dict(arrowstyle='->', lw=1.5, color=c_green))
ax.text(7.4, 2.45, 'S(q) >= alpha (In-Dist)', ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_green)

# 6. Reject Box (Left)
rect_rej = patches.FancyBboxPatch((0.6, 0.40), 4.1, 1.45, boxstyle='round,pad=0.2',
                                 facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=1.3)
ax.add_patch(rect_rej)
ax.text(2.65, 1.50, '[Bypass LLM Generation]', ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_red)
ax.text(2.65, 1.10, 'Return Grounded Uncertainty Flag:', ha='center', va='center', fontsize=7.8, color=c_dark)
ax.text(2.65, 0.82, '"Query outside knowledge base"', ha='center', va='center', fontsize=7.8, fontweight='bold', color=c_dark)
ax.text(2.65, 0.55, 'Eliminates ungrounded hallucinations', ha='center', va='center', fontsize=7.2, fontstyle='italic', color='#7F1D1D')

# 7. Pass Box (Right)
rect_pass = patches.FancyBboxPatch((5.3, 0.40), 4.1, 1.45, boxstyle='round,pad=0.2',
                                  facecolor='#F0FDF4', edgecolor='#86EFAC', lw=1.3)
ax.add_patch(rect_pass)
ax.text(7.35, 1.50, '[Pass to Vector Database]', ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_green)
ax.text(7.35, 1.10, 'Retrieve Top-K Context Chunks &', ha='center', va='center', fontsize=7.8, color=c_dark)
ax.text(7.35, 0.82, 'Forward to LLM Generator', ha='center', va='center', fontsize=7.8, fontweight='bold', color=c_dark)
ax.text(7.35, 0.55, 'Executes grounded generation safely', ha='center', va='center', fontsize=7.2, fontstyle='italic', color='#14532D')

plt.tight_layout()
plt.savefig('workflow_diagram.png', dpi=300, bbox_inches='tight')
print('Regenerated workflow_diagram.png clean!')
