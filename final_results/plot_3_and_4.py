import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 顶会高级配色方案 (Nature / Science 风格)
# ==========================================
color_obj = '#4A7BA7'    # 沉稳的钢蓝色 (Object)
color_intent = '#B0B8B4' # 高级冷灰色 (Coupled baseline)
color_cra = '#D76364'    # 柔和的砖红色 (CRA, hatch)

# 全局字体设置
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11
})

# 自动打标签函数
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='medium')

# =========================================================================
# 共同物理设定：确保生成完全一致的 axes 矩形大小
# ==========================================
# 设置一个统一且合理的 figsize，不依靠 tight_layout 自动裁剪
UNIFIED_FIGSIZE = (6.0, 4.8) 
# 显式设定 subplot 的绝对边距 (0~1 之间)，留够空间给标签和图例
# 这是确保两个 PDF 尺寸和 plot area 完全一致的关键
UNIFIED_SUBPLOTS_ADJUST = {'left': 0.18, 'right': 0.98, 'top': 0.85, 'bottom': 0.15}


# -------------------------------------------------------------------------
# 生成图 (a): 6.4 感知与推理的解耦 (Disentangling Perception and Reasoning)
# -------------------------------------------------------------------------
phases_a = ['Pre-transit', 'Transit\n(30-60%)', 'Transit\n(60-90%)', 'Post-transit']
obj_acc = [33.0, 41.7, 39.0, 30.0]
intent_acc = [12.1, 12.3, 17.0, 25.4]
cra_acc = [60.0, 44.4, 43.8, 43.9]

x_a = np.arange(len(phases_a))
width = 0.25

fig, ax1 = plt.subplots(figsize=UNIFIED_FIGSIZE)
# 强制应用统一边距
fig.subplots_adjust(**UNIFIED_SUBPLOTS_ADJUST)

# 画3组柱状图
rects1_a = ax1.bar(x_a - width, obj_acc, width, label='Object Perception', color=color_obj, edgecolor='black', linewidth=1.1)
rects2_a = ax1.bar(x_a, intent_acc, width, label='Coupled Macro-Intent', color=color_intent, edgecolor='black', linewidth=1.1)
rects3_a = ax1.bar(x_a + width, cra_acc, width, label='Decoupled Reasoning (CRA)', color=color_cra, edgecolor='black', hatch='//', linewidth=1.1)

autolabel(rects1_a, ax1)
autolabel(rects2_a, ax1)
autolabel(rects3_a, ax1)

# 坐标轴美化
ax1.set_ylabel('Accuracy (%)', fontweight='bold', labelpad=8)
ax1.set_xticks(x_a)
ax1.set_xticklabels(phases_a, fontweight='medium')
ax1.set_ylim(0, 75)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', linestyle='--', alpha=0.6, color='gray')

# 图例放在顶部外侧 ( ncol=2 保持两行， ncol=3 可排一行)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)

# 重要：不要使用 tight_layout() 和 bbox_inches='tight'，保留显式设定的白边
plt.savefig('fig_diagnostic_a_final.pdf', dpi=1200) 
plt.close()


# -------------------------------------------------------------------------
# 生成图 (b): 6.5 消融实验与难度分析 (Ablations on Semantic Priors & Difficulty)
# -------------------------------------------------------------------------
categories_b = ['Overall', 'Easy', 'Medium', 'Hard']
ego_raw = [34.1, 34.1, 39.4, 27.7]
ego_last = [41.3, 37.0, 45.9, 38.8]
ego_inst = [47.3, 56.3, 50.2, 37.2]

x_b = np.arange(len(categories_b))

fig, ax2 = plt.subplots(figsize=UNIFIED_FIGSIZE)
# 强制应用统一边距，即使 (b) 的 Y轴标签更长，也会被压缩在 0.18 的左边距内，而不会改变 plot area 大小
fig.subplots_adjust(**UNIFIED_SUBPLOTS_ADJUST)

# 画分组柱状图
# 颜色分配遵循逻辑： Baseline=灰色，Last_frame=蓝色，With_inst=红色(关键突破)
rects1_b = ax2.bar(x_b - width, ego_raw, width, label='Ego_raw (8-frame)', color=color_intent, edgecolor='black', linewidth=1.1)
rects2_b = ax2.bar(x_b, ego_last, width, label='Ego_last_frame', color=color_obj, edgecolor='black', linewidth=1.1)
rects3_b = ax2.bar(x_b + width, ego_inst, width, label='Ego_with_instruction', color=color_cra, edgecolor='black', hatch='//', linewidth=1.1)

autolabel(rects1_b, ax2)
autolabel(rects2_b, ax2)
autolabel(rects3_b, ax2)

# 坐标轴美化 (这里有一个很长的 LaTeX 标签)
ax2.set_ylabel(r'Reasoning Consistency ($\mathrm{SCA}_R$) %', fontweight='bold', labelpad=8)
ax2.set_xticks(x_b)
ax2.set_xticklabels(categories_b, fontweight='medium')
ax2.set_ylim(0, 65)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.6, color='gray')

# 图例放在顶部外侧
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)

# 重要：不要使用 tight_layout() 和 bbox_inches='tight'，保留显式设定的白边
plt.savefig('fig_diagnostic_b_final.pdf', dpi=1200)
plt.close()

print("✅ Fixed size mismatch. Generated: fig_diagnostic_a_final.pdf and fig_diagnostic_b_final.pdf")