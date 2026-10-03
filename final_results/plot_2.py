import matplotlib.pyplot as plt
import numpy as np

# Qwen2-VL-7B-Instruct在 ego_ann 下的真实数据
phases = ['Pre-transit', 'Transit\n(30%-60%)', 'Transit\n(60%-90%)', 'Post-transit']
obj_acc = [33.0, 41.7, 39.0, 30.0]      # 底层感知：看清物体的能力
intent_acc = [12.1, 12.3, 17.0, 25.4]   # 端到端结果：综合意图准确率
cra_acc = [60.0, 44.4, 43.8, 43.9]      # 纯推理能力：条件预期准确率

x = np.arange(len(phases))
width = 0.25  # 柱宽

plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})
fig, ax = plt.subplots(figsize=(8, 4.5))

# ==========================================
# 顶会高级配色方案 (Nature / Science 风格)
# ==========================================
color_obj = '#4A7BA7'    # 沉稳的钢蓝色 (Steel Blue)
color_intent = '#B0B8B4' # 高级冷灰色 (Cool Gray) 
color_cra = '#D76364'    # 柔和的砖红色 (Muted Crimson)

# 画3组柱状图，增加 linewidth 提升质感
rects1 = ax.bar(x - width, obj_acc, width, label='Object Perception ($Acc_{obj}$)', color=color_obj, edgecolor='black', linewidth=1.2)
rects2 = ax.bar(x, intent_acc, width, label='Coupled Macro-Intent', color=color_intent, edgecolor='black', linewidth=1.2)
rects3 = ax.bar(x + width, cra_acc, width, label='Decoupled Reasoning (CRA)', color=color_cra, edgecolor='black', hatch='//', linewidth=1.2)

# 自动打标签函数
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='medium')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 坐标轴美化
ax.set_ylabel('Accuracy (%)', fontweight='bold', labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(phases, fontweight='medium')
ax.set_ylim(0, 75)

# 隐藏顶部和右侧的边框 (顶会最爱的高级极简风)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例和网格
ax.legend(loc='upper left', frameon=True, fontsize=11, ncol=1, edgecolor='black')
ax.grid(axis='y', linestyle='--', alpha=0.6, color='gray')

plt.tight_layout()
plt.savefig('fig_disentangle_premium.pdf', dpi=1200, bbox_inches='tight')
print("✅ Successfully generated fig_disentangle_premium.pdf")