import matplotlib.pyplot as plt
import os

def plot_temporal_dilution_trend(save_path):
    # ================= 提取的实验数据 =================
    # 横坐标：帧数 N
    frames = ['1\n(Last Frame)', '2', '4', '8', '16']
    
    # 纵坐标：Qwen 和 InternVL 的 SCA_R 准确率 (%)
    qwen_scar = [41.3, 32.1, 38.3, 36.5, 30.0]
    internvl_scar = [36.1, 44.8, 45.5, 44.8, 49.6]

    # ================= ECCV 全局学术字体设置 =================
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    # 创建画布 (黄金比例 4:3)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # ================= 绘制两条核心趋势线 =================
    # 1. Qwen 曲线：使用显眼的砖红色实线 + 圆形标记，突出“衰退”
    ax.plot(frames, qwen_scar, marker='o', markersize=8, linestyle='-', 
            linewidth=2.5, color='#c0392b', label='Qwen-VL', zorder=3)
    
    # 2. InternVL 曲线：使用稳重的深蓝色虚线 + 方块标记，突出“饱和/震荡”
    ax.plot(frames, internvl_scar, marker='s', markersize=7, linestyle='--', 
            linewidth=2.5, color='#2980b9', label='InternVL', zorder=3)

    # ================= 图表修饰 =================
    # 设置标签
    ax.set_xlabel('Number of Sampled Frames (N)', labelpad=10)
    ax.set_ylabel(r'$\mathrm{SCA}_\mathrm{R}$ Accuracy (%)', labelpad=10)
    
    # 设置 Y 轴范围，留出一点上下边距让图形更舒展
    ax.set_ylim(25, 55)
    
    # 开启精致的虚线网格，方便审稿人对齐数值
    ax.grid(True, linestyle=':', alpha=0.7, zorder=0)
    
    # 添加图例
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='black')

    # ================= 保存输出 =================
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight')
    print(f"\n🎉 完美趋势折线图已保存至: {save_path}")

if __name__ == "__main__":
    SAVE_PATH = "final_results/temporal_dilution_trend.pdf"
    plot_temporal_dilution_trend(SAVE_PATH)