import matplotlib.pyplot as plt

# Qwen2-VL-7B-Instruct SCA_R Data
# 1. Temporal Phases Data
phases = ['Pre-transit', 'Transit\n(30%-60%)', 'Transit\n(60%-90%)', 'Post-transit']
ego_phases = [23.3, 22.4, 24.7, 34.1]
third_phases = [24.6, 21.1, 27.1, 41.5]
oracle_phases = [23.5, 23.8, 27.4, 41.7]

# 2. Accumulative Progress Data
acc_x = ['30%', '60%', '90%']
ego_acc = [21.5, 16.4, 13.4]
third_acc = [28.0, 18.8, 14.1]
oracle_acc = [25.8, 23.3, 23.8]

plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})

# ================= 生成图 (a) =================
fig, ax1 = plt.subplots(figsize=(5, 4.5))
ax1.plot(phases, oracle_phases, marker='D', markersize=8, linewidth=2, color='#d62728', label='Oracle')
ax1.plot(phases, third_phases, marker='s', markersize=8, linewidth=2, linestyle='--', color='#1f77b4', label='Third')
ax1.plot(phases, ego_phases, marker='o', markersize=8, linewidth=2, linestyle='-.', color='#ff7f0e', label='Ego_raw')
ax1.set_ylabel(r"Reasoning Consistency ($\mathrm{SCA}_R$) %")
ax1.set_ylim(10, 45)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right', fontsize=11)
plt.tight_layout()
# 严格按照 ECCV 要求设置 1200 dpi
plt.savefig('fig_temporal_a.pdf', dpi=1200, bbox_inches='tight') 
plt.close()

# ================= 生成图 (b) =================
fig, ax2 = plt.subplots(figsize=(5, 4.5))
ax2.plot(acc_x, oracle_acc, marker='D', markersize=8, linewidth=2, color='#d62728', label='Oracle')
ax2.plot(acc_x, third_acc, marker='s', markersize=8, linewidth=2, linestyle='--', color='#1f77b4', label='Third')
ax2.plot(acc_x, ego_acc, marker='o', markersize=8, linewidth=2, linestyle='-.', color='#ff7f0e', label='Ego_raw')
ax2.set_xlabel(r"Accumulative video length ($\rho$)")
ax2.set_ylim(10, 45)
ax2.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# 严格按照 ECCV 要求设置 1200 dpi
plt.savefig('fig_temporal_b.pdf', dpi=1200, bbox_inches='tight')
plt.close()

print("✅ Successfully generated fig_temporal_a.pdf and fig_temporal_b.pdf at 1200 dpi.")