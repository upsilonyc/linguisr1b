import matplotlib.pyplot as plt
import pandas as pd

# raw data
raw = [
    # p2-1
    ('p2-1', 'DeepSeek-V3', 'F', 0),
    ('p2-1', 'DeepSeek-V3', 'M', 3),
    ('p2-1', 'DeepSeek-V3', 'U', 1),
    ('p2-1', 'DeepSeek-R1', 'F', 0),
    ('p2-1', 'DeepSeek-R1', 'M', 0),
    ('p2-1', 'DeepSeek-R1', 'U', 4),
    # p2-2
    ('p2-2', 'DeepSeek-V3', 'F', 0),
    ('p2-2', 'DeepSeek-V3', 'M', 6),
    ('p2-2', 'DeepSeek-V3', 'U', 0),
    ('p2-2', 'DeepSeek-R1', 'F', 0),
    ('p2-2', 'DeepSeek-R1', 'M', 4),
    ('p2-2', 'DeepSeek-R1', 'U', 2),
]

df = pd.DataFrame(raw, columns=['prompt', 'model', 'gender', 'count'])

# Figure1: R1: p2-1 vs p2-2 --> Decline of U vs rise of V (overlay line plot)
# 2.1 取 R1 子集
r1 = df[df['model'] == 'DeepSeek-R1']

# 2.2 把 gender 转成列，方便画线
r1_pivot = r1.pivot(index='prompt', columns='gender', values='count').fillna(0)

# 2.3 画图
fig1, ax1 = plt.subplots(figsize=(4,3))
for g in ['U', 'M']:          # U=decline, M=rise
    ax1.plot(r1_pivot.index, r1_pivot[g], marker='o', label=f'Gender={g}')
ax1.set_title('R1: p2-1 → p2-2\nDecline of U vs Rise of M')
ax1.set_ylabel('Count')
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure2: p2-1: R1 vs V3, UFM --> effect of alignment on bias recognition
# 3.1 取 p2-1 子集
p21 = df[df['prompt'] == 'p2-1']

# 3.2 把 model 转成列，方便并排 bar
p21_pivot = p21.pivot(index='gender', columns='model', values='count').fillna(0)

# 3.3 画图
fig2, ax2 = plt.subplots(figsize=(4,3))
p21_pivot.plot(kind='bar', ax=ax2, rot=0)
ax2.set_title('p2-1: R1 vs V3\nEffect of Alignment on Bias Recognition')
ax2.set_ylabel('Count')
ax2.legend(title='Model')
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
