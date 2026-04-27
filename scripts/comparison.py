from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['GPT-5-mini', 'DeepSeek-Coder', 'DeepSeek-Chat', 'GPT-4o-mini']
agent = np.array([18.9, 15.9, 13.2, 7.9])
retrieval = np.array([10.6, 9.9, 9.5, 1.9])
diff = agent - retrieval

# Output path
repo_root = Path(__file__).resolve().parents[1]
output_dir = repo_root / "evaluation_plot"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "rq2_strategy_bar.pdf"

# Positions
x = np.arange(len(models))
width = 0.34

# Figure
plt.figure(figsize=(7, 4.5))

bars1 = plt.bar(x - width/2, agent, width, label='Agent')
bars2 = plt.bar(x + width/2, retrieval, width, label='Retrieval')

# Labels
plt.xlabel('Models')
plt.ylabel('Repair Success Rate (Pass@1, %)')
plt.xticks(x, models)
plt.ylim(0, max(agent) + 4)

plt.legend(frameon=False)

# Add improvement labels (clean and subtle)
for i in range(len(models)):
    y = max(agent[i], retrieval[i]) + 0.5
    plt.text(x[i], y, f'+{diff[i]:.1f} pp', ha='center', fontsize=9)

# Layout
plt.tight_layout()
plt.savefig(output_file, bbox_inches='tight')
plt.close()

print(f"Saved to: {output_file}")
