import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd 

df = pd.read_csv("./hs/eval_collapse/data/n=21.csv")
eval1 = df['eval1'].to_numpy()
eval2 = df['eval2'].to_numpy()
bias1 = df['bias1'].to_numpy()
bias2 = df['bias2'].to_numpy()


fig, ax = plt.subplots(figsize=(4, 2.75)) 

norm = Normalize(vmin=-0.50, vmax=0.50)

sc = ax.scatter(np.arange(9), eval1, c=bias1, cmap="coolwarm", norm=norm)
sc = ax.scatter(np.arange(9), eval2, c=bias2, cmap="coolwarm", norm=norm)

ax.set_xticks(np.arange(9))
cbar = plt.colorbar(sc, ticks=[-0.5, 0, 0.5])
cbar.ax.set_yticklabels(["100\% CQ", "0\%", "100\% $H$"])
ax.set_yscale("log")
ax.set_xlabel('Truncations')
ax.set_ylabel('$ \lambda_0 $')


fname = "./paper-figures/drafts/cons-quantity-vs-h_v0"
plt.savefig(fname + ".png")
plt.savefig(fname + ".pdf")
