import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as mticker 

def hs_coeff(r, n):
	return np.pi ** 2 / (n * np.sin(np. pi * r / n))**2 
plt.rc('text.latex', preamble='\\usepackage{amsmath}\n\\usepackage{physics}')

n = 21
df = pd.read_csv(f'./hs/data/decim_n={n}.csv')

n_ops = df['n_ops'].unique()
max_ops = np.max(n_ops)

evals = []
for i in n_ops: 
	evals.append(np.min(df.loc[df['n_ops'] == i]['evals'].to_numpy()))
evals = np.array(evals)

# Get HS coeffs (from highest number of operators in basis)
# hs_anal = np.flip(df.loc[df['n_ops'] == max(n_ops)]['anal'].to_numpy())
r = np.linspace(1, max_ops, 100)
hs_anal = hs_coeff(r, n)

fig, ax = plt.subplots(figsize=(4, 5), nrows=3)

# NOTE: normalizing eigenvalues
# ax[0].set_title('HS Coefficients over Lowest Eigenvalue ')


ax[0].plot(r, hs_anal ** 2, label=r'$J_{\textrm{HS}}^2$', color='gray', alpha=0.85, linestyle='dashed')
ax[0].scatter(n_ops, evals / np.linalg.norm(evals), color='black', label='lowest $\lambda$', alpha=0.65)
ax[1].plot(r, hs_anal ** 2, label=r'$J_{\textrm{HS}}^2$', color='gray', alpha=0.85, linestyle='dashed')
ax[1].scatter(n_ops, evals / np.linalg.norm(evals), color='black', label='lowest $\lambda$', alpha=0.65)

ax[0].set_ylabel('$ \lambda_0 $')

ax[0].set_xticks(n_ops)

# ax[0].set_xlabel('$N$ operators')
ax[0].legend(frameon=True)

ax[1].set_ylabel('$ \lambda_0 $')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
# ax[1].set_xlabel('$N$ operators')
ax[1].set_xticks(n_ops)

# Do not use scientific notation for x axis
ax[1].xaxis.set_major_formatter(mticker.ScalarFormatter())

ax[2].plot(n_ops, evals / hs_coeff(n_ops, n)**2, color='black', alpha=0.65)
ax[2].set_ylabel('$Q(R)$')
ax[2].set_xticks(n_ops)
ax[2].set_xlabel('$N$ operators')
ax[2].set_ylim(1e-14, 1e-1)


for a, label in zip(ax, ['(a)', '(b)', '(c)']):
	a.text(0.02, 0.22, label, transform=a.transAxes,
            fontsize=14, va='top')

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax[2].set_yscale('log')
# plt.savefig(os.path.join(outdir, 'hs_coeffs_lowest_eval_log.pdf'))
# plt.savefig(os.path.join(outdir, 'hs_coeffs_lowest_eval_log.png'), dpi=300)
plt.tight_layout()
# plt.show()

fname = "./paper-figures/drafts/hs-truncation-n=21_v0"

plt.savefig(fname + ".pdf")
plt.savefig(fname + ".png")