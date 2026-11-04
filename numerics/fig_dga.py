import json, numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '.'); import pinn
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42})
fd = np.load('fd_ref.npz'); R = json.load(open('pinn.json'))
net = pinn.Net(); net.load_state_dict(torch.load('pinn.pt')); net.eval()
nt = fd['q'].shape[0]; tfd = np.arange(nt) / nt
qp = np.array(R['q_final']); tq = np.linspace(0, 1, len(qp))
C = ['black', '#1f5fbf', '#e07b00']; lab = ['no bid', 'bid h', 'bid 2h']
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
for a in range(3):
    ax[0].plot(tfd, fd['q'][:, a], color=C[a], lw=1.2, label=f'{lab[a]} (FD)')
    ax[0].plot(tq[::3], qp[::3, a], ls='none', marker='o', ms=2.6, mfc='white', mec=C[a], mew=.8, label=f'{lab[a]} (PINN)')
ax[0].set_ylim(0, .75); ax[0].set_xlabel('time'); ax[0].set_ylabel('population bid share')
ax[0].legend(ncol=2, fontsize=6.5, loc='upper right'); ax[0].set_title('(a)', loc='left', fontsize=8, fontweight='bold')
p = fd['p']; pp = torch.tensor(p, dtype=torch.float32)
with torch.no_grad(): Vn = net(torch.zeros(len(p)), pp).numpy()
ls = ['-', '--', '-.', ':']
for b in range(1, 5):
    ax[1].plot(p, 1e5 * (Vn[:, b] - fd['V'][0, b]), color='black', ls=ls[b - 1], lw=1, label=f'budget {b}h')
ax[1].axhline(0, color='0.7', lw=.6)
ax[1].set_xlabel('belief $p$'); ax[1].set_ylabel(r'$V_\theta(0,b,p)-V_h(0,b,p)$  ($\times10^{-5}$)')
ax[1].legend(fontsize=6.5, ncol=2, loc='lower center'); ax[1].set_title('(b)', loc='left', fontsize=8, fontweight='bold')
fig.tight_layout(); fig.savefig('Fig3.pdf')
