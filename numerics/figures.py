import json
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pinn

C = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False})
fd = np.load('fd_ref.npz'); R = json.load(open('pinn.json'))
net = pinn.Net(); net.load_state_dict(torch.load('pinn.pt')); net.eval()
nt = fd['q'].shape[0]; tfd = np.arange(nt) / nt
qp = np.array(R['q_final']); tq = np.linspace(0, 1, len(qp))
lab = ['bid 0', 'bid h', 'bid 2h']

# Fig 1: equilibrium bid law
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for a in range(3):
    ax.plot(tfd, fd['q'][:, a], color=C[a], label=f'{lab[a]}, finite difference')
    ax.plot(tq, qp[:, a], color=C[a], ls=(0, (4, 2)), lw=1.6, label=f'{lab[a]}, PINN')
ax.set_xlabel('time t'); ax.set_ylabel('population bid share $q^*_t(a)$'); ax.set_ylim(0, .6)
ax.legend(ncol=3, fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, 1.18))
fig.tight_layout(); fig.savefig('fig_q.png', dpi=200)

# Fig 2: value function at t=0 per budget
p = fd['p']; tt = torch.zeros(len(p), dtype=torch.float32); pp = torch.tensor(p, dtype=torch.float32)
with torch.no_grad(): Vn = net(tt, pp).numpy()
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
for b in range(1, 5):
    ax[0].plot(p, fd['V'][0, b], color=C[b - 1], label=f'b = {b}h')
    ax[0].plot(p, Vn[:, b], color=C[b - 1], ls=(0, (4, 2)), lw=1.4)
    ax[1].plot(p, Vn[:, b] - fd['V'][0, b], color=C[b - 1], label=f'b = {b}h')
ax[0].set_xlabel('belief p'); ax[0].set_ylabel('V(0, b, p)'); ax[0].legend(fontsize=7.5)
ax[0].set_title('Value at t = 0 (solid FD, dashed PINN)', fontsize=9, color=INK)
ax[1].set_xlabel('belief p'); ax[1].set_ylabel('PINN − FD'); ax[1].set_title('Pointwise difference', fontsize=9, color=INK)
fig.tight_layout(); fig.savefig('fig_V.png', dpi=200)

# Fig 3: PINN loss
fig, ax = plt.subplots(figsize=(5, 2.8))
ax.semilogy(np.arange(len(R['loss_log'])) * 500, R['loss_log'], color=C[0])
ax.set_xlabel('Adam step (all outer rounds)'); ax.set_ylabel('mean squared HJB residual')
fig.tight_layout(); fig.savefig('fig_loss.png', dpi=200)

# error summary on (t,p,b) grid
tg = np.linspace(0, 1, nt + 1)[:-1]; err = 0; rel = 0
for k in range(0, nt, 8):
    with torch.no_grad(): Vk = net(torch.full((len(p),), tg[k], dtype=torch.float32), pp).numpy()
    err = max(err, np.abs(Vk[:, 1:] - fd['V'][k, 1:].T).max())
qerr = np.abs(np.array([np.interp(tq, tfd, fd['q'][:, a]) for a in range(3)]).T - qp).max()
json.dump(dict(V_sup_err=float(err), V_scale=float(np.abs(fd['V']).max()), q_sup_err=float(qerr)),
          open('compare.json', 'w'), indent=1)
print(open('compare.json').read())
