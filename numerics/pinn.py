"""Physics-informed neural network for the mean-field auction equilibrium.

Backward part (PINN): a network V_theta(t,p) in R^4 (budgets h..4h; V(.,0,.)=0 exactly) with the
terminal condition built in, V = (T-t) * N_theta(t,p). The loss is the squared residual of the
HJB partial integro-differential equation
   V_t + 1/2 sigma^2 p^2(1-p)^2 V_pp + lambda [ LSE_eps( r^Q(p,a) + K^a V ) - V ] = 0,
where the nonlocal Bayes-jump operator K^a is evaluated by calling the network at the
posterior beliefs Psi_1(p,a), Psi_0(p,a) and at the post-win budget.

Forward part: an independent Monte Carlo particle system with *true* hidden regimes
(signal d xi = sigma S dt + dB, Wonham filter, Poisson auctions, rival draws, wins under the
true regime), with bids drawn from the PINN's Gibbs policy. The population bid law is the
Rao-Blackwellised average of the policy vectors.

Equilibrium: fictitious-play outer loop  q <- (1-alpha) q + alpha * q_particles(pi_theta[q]).
"""
import json, time, sys
import numpy as np
import torch
torch.set_default_dtype(torch.float32)
from torch.func import vmap, jacfwd, functional_call
torch.set_num_threads(2)
torch.manual_seed(0); np.random.seed(0)

h, sigma, zeta = 0.05, 0.20, 0.20
d = torch.tensor([-0.01, 0.01]); q0 = torch.tensor([0.2, 0.4, 0.4])
T, lam, eps, eta = 1.0, 2.25, 0.05, 0.45
acts = h * torch.arange(3.)
mgrid = h * torch.arange(3.)
Dl = torch.stack([0.060 + 0.03 * mgrid, 0.095 + 0.03 * mgrid])          # Delta[s,m]
wk = torch.zeros(2, 3, 3)
for s in range(2):
    for a in (1, 2):
        wk[s, a] = torch.sigmoid((acts[a] - mgrid - d[s]) / zeta)
NQ = 61; tq = torch.linspace(0, T, NQ)                                    # q-flow knots


def q_at(qk, t):
    """piecewise-linear interpolation of the flow qk[NQ,3] at times t[n]."""
    x = (t / T * (NQ - 1)).clamp(0, NQ - 1); i = x.long().clamp(max=NQ - 2); f = (x - i)[:, None]
    return qk[i] * (1 - f) + qk[i + 1] * f


def market(qt, p):
    """qt[n,3], p[n] -> W, r, psi1, psi0 each [n,3]."""
    Q = (1 - eta) * q0[None] + eta * qt
    ell = torch.einsum('sam,nm->nsa', wk, Q)
    gain = torch.einsum('sam,sm,nm->nsa', wk, Dl, Q) - acts[None, None] * ell
    pp = p[:, None]
    W = (1 - pp) * ell[:, 0] + pp * ell[:, 1]
    r = (1 - pp) * gain[:, 0] + pp * gain[:, 1]
    psi1 = torch.where(W > 0, pp * ell[:, 1] / W.clamp_min(1e-300), pp)
    psi0 = pp * (1 - ell[:, 1]) / (1 - W)
    return W, r, psi1.clamp(0, 1), psi0.clamp(0, 1), ell


class Net(torch.nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        L = [torch.nn.Linear(2, width), torch.nn.Tanh()]
        for _ in range(depth - 1): L += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        L += [torch.nn.Linear(width, 4)]
        self.f = torch.nn.Sequential(*L)

    def forward(self, t, p):
        z = torch.stack([2 * t / T - 1, 2 * p - 1], -1)
        V = (T - t)[:, None] * 0.2 * self.f(z)                 # V(T)=0 hard
        return torch.cat([torch.zeros_like(V[:, :1]), V], 1)    # column b=0 is identically 0


def scores(net, t, p, qt, Vtp=None):
    """x[n, b, a] = r + K^a V  (b=0..4), -inf where a>b."""
    W, r, psi1, psi0, _ = market(qt, p)
    V = net(t, p) if Vtp is None else Vtp
    n = len(t); x = torch.full((n, 5, 3), -torch.inf)
    x[:, :, 0] = V
    for a in (1, 2):
        V1 = net(t, psi1[:, a]); V0 = net(t, psi0[:, a])
        for b in range(a, 5):
            x[:, b, a] = r[:, a] + W[:, a] * V1[:, b - a] + (1 - W[:, a]) * V0[:, b]
    return x


def residual(net, t, p, qt):
    params = dict(net.named_parameters())
    f = lambda tt, pp: functional_call(net, params, (tt[None], pp[None]))[0, 1:]
    Vt = vmap(jacfwd(f, 0))(t, p)                               # [n,4]
    Vpp = vmap(jacfwd(jacfwd(f, 1), 1))(t, p)                   # [n,4]
    V = net(t, p)
    x = scores(net, t, p, qt, V)
    lse = eps * torch.logsumexp(x / eps, -1)                    # [n,5]
    diff = 0.5 * sigma**2 * (p * (1 - p))**2
    R = Vt + diff[:, None] * Vpp + lam * (lse[:, 1:] - V[:, 1:])
    return R


def sample(n):
    t = torch.rand(n) * T
    p = torch.rand(n)
    k = n // 8                                                 # extra points at/near p in {0,1/2,1}
    p[:k] = torch.where(torch.rand(k) < .5, 0., 1.) * 1.0
    p[k:2 * k] = 0.5 + 0.05 * torch.randn(k)
    return t, p.clamp(0, 1)


def train(net, qk, steps, lr=2e-3, n=1024, log=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 20)
    for it in range(steps):
        t, p = sample(n)
        R = residual(net, t, p, q_at(qk, t))
        loss = (R**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log is not None and it % 500 == 0: log.append(float(loss)); print('  step', it, 'loss %.3e' % loss.item(), flush=True)
    return float(loss.detach())


@torch.no_grad()
def policy(net, t, b, p, qt):
    x = scores(net, t, p, qt)                                  # [n,5,3]
    x = x[torch.arange(len(t)), b]
    return torch.softmax(x / eps, -1)


@torch.no_grad()
def particles(net, qk, N=100_000, nsteps=300, seed=1):
    """True-type particle simulation. Returns q flow at knots and Monte Carlo monetary profit."""
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand(N, generator=g) < 0.5).long()
    b = torch.full((N,), 4, dtype=torch.long); p = torch.full((N,), 0.5)
    dt = T / nsteps; profit = torch.zeros(N)
    qs = torch.zeros(nsteps + 1, 3)
    for k in range(nsteps + 1):
        t = torch.full((N,), k * dt)
        qt = q_at(qk, t[:1]).expand(N, 3)
        pi = policy(net, t, b, p, qt)
        qs[k] = pi.mean(0)
        if k == nsteps: break
        # auction epochs (exact Poisson count over [t,t+dt) is ~ Bernoulli for lam*dt small; use Poisson thinning by 1 event)
        hit = torch.rand(N, generator=g) < 1 - np.exp(-lam * dt)
        a = torch.multinomial(pi, 1, generator=g)[:, 0]
        Q = (1 - eta) * q0 + eta * qt[0]
        m = torch.multinomial(Q.expand(N, 3), 1, replacement=True, generator=g)[:, 0]
        win = hit & (a > 0) & (torch.rand(N, generator=g) < wk[s, a, m])
        lose = hit & (a > 0) & ~win
        profit += win * (Dl[s, m] - acts[a])
        _, _, psi1, psi0, _ = market(qt, p)
        pa1 = psi1[torch.arange(N), a]; pa0 = psi0[torch.arange(N), a]
        p = torch.where(win, pa1, torch.where(lose, pa0, p))
        b = torch.where(win, b - a, b)
        # Wonham filter under the true regime
        dB = torch.randn(N, generator=g) * np.sqrt(dt)
        p = (p + sigma**2 * p * (1 - p) * (s - p) * dt + sigma * p * (1 - p) * dB).clamp(0, 1)
    tt = torch.linspace(0, T, nsteps + 1)
    qk_new = torch.stack([torch.from_numpy(np.interp(tq.numpy(), tt.numpy(), qs[:, j].numpy())) for j in range(3)], 1).float()
    se = profit.std() / np.sqrt(N)
    return qk_new, float(profit.mean()), float(se)


if __name__ == '__main__':
    t0 = time.time()
    net = Net()
    qk = torch.full((NQ, 3), 1 / 3)
    hist = []; logl = []
    outer = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for k in range(outer):
        steps = 4000 if k == 0 else 1500
        loss = train(net, qk, steps, lr=2e-3 if k == 0 else 1e-3, log=logl)
        qn, J, se = particles(net, qk, N=100_000, seed=k + 1)
        dq = float((qn - qk).abs().max())
        alpha = 0.5
        qk = (1 - alpha) * qk + alpha * qn
        V0 = float(net(torch.tensor([0.]), torch.tensor([0.5]))[0, 4])
        hist.append(dict(outer=k, loss=loss, dq=dq, J_mc=J, J_se=se, V0=V0, time=time.time() - t0))
        print(hist[-1], flush=True)
        torch.save(dict(net=net.state_dict(), qk=qk, hist=hist, logl=logl), 'ckpt.pt')
    # final polish against the final flow
    loss = train(net, qk, 3000, lr=5e-4, n=2048, log=logl)
    qn, J, se = particles(net, qk, N=400_000, seed=99)
    torch.save(net.state_dict(), 'pinn.pt')
    # residual on a test grid
    tt, pp = torch.meshgrid(torch.linspace(0, T * 0.999, 61), torch.linspace(0, 1, 101), indexing='ij')
    tt, pp = tt.reshape(-1), pp.reshape(-1)
    R = residual(net, tt.clone(), pp.clone(), q_at(qk, tt)).detach()
    res = dict(history=hist, loss_log=logl, final_loss=loss, J_mc=J, J_se=se,
               V0=float(net(torch.tensor([0.]), torch.tensor([0.5]))[0, 4]),
               resid_rms=float(R.pow(2).mean().sqrt()), resid_max=float(R.abs().max()),
               q_final=qk.tolist(), q_particles_last=qn.tolist(), dq_last=float((qn - qk).abs().max()),
               time=time.time() - t0)
    json.dump(res, open('pinn.json', 'w'), indent=1)
    print({k: v for k, v in res.items() if k not in ('history', 'loss_log', 'q_final', 'q_particles_last')})
