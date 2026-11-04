"""Finite-difference solver for the logit mean-field equilibrium of the partially observed
first-price auction (budget b, belief p).

Backward (Bellman/HJB, mild form discretised by splitting):
    dV/dt + 1/2 sigma^2 p^2 (1-p)^2 V_pp + lambda [ LSE_eps( r^Q(p,a) + K^a V )_{a in A_b} - V ] = 0,  V(T)=0
Forward (Kolmogorov, exact discrete adjoint of the backward transition):
    m_{k+1} = A^{-T} J_k^T m_k,  q_k(a) = sum_{b,p} m_k(b,p) pi_k(b,p)(a)
Equilibrium: q = Phi(q), solved by damped fixed-point iteration.

The population law is tracked under the observation measure: given the observed history the
regime is Bernoulli(p), so the law of (b,p) alone determines the population bid law.
"""
import numpy as np
from scipy.special import expit
from scipy.linalg import solve_banded


class Params:
    def __init__(self, **kw):
        self.h = 0.05
        self.nb = 5                       # budgets 0..4h
        self.sigma = 0.20
        self.d = np.array([-0.01, 0.01])
        self.zeta = 0.20
        self.q0 = np.array([0.2, 0.4, 0.4])
        self.T, self.lam, self.eps, self.eta = 1.0, 2.25, 0.05, 0.45
        self.p0 = 0.5
        self.b0 = 4
        self.__dict__.update(kw)
        self.acts = self.h * np.arange(3)          # {0,h,2h}
        self.mgrid = self.h * np.arange(len(self.q0))

    def Delta(self, s, m):
        return (0.060 if s == 0 else 0.095) + 0.03 * m


class FDSolver:
    def __init__(self, P, n_p=101, nt=600):
        self.P = P
        self.n_p, self.nt = n_p, nt
        self.p = np.linspace(0, 1, n_p)
        self.dp = 1.0 / (n_p - 1)
        self.dt = P.T / nt
        # win kernel w[s, a, m]
        self.w = np.zeros((2, 3, len(P.q0)))
        self.D = np.zeros((2, len(P.q0)))
        for s in range(2):
            self.D[s] = [P.Delta(s, m) for m in P.mgrid]
            for ia in range(1, 3):
                self.w[s, ia] = expit((P.acts[ia] - P.mgrid - P.d[s]) / P.zeta)
        # implicit diffusion matrix A = I - dt * 1/2 sigma^2 p^2(1-p)^2 d_pp  (banded form)
        c = 0.5 * P.sigma**2 * self.p**2 * (1 - self.p)**2 * self.dt / self.dp**2
        ab = np.zeros((3, n_p))
        ab[1] = 1 + 2 * c
        ab[0, 1:] = -c[:-1]      # super-diagonal: A[i,i+1] = -c_i
        ab[2, :-1] = -c[1:]      # sub-diagonal:   A[i+1,i] = -c_{i+1}
        ab[1, 0] = ab[1, -1] = 1.0   # c=0 at the endpoints anyway
        self.ab = ab
        # transpose band
        abT = np.zeros_like(ab)
        abT[1] = ab[1]
        abT[0, 1:] = ab[2, :-1]
        abT[2, :-1] = ab[0, 1:]
        self.abT = abT

    # ----- one-step market primitives given rival law Q -----
    def market(self, qt):
        P = self.P
        Q = (1 - P.eta) * P.q0.copy()
        Q[:3] += P.eta * qt
        ell = np.einsum('sam,m->sa', self.w, Q)                       # ell[s,a]
        gain = np.einsum('sam,sm,m->sa', self.w, self.D, Q) - P.acts[None] * ell
        p = self.p[:, None]
        W = (1 - p) * ell[0] + p * ell[1]                              # [np, a]
        r = (1 - p) * gain[0] + p * gain[1]
        with np.errstate(divide='ignore', invalid='ignore'):
            psi1 = np.where(W > 0, p * ell[1] / W, p)
            psi0 = np.where(1 - W > 0, p * (1 - ell[1]) / (1 - W), p)
        psi1[:, 0] = psi0[:, 0] = self.p
        W[:, 0] = 0; r[:, 0] = 0
        return W, r, np.clip(psi1, 0, 1), np.clip(psi0, 0, 1)

    def _interp(self, x):
        i = np.minimum((x / self.dp).astype(int), self.n_p - 2)
        return i, x / self.dp - i

    def K(self, U, W, psi1, psi0):
        """K^a U for all (b, p, a). U: [nb, np]. Returns [nb, np, 3], -inf where infeasible."""
        nb = self.P.nb
        out = np.full((nb, self.n_p, 3), -np.inf)
        i0, f0 = self._interp(psi0); i1, f1 = self._interp(psi1)
        for b in range(nb):
            out[b, :, 0] = U[b]
            for ia in (1, 2):
                if ia > b: continue
                lose = U[b, i0[:, ia]] * (1 - f0[:, ia]) + U[b, i0[:, ia] + 1] * f0[:, ia]
                win = U[b - ia, i1[:, ia]] * (1 - f1[:, ia]) + U[b - ia, i1[:, ia] + 1] * f1[:, ia]
                out[b, :, ia] = W[:, ia] * win + (1 - W[:, ia]) * lose
        return out

    def KT(self, mu, W, psi1, psi0):
        """Adjoint of the policy-weighted jump: mu[b,p,a] = mass choosing a at (b,p)."""
        nb = self.P.nb
        out = np.zeros((nb, self.n_p))
        i0, f0 = self._interp(psi0); i1, f1 = self._interp(psi1)
        for b in range(nb):
            out[b] += mu[b, :, 0]
            for ia in (1, 2):
                if ia > b: continue
                ml = mu[b, :, ia] * (1 - W[:, ia]); mw = mu[b, :, ia] * W[:, ia]
                np.add.at(out[b], i0[:, ia], ml * (1 - f0[:, ia])); np.add.at(out[b], i0[:, ia] + 1, ml * f0[:, ia])
                np.add.at(out[b - ia], i1[:, ia], mw * (1 - f1[:, ia])); np.add.at(out[b - ia], i1[:, ia] + 1, mw * f1[:, ia])
        return out

    def diffuse(self, V):
        return solve_banded((1, 1), self.ab, V.T).T

    def diffuseT(self, m):
        return solve_banded((1, 1), self.abT, m.T).T

    def scores(self, U, qt):
        W, r, psi1, psi0 = self.market(qt)
        KU = self.K(U, W, psi1, psi0)
        x = KU + r[None]
        return x, (W, r, psi1, psi0)

    # ----- backward: entropy-regularised best response -----
    def backward(self, q, eps=None, hard=False):
        P = self.P; eps = P.eps if eps is None else eps
        V = np.zeros((P.nb, self.n_p)); pol = np.zeros((self.nt, P.nb, self.n_p, 3))
        Vs = np.zeros((self.nt + 1, P.nb, self.n_p))
        lam_dt = P.lam * self.dt
        for k in range(self.nt - 1, -1, -1):
            U = self.diffuse(V)
            x, _ = self.scores(U, q[k])
            if hard:
                best = x.max(-1, keepdims=True)
                pk = (x >= best - 1e-14).astype(float); pk /= pk.sum(-1, keepdims=True)
                lse = best[..., 0]
            else:
                mx = x.max(-1, keepdims=True)
                z = np.exp((x - mx) / eps)
                lse = mx[..., 0] + eps * np.log(z.sum(-1))
                pk = z / z.sum(-1, keepdims=True)
            V = U + lam_dt * (lse - U)
            pol[k] = pk; Vs[k] = V
        return Vs, pol

    def evaluate(self, q, pol):
        """Monetary value of a given Markov policy against rival flow q."""
        P = self.P; V = np.zeros((P.nb, self.n_p)); lam_dt = P.lam * self.dt
        for k in range(self.nt - 1, -1, -1):
            U = self.diffuse(V)
            x, _ = self.scores(U, q[k])
            x = np.where(np.isfinite(x), x, 0.0)
            V = U + lam_dt * ((pol[k] * x).sum(-1) - U)
        return V

    def init_mass(self):
        m = np.zeros((self.P.nb, self.n_p))
        j = np.searchsorted(self.p, self.P.p0)
        if abs(self.p[j] - self.P.p0) < 1e-12:
            m[self.P.b0, j] = 1.0
        else:  # linear split
            f = (self.P.p0 - self.p[j - 1]) / self.dp
            m[self.P.b0, j - 1], m[self.P.b0, j] = 1 - f, f
        return m

    def forward(self, q, pol):
        P = self.P; m = self.init_mass(); lam_dt = P.lam * self.dt
        qnew = np.zeros((self.nt, 3)); ms = [m]
        for k in range(self.nt):
            mu = m[..., None] * pol[k]                    # mass by action
            qnew[k] = mu.sum((0, 1))
            W, r, psi1, psi0 = self.market(q[k])
            jumped = self.KT(mu, W, psi1, psi0)
            m = (1 - lam_dt) * m + lam_dt * jumped
            m = self.diffuseT(m)
            ms.append(m)
        return qnew, np.array(ms)

    def Phi(self, q):
        _, pol = self.backward(q)
        return self.forward(q, pol)[0]

    def solve(self, q_init=None, damping=0.5, tol=1e-11, maxit=2000, verbose=False):
        q = np.full((self.nt, 3), 1 / 3) if q_init is None else q_init.copy()
        hist = []
        for it in range(maxit):
            qn = self.Phi(q)
            err = np.abs(qn - q).max()
            hist.append(err)
            if verbose and it % 10 == 0: print(it, err)
            q = (1 - damping) * q + damping * qn
            if err < tol: break
        return q, hist


def report(S, q):
    P = S.P
    Vs, pol = S.backward(q)
    i0 = np.argmin(abs(S.p - P.p0))
    Jeq = S.evaluate(q, pol)[P.b0, i0]
    Vbr, polbr = S.backward(q, hard=True)
    Jbr = Vbr[0, P.b0, i0]
    return dict(J_eq=Jeq, J_br=Jbr, ratio=Jbr / Jeq, V0=Vs[0, P.b0, i0],
                max_dev=np.abs(q - 1 / 3).max())


if __name__ == '__main__':
    import time
    for n_p, nt in [(21, 60), (101, 600)]:
        P = Params()
        S = FDSolver(P, n_p, nt)
        t0 = time.time(); q, h = S.solve()
        print(n_p, nt, 'iters', len(h), 'time %.1fs' % (time.time() - t0), report(S, q))
