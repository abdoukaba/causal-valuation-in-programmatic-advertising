"""Arb certificates for the time-profiled weighted criterion of Theorem 1, w(t)=1+alpha*t, for every table of the
paper.  For each case the bound is minimized over a grid of (theta, alpha); alpha=0 is the scalar criterion.
Run from a directory containing criteo_regimes.json, calib_season1.json, ctr_bins.json (for the calibrated rows).
Writes cert_weighted_all.json.  All bounds are Arb upper ends (80-bit balls, outward rounding)."""
import json, sys
import cert_interval as ci, cert_general as cg, cert_calib as cc
THETAS = ('4', '8', '10', '12', '14', '16', '18', '20', '25', '30')
ALPHAS = ('0', '0.5', '1', '1.5', '2', '3')
out = {}
def best_ci(eps, eta, nt=4000, thetas=THETAS, alphas=ALPHAS, T=1, lam='2.25'):
    b = None
    for al in alphas:
        for th in thetas:
            r = ci.sharp_kappa(T, lam, eps, eta, th, nt=nt, alpha=al); k = float(r['kappa'].upper())
            if b is None or k < b['bound']: b = dict(bound=k, theta=th, alpha=al, certified=bool(r['certified']), ball=r['kappa'].str(12))
    return b
def scalar_ci(eps, eta, nt=4000): return best_ci(eps, eta, nt, alphas=('0',))
# 1. eta table at the benchmark
out['eta'] = {}
for eta in ('0.30', '0.40', '0.45', '0.47', '0.48', '0.485', '0.50', '0.55', '0.58', '0.60', '0.61', '0.62'):
    out['eta'][eta] = dict(weighted=best_ci('0.05', eta), scalar=scalar_ci('0.05', eta)); print('eta', eta, out['eta'][eta], flush=True)
# 2. threshold for the weighted criterion over the grid (step 0.0025)
lo, hi = 0.60, 0.62
for _ in range(4):
    mid = round((lo + hi) / 2, 4); r = best_ci('0.05', str(mid), nt=4000)
    if r['certified']: lo = mid
    else: hi = mid
out['threshold'] = dict(largest_certified=lo, smallest_failed=hi); print('threshold', out['threshold'], flush=True)
# 3. entropy grid at eta=0.45 and 0.5
out['eps'] = {}
for eta in ('0.45', '0.5'):
    for eps in ('1', '0.5', '0.3', '0.2', '0.15', '0.10', '0.08', '0.07', '0.06', '0.05', '0.04', '0.03', '0.02', '0.01'):
        out['eps'][f'{eta},{eps}'] = best_ci(eps, eta, nt=2000, thetas=THETAS + ('40', '60')); print('eps', eta, eps, out['eps'][f'{eta},{eps}'], flush=True)
# 4. robustness (Table robust)
out['robust'] = {}
for name, zeta, q0 in (('baseline', '0.2', ('0.2', '0.4', '0.4')), ('zeta=0.1', '0.1', ('0.2', '0.4', '0.4')), ('zeta=0.05', '0.05', ('0.2', '0.4', '0.4')),
                       ('q0 low-bid', '0.2', ('0.6', '0.3', '0.1')), ('q0 high-bid', '0.2', ('0.1', '0.3', '0.6'))):
    b = None
    for al in ALPHAS:
        for th in THETAS:
            r = cg.kappa([0, .05, .1], list(q0), zeta, 1, '2.25', '0.05', '0.45', th, nt=2000, alpha=al); k = float(r['kappa'].upper())
            if b is None or k < b['bound']: b = dict(bound=k, theta=th, alpha=al, certified=bool(r['certified']))
    out['robust'][name] = b; print('robust', name, b, flush=True)
# 5. anchored references (Table anchored)
D = [[str(0.06 + 0.03 * m) for m in (0, .05, .1)], [str(0.095 + 0.03 * m) for m in (0, .05, .1)]]
out['anchored'] = {}
for rm in (None, '0.2', '0.1', '0.05', '0.02'):
    for lab, (T, lam, eps, eta) in (('long', (1, '2.25', '0.05', '0.45')), ('short', ('0.2', '0.1', '0.5', '0.5'))):
        r = cc.best([0, .05, .1], ['0.2', '0.4', '0.4'], D, ['-0.01', '0.01'], '0.2', T, lam, eps, eta, thetas=('1',) + THETAS, rho_min=rm, nt=4000, alphas=ALPHAS)
        out['anchored'][f'{lab},{rm}'] = r; print('anchored', lab, rm, r, flush=True)
# 6. iPinYou background law (Table ipinyou)
S = json.load(open('calib_season1.json')); out['ipinyou'] = {}
for top in (450, 600):
    res = S[f'top={top}']; q0 = res['q0']; mg = [0, 150 / 3000, 300 / 3000, top / 3000]
    qs = [str(round(x, 6)) for x in q0[:-1]]; qs.append(str(round(1 - sum(round(x, 6) for x in q0[:-1]), 6)))
    for zeta in ('0.1', '0.2'):
        for eta in ('0.3', '0.45'):
            b = None
            for al in ALPHAS:
                for th in THETAS:
                    r = cg.kappa([round(x, 6) for x in mg], qs, zeta, 1, '2.25', '0.05', eta, th, nt=2000, alpha=al); k = float(r['kappa'].upper())
                    if b is None or k < b['bound']: b = dict(bound=k, theta=th, alpha=al, certified=bool(r['certified']))
            out['ipinyou'][f'top={top},zeta={zeta},eta={eta}'] = b; print('ipinyou', top, zeta, eta, b, flush=True)
# 7. calibrated market at zeta=0.2 (Section 9.2)
import calib_full as cf
P = cf.primitives(); Ds = [[repr(float(x)) for x in row] for row in P['Delta']]; ds = [repr(float(x)) for x in P['d']]; qs = [repr(float(x)) for x in P['q0']]
out['calibrated'] = {}
for rm in (None, '0.2', '0.05'):
    try: out['calibrated'][f'zeta0.2,rho_min={rm}'] = cc.best(cf.MG.tolist(), qs, Ds, ds, '0.2', 1, '2.25', '0.05', '0.45', thetas=('1',) + THETAS, rho_min=rm, alphas=ALPHAS)
    except ValueError as e: out['calibrated'][f'zeta0.2,rho_min={rm}'] = str(e)
    print('calibrated', rm, out['calibrated'][f'zeta0.2,rho_min={rm}'], flush=True)
json.dump(out, open('cert_weighted_all.json', 'w'), indent=1)
