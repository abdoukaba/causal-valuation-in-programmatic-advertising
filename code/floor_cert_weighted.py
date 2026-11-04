"""Floor-censored iPinYou background law: Arb bounds with exactly the settings of Table ipinyou
(cert_weighted_all.py, section 6): weight w(t)=1+alpha t, alpha in ALPHAS, theta in THETAS, 2000 cells,
top point at 3h=450. Writes ../results/floor_cert_weighted.json."""
import json, cert_general as cg
from cert_weighted_all_grid import ALPHAS, THETAS
C = json.load(open('../results/floor_censored_calib.json'))['inputs']['q0_censored']
mg = [0, 150 / 3000, 300 / 3000, 450 / 3000]
qs = [str(round(x, 6)) for x in C[:-1]]; qs.append(str(round(1 - sum(round(x, 6) for x in C[:-1]), 6)))
out = {}
for zeta in ('0.1', '0.2'):
    for eta in ('0.3', '0.45'):
        b = None
        for al in ALPHAS:
            for th in THETAS:
                r = cg.kappa([round(x, 6) for x in mg], qs, zeta, 1, '2.25', '0.05', eta, th, nt=2000, alpha=al); k = float(r['kappa'].upper())
                if b is None or k < b['bound']: b = dict(bound=k, theta=th, alpha=al, certified=bool(r['certified']))
        out[f'zeta={zeta},eta={eta}'] = b; print(zeta, eta, b, flush=True)
json.dump(dict(q0_censored=qs, settings='as Table ipinyou: w=1+alpha t, 2000 cells, top=450', bounds=out), open('../results/floor_cert_weighted.json', 'w'), indent=1)
# fitted (censored) score scale, with the settings of the fitted row of Table ipinyou (cert_weighted_fitted.py)
zf = str(round(json.load(open('../results/floor_censored_calib.json'))['zeta_censored']['zeta'], 4))
for eta in ('0.3', '0.45'):
    best = None
    for al in ('0', '1', '2'):
        for th in ('4', '8', '16', '32'):
            k = cg.kappa([round(x, 6) for x in mg], qs, zf, 1, '2.25', '0.05', eta, th, nt=2000, alpha=al)['kappa'].upper()
            if best is None or k < best[0]: best = (k, th, al)
    out[f'zeta={zf},eta={eta}'] = dict(bound=best[0].str(3, radius=False), theta=best[1], alpha=best[2]); print(zf, eta, out[f'zeta={zf},eta={eta}'], flush=True)
json.dump(dict(q0_censored=qs, settings='as Table ipinyou: w=1+alpha t, 2000 cells, top=450; fitted row as cert_weighted_fitted.py', bounds=out), open('../results/floor_cert_weighted.json', 'w'), indent=1)
