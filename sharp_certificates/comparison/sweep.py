"""Certify kappa < 1 for every eta in [lo, hi] by covering with Arb balls (adaptive bisection).
Usage: python sweep.py model eps lo hi cells out.json"""
import sys, json, subprocess, os
model, eps, lo, hi, cells, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proof', 'budget_gain.py')
def run(c, r):
    ball = f'[{c!r} +/- {r!r}]'
    env = dict(os.environ, EPS=eps, ARB_BITS='224')
    t = subprocess.run([sys.executable, P, model, ball, cells, '1'], capture_output=True, text=True, env=env, cwd=os.path.dirname(P)).stdout
    i, j = t.find('{'), t.rfind('}')
    if i < 0: return None
    d = json.loads(t[i:j + 1]); return d
done, todo, failed = [], [(lo, hi, 0)], []
n0 = int(os.environ.get('N0', 20))
w = (hi - lo) / n0; todo = [(lo + k * w, lo + (k + 1) * w, 0) for k in range(n0)]
while todo:
    a, b, depth = todo.pop(0)
    d = run((a + b) / 2, (b - a) / 2 * 1.000001)
    ok = d is not None and d['below_one']
    if ok: done.append(dict(lo=a, hi=b, bound=d['bound'], unit=d['unweighted_bound']))
    elif depth < 4: m = (a + b) / 2; todo = [(a, m, depth + 1), (m, b, depth + 1)] + todo
    else: failed.append(dict(lo=a, hi=b, bound=None if d is None else d['bound']))
    print(model, eps, a, b, ok, None if d is None else d['bound'], flush=True)
json.dump(dict(model=model, eps=eps, lo=lo, hi=hi, cells=cells, certified=done, failed=failed,
               all_certified=not failed), open(out, 'w'), indent=1)
print('ALL CERTIFIED' if not failed else f'FAILED intervals: {len(failed)}')
