"""Summarize the extension sweeps (logs_ext/*.log, produced by sweep.py with N0 balls of width ~0.002,
bisected up to four times) into the contiguous certified range starting at the previous eta_bar.
Each log line: model eps lo hi certified bound. Only lines with certified=True are used."""
import json
def cover(log, lo):
    iv = []
    for l in open(log):
        p = l.split(None, 5)
        if len(p) >= 6 and p[4] == 'True': iv.append((float(p[2]), float(p[3]), p[5].strip()))
    iv.sort(); x = lo; used = []
    for a, b, s in iv:
        if a <= x + 1e-12 and b > x: used.append(dict(lo=a, hi=b, bound=s)); x = b
    return x, used
runs = [('benchmark', '.01', 'logs_ext/bench_eps01_ext.log', 0.73125, 'point 0.845: [1.0021931744415 +/- 4.70e-14]'),
        ('auction', '.05', 'logs_ext/ipinyou_ext.log', 0.4625, 'point 0.545: [1.0085807088252 +/- 1.15e-14]'),
        ('archived', '.05', 'logs_ext/composite_ext.log', 0.1675, 'point 0.20: [1.0261821693817 +/- 5.56e-15]')]
out = []
for m, e, log, lo, beyond in runs:
    hi, used = cover(log, lo)
    out.append(dict(model=m, eps=e, previous_hi=lo, certified_hi=hi, extension_balls=len(used), point_beyond=beyond, certified_intervals=used))
    print(m, e, 'from', lo, 'to', round(hi, 6), 'with', len(used), 'balls')
json.dump(out, open('results_ext/extension_summary.json', 'w'), indent=1)
