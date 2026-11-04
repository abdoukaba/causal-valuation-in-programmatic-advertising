"""Replay the distributed ledgers, or regenerate all enclosures before replaying."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from replay_weighted_integer import bound

ROOT = Path(__file__).resolve().parent
CASES = {
    'lower': ('.195', '.36', 64, 'archived', 'shannon', '.05'),
    'middle': ('.36', '.42', 112, 'archived', 'shannon', '.05'),
    'upper': ('.42', '.45', 192, 'archived', 'shannon', '.05'),
    'point96': ('.45', '.45', 96, 'archived', 'shannon', '.05'),
    'point192': ('.45', '.45', 192, 'archived', 'shannon', '.05'),
    'refit': ('.45', '.45', 160, 'refit', 'shannon', '.05'),
    'kl05': ('.45', '.45', 80, 'archived', 'anchored', '.05'),
    'kl03': ('.45', '.45', 80, 'archived', 'anchored', '.03'),
    'eps08': ('.45', '.45', 80, 'archived', 'shannon', '.08'),
    'eps03': ('.45', '.45', 80, 'archived', 'shannon', '.03'),
    'tail4h': ('.45', '.45', 160, 'tail4h', 'shannon', '.05'),
    'pooled': ('.45', '.45', 160, 'pooled', 'shannon', '.05'),
    'attributed': ('.45', '.45', 160, 'attributed', 'shannon', '.05'),
}

def check(name):
    lo, hi, n, inputs, policy, epsilon = CASES[name]
    folder = ROOT/'results'/name
    if args.regenerate:
        command = [sys.executable, str(ROOT/'conditional_arb.py'), str(folder),
                   '--eta', lo, hi, '--cells', str(n), '--primitives', inputs,
                   '--policy', policy, '--epsilon', epsilon]
        with (ROOT/'results'/(name+'.log')).open('w') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode and 'The sufficient contraction test is inconclusive' not in (ROOT/'results'/(name+'.log')).read_text():
            raise RuntimeError(name+' failed to enclose its coefficients')
    record = json.loads((folder/'gain.json').read_text())
    assert record['cells']==n and record['epsilon']==epsilon and record['policy']==policy
    assert record['primitives']==inputs and record['eta_endpoints']==list(dict.fromkeys([lo,hi]))
    blob = (folder/'gain.ledger.json.gz').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==record['ledger_sha256']
    ledger = json.loads(gzip.decompress(blob))
    numerator = bound(ledger)
    assert numerator==record['integer_aggregation_numerator']
    passing = numerator < record['integer_aggregation_denominator']
    assert passing==record['integer_aggregation_below_one']
    assert passing==(name not in ('eps03','attributed')), name
    input_dir = ROOT/'v5/code/sharp_certificates/comparison/inputs'
    for filename, digest in record['input_sha256'].items():
        assert hashlib.sha256((input_dir/filename).read_bytes()).hexdigest()==digest
    assert ledger['box']==json.loads((folder/'boxes.json').read_text())['boxes'][-1]
    assert ledger['policy']==policy and ledger['epsilon']==epsilon
    print(name, numerator, '/', record['integer_aggregation_denominator'], 'PASS' if passing else 'INCONCLUSIVE', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('cases', nargs='*')
    args = parser.parse_args()
    bands = [(Fraction(CASES[n][0]), Fraction(CASES[n][1])) for n in ('lower','middle','upper')]
    assert bands[0][0]==Fraction(195,1000) and bands[-1][1]==Fraction(45,100)
    assert all(a[1]==b[0] for a,b in zip(bands,bands[1:]))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(check, args.cases or CASES))
