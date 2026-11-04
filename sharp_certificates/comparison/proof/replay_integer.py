"""Detached verification of the last gain bound. Only Python's integer arithmetic is used."""
import gzip
import json
import sys


def ceil_ratio(a, b):
    return -(-a // b)


def bound(ledger):
    n, scale, cells = ledger["n"], ledger["scale"], ledger["cells"]
    assert len(cells) == n and ledger["weight"] == "identically one"

    def expected_by_budget(values, prob):
        # The dominating process spending k or more than three units leaves at least 4-k.
        return ceil_ratio(sum(p * max(values[4-min(4,k):]) for k,p in enumerate(prob)), scale)

    def arrival_integral(x):
        return ceil_ratio(9*x, 4*n)  # lambda=9/4, T=1

    values = []
    for i, cell in enumerate(cells):
        policy = []
        for b in range(5):
            future = arrival_integral(sum(c["source_span"][b] for c in cells[i:]))
            inst = cell["direct"][b]
            deriv = cell["gibbs_slope"][b]
            policy.append([d + ceil_ratio(20*h*future, scale) for d,h in zip(inst,deriv)])
        values.append(policy)
    outputs = []
    for t, kernels_for_output in enumerate(ledger["kernels"]):
        for j, time_rows in enumerate(kernels_for_output):
            at_t = [p[j] if len(p)>j else 0 for p in values[t]]
            term = expected_by_budget(at_t, cells[t]["prob"])
            past = 0
            for i,(spans,environment) in enumerate(time_rows):
                state = [env + ceil_ratio(sp*max(values[i][b]),scale)
                         for b,(sp,env) in enumerate(zip(spans,environment))]
                past += expected_by_budget(state, cells[i]["prob"])
            outputs.append(term + arrival_integral(past))
    return max(outputs)


if __name__ == "__main__":
    ledger=json.loads(gzip.decompress(open(sys.argv[1],"rb").read()))
    num=bound(ledger)
    print(json.dumps(dict(model=ledger["calibration"],eta=ledger["share"],cells=ledger["n"],
                          numerator=num, denominator=ledger["scale"],
                          strictly_below_one=num<ledger["scale"]),indent=2))
