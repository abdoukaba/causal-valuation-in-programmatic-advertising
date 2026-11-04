"""Positive comparison for the exact response. Arithmetic backend is float for search or Arb for proof."""
import math, json, sys
from types import SimpleNamespace
import numpy as np
from enclose import stored, A, ctx
from flint import arb

class Math:

    def __init__(self, rig):
        self.rig = rig
        # float search mode accepts an Arb ball string such as '[0.5 +/- 0.05]' and uses its midpoint
        self.n = A if rig else (lambda x: float(A(x).mid()) if isinstance(x, str) and x.startswith('[') else float(x))

    def maximum(self, items):
        iterator = iter(items)
        x = next(iterator)
        for y in iterator:
            x = x.max(y) if self.rig else max(x, y)
        return x

    def minimum(self, items):
        return -self.maximum((-x for x in items))

    def exp(self, x):
        return x.exp() if self.rig else math.exp(x)

    def log(self, x):
        return x.log() if self.rig else math.log(x)

    def sqrt(self, x):
        return x.sqrt() if self.rig else math.sqrt(x)

    def box_max(self, values, lo, hi):
        # Dual formula for maximization over a box intersected with the unit simplex.
        return self.minimum((c + sum((h * self.maximum([v - c, self.z]) + l * self.minimum([v - c, self.z]) for v, l, h in zip(values, lo, hi))) for c in values))

    def box_min(self, values, lo, hi):
        return -self.box_max([-v for v in values], lo, hi)
edges = [(0, 1), (0, 2), (1, 2)]

def vertices(lower, upper):
    # supplied as EXACT integer millionths. Feasible vertices fix two coordinates.
    out = set()
    for i, j in edges:
        k = 3 - i - j
        for a in [lower[i], upper[i]]:
            for b in [lower[j], upper[j]]:
                x = [0] * 3
                x[i] = a
                x[j] = b
                x[k] = 1000000 - a - b
                if lower[k] <= x[k] <= upper[k]:
                    out.add(tuple(x))
    assert out
    return sorted(out)

def inputs(name, eta, math_ops, box):
    grid, q, z, d, D = stored(name)
    if not math_ops.rig:
        grid, q, z, d, D = (list(map(float, grid)), list(map(float, q)), float(z), list(map(float, d)), [list(map(float, row)) for row in D])
    eta = math_ops.n(eta)
    vertices_q = [[(1 - eta) * q[j] + (eta * v[j] / 1000000 if j < 3 else 0) for j in range(4)] for v in vertices(*box)]
    actions = [None]
    for a in grid[1:3]:
        wins = [[1 / (1 + math_ops.exp(-(a - v - off) / z)) for v in grid] for off in d]
        rewards = [[w * (v - a) for w, v in zip(ws, vs)] for ws, vs in zip(wins, D)]
        likelihoods = [tuple((sum((v[j] * ws[j] for j in range(4))) for ws in wins)) for v in vertices_q]
        g = [tuple((sum((v[j] * gs[j] for j in range(4))) for gs in rewards)) for v in vertices_q]
        win_entries = [x for row in likelihoods for x in row]
        Lg = math_ops.maximum((abs(x - y) for x, y in g))
        Lw = math_ops.maximum((abs(x - y) for x, y in likelihoods))
        posterior_span = math_ops.z
        for s in [0, 1]:
            sr = math_ops.sqrt(math_ops.maximum((row[s] / row[1 - s] for row in likelihoods)) * math_ops.maximum(((1 - row[1 - s]) / (1 - row[s]) for row in likelihoods)))
            posterior_span = math_ops.maximum([posterior_span, (sr - 1) / (sr + 1)])
        edges_data = []
        for i, j in edges:
            w0, w1 = (wins[0][i] - wins[0][j], wins[1][i] - wins[1][j])
            assert w0 > 0 and w1 > 0
            beta = math_ops.z
            for rows in (likelihoods, [(1 - x, 1 - y) for x, y in likelihoods]):
                den = (math_ops.sqrt(math_ops.minimum((x for x, y in rows))) + math_ops.sqrt(math_ops.minimum((y for x, y in rows)))) ** 2
                beta += math_ops.maximum((abs(x * w1 - y * w0) for x, y in rows)) / den
            edges_data.append(([rewards[s][i] - rewards[s][j] for s in range(2)], [w0, w1], beta))
        actions.append(SimpleNamespace(likelihoods=win_entries, g=[x for row in g for x in row], lo=math_ops.minimum(win_entries), hi=math_ops.maximum(win_entries), Lg=Lg, Lw=Lw, sep=posterior_span, edge=edges_data))
    return actions

def model(name, eta, n, rig, box):
    math_ops = Math(rig)
    math_ops.z = math_ops.n(0)
    z = math_ops.z
    one = math_ops.n(1)
    e = math_ops.n(__import__('os').environ.get('EPS','.05'))
    la = math_ops.n('2.25')
    dt = one / n
    a = inputs(name, eta, math_ops, box)
    eta = math_ops.n(eta)
    R = e * math_ops.log(1 + sum((math_ops.exp(math_ops.maximum(v.g) / e) for v in a[1:])))
    no = math_ops.exp(-la * dt)
    hit = 1 - no
    right_lower = [z] * 5
    right_upper = [z] * 5
    lip = z
    cells = [None] * n
    for k in reversed(range(n)):
        cell_lower = [no * v for v in right_lower]
        cell_upper = [math_ops.maximum(right_upper[:b + 1]) + la * dt * R if b else z for b in range(5)]
        # Budget monotonicity permits clipping differences below by zero.
        budget_gaps = [[(math_ops.maximum([z, cell_lower[b] - cell_upper[b - j]]), math_ops.maximum([z, cell_upper[b] - cell_lower[b - j]])) for j in range(min(2, b) + 1)] for b in range(5)]
        rate = math_ops.maximum((a[j].Lg + a[j].Lw * budget_gaps[b][j][1] for b in range(1, 5) for j in range(1, min(2, b) + 1)))
        lip += la * dt * rate
        budget_rows = []
        next_lower = []
        next_upper = []
        for b in range(5):
            bids = range(min(2, b) + 1)
            center_lower = [z]
            center_upper = [z]
            score_lower = [cell_lower[b]]
            score_upper = [cell_upper[b]]
            for j in range(1, len(bids)):
                u = a[j]
                gap_lower, gap_upper = budget_gaps[b][j]
                center_lower.append(math_ops.minimum((g - w * gap_upper for g, w in zip(u.g, u.likelihoods))))
                center_upper.append(math_ops.maximum((g - w * gap_lower for g, w in zip(u.g, u.likelihoods))) + lip * u.Lw / 2)
                score_lower.append(math_ops.minimum((g + (1 - w) * cell_lower[b] + w * cell_lower[b - j] for g, w in zip(u.g, u.likelihoods))))
                score_upper.append(math_ops.maximum((g + (1 - w) * cell_upper[b] + w * cell_upper[b - j] for g, w in zip(u.g, u.likelihoods))))
            next_lower.append(no * right_lower[b] + hit * e * math_ops.log(sum((math_ops.exp(x / e) for x in score_lower))))
            next_upper.append(no * right_upper[b] + hit * e * math_ops.log(sum((math_ops.exp(x / e) for x in score_upper))))
            policy_lower = [one / (1 + sum((math_ops.exp((center_upper[l] - center_lower[j]) / e) for l in bids if l != j))) for j in bids]
            policy_upper = [one / (1 + sum((math_ops.exp((center_lower[l] - center_upper[j]) / e) for l in bids if l != j))) for j in bids]
            gibbs_factor = [math_ops.minimum([one / 4, x * (1 - x) if x > one / 2 else one / 4, y * (1 - y) if y < one / 2 else one / 4]) for x, y in zip(policy_lower, policy_upper)]
            # The decision-time optimal policy is obtained from centered scores.
            score_lipschitz = [lip] + [lip + a[j].Lg + a[j].Lw * budget_gaps[b][j][1] for j in bids if j]
            policy_lipschitz = [gibbs_factor[j] * (score_lipschitz[j] + math_ops.maximum((score_lipschitz[l] for l in bids if l != j))) / e if len(bids) > 1 else z for j in bids]
            average_forcing = []
            instant_gain = [z] * len(bids)
            for v in range(3):
                force_lower = [z]
                force_upper = [z]
                for j in range(1, len(bids)):
                    u = a[j]
                    gg, ww, beta = u.edge[v]
                    gap_lower, gap_upper = budget_gaps[b][j]
                    swing = lip * (math_ops.maximum(ww) * u.sep + beta)
                    force_lower.append(eta * (math_ops.minimum((g - w * gap_upper for g, w in zip(gg, ww))) - swing))
                    force_upper.append(eta * (math_ops.maximum((g - w * gap_lower for g, w in zip(gg, ww))) + swing))
                for j in bids:
                    if len(bids) > 1:
                        instant_gain[j] = math_ops.maximum([instant_gain[j], gibbs_factor[j] / e * math_ops.maximum([force_upper[j] - math_ops.minimum((force_lower[l] for l in bids if l != j)), math_ops.maximum((force_upper[l] for l in bids if l != j)) - force_lower[j]])])
                average_forcing.append((math_ops.box_min(force_lower, policy_lower, policy_upper), math_ops.box_max(force_upper, policy_lower, policy_upper)))
            budget_rows.append(SimpleNamespace(policy_lower=policy_lower, policy_upper=policy_upper, gibbs_factor=gibbs_factor, policy_lipschitz=policy_lipschitz, instant_gain=instant_gain, average_forcing=average_forcing))
        right_lower, right_upper = (next_lower, next_upper)
        # Range of the expected signed forcing at a future instant reachable from budget b.
        forcing_span = [math_ops.maximum((math_ops.maximum((budget_rows[i].average_forcing[v][1] for i in range(b + 1))) - math_ops.minimum((budget_rows[i].average_forcing[v][0] for i in range(b + 1))) for v in range(3))) for b in range(5)]
        # Each mark dominates the actual spend given the past (units of h).
        spend_one_or_more = math_ops.maximum((math_ops.box_max([z] + [a[j].hi for j in range(1, len(row.policy_lower))], row.policy_lower, row.policy_upper) for row in budget_rows))
        spend_two = math_ops.maximum([z] + [row.policy_upper[2] * a[2].hi for row in budget_rows if len(row.policy_lower) == 3])
        spend_one_or_more = math_ops.maximum([spend_one_or_more, spend_two])
        assert 0 <= spend_two and spend_one_or_more < 1
        cells[k] = SimpleNamespace(row=budget_rows, forcing_span=forcing_span, spend_one_or_more=spend_one_or_more, spend_two=spend_two)
    # Compound Poisson dominating expenditures, upper CDF of the accessible lower budgets.
    mu1 = mu2 = z
    for c in cells:
        mu1 += la * dt * (c.spend_one_or_more - c.spend_two)
        mu2 += la * dt * c.spend_two
        h0 = math_ops.exp(-mu1 - mu2)
        p = [h0, h0 * mu1, h0 * (mu1 ** 2 / 2 + mu2), h0 * (mu1 ** 3 / 6 + mu1 * mu2)]
        c.probs = p + [1 - sum(p)]
    return SimpleNamespace(math_ops=math_ops, a=a, c=cells, e=e, la=la, h=dt, z=z, one=one, eta=eta, no=no, hit=hit)

def occ(math_ops, v, p):
    # includes times earlier than the right endpoint through the same stochastic upper bound.
    return sum((p[n] * math_ops.maximum((v[b] for b in range(4 - n, 5))) for n in range(5)))

def restrict(state):
    math_ops, c = (state.math_ops, state.c)
    l = []
    h = []
    for j in range(3):
        lower = math_ops.minimum((-occ(math_ops, [-(row.policy_lower[j] if j < len(row.policy_lower) else state.z) for row in a.row], a.probs) for a in c))
        upper = math_ops.maximum((occ(math_ops, [row.policy_upper[j] if j < len(row.policy_lower) else state.z for row in a.row], a.probs) for a in c))
        lo_f = float(lower.lower()) if math_ops.rig else float(lower); hi_f = float(upper.upper()) if math_ops.rig else float(upper)
        l.append(max(0, math.floor(lo_f * 1000000.0) - 2))
        h.append(min(1000000, math.ceil(hi_f * 1000000.0) + 2))
        assert (l[-1] == 0 or state.one * l[-1] / 1000000 <= lower) and (h[-1] == 1000000 or state.one * h[-1] / 1000000 >= upper), (j, lower.str(10) if math_ops.rig else lower, upper.str(10) if math_ops.rig else upper, l[-1], h[-1])
    return (l, h)

def kernels(state):
    math_ops, z, one, la, dt = (state.math_ops, state.z, state.one, state.la, state.h)
    data = []
    for t, terminal_cell in enumerate(state.c):
        terminal = []
        for j in range(3):
            end_lower = [row.policy_lower[j] if j < len(row.policy_lower) else z for row in terminal_cell.row]
            end_upper = [row.policy_upper[j] if j < len(row.policy_lower) else z for row in terminal_cell.row]
            test_lip = math_ops.maximum([z] + [row.policy_lipschitz[j] for row in terminal_cell.row if j < len(row.policy_lower)])
            rows = []
            for s in reversed(range(t + 1)):
                c = state.c[s]
                low = [state.no * v + state.hit * math_ops.minimum(end_lower[:b + 1]) for b, v in enumerate(end_lower)]
                high = [state.no * v + state.hit * math_ops.maximum(end_upper[:b + 1]) for b, v in enumerate(end_upper)]
                action_span = []
                lip_growth = []
                cross = []
                left_lower = []
                left_upper = []
                for b, row in enumerate(c.row):
                    bids = range(len(row.policy_lower))
                    kernel_lower = [low[b]]
                    kernel_upper = [high[b]]
                    cross_budget = [z]
                    for i in range(1, len(bids)):
                        aa = state.a[i]
                        kernel_lower.append(math_ops.minimum(((1 - w) * low[b] + w * low[b - i] for w in (aa.lo, aa.hi))))
                        kernel_upper.append(math_ops.maximum(((1 - w) * high[b] + w * high[b - i] for w in (aa.lo, aa.hi))))
                        cross_budget.append(math_ops.maximum([high[b] - low[b - i], high[b - i] - low[b]]))
                    span = math_ops.maximum([z] + [kernel_upper[i] - kernel_lower[h] for i in bids for h in bids if i != h])
                    action_span.append(span)
                    cross.append(cross_budget)
                    lip_growth.append(math_ops.box_max([z] + [state.a[i].Lw * cross_budget[i] for i in bids if i], row.policy_lower, row.policy_upper) + math_ops.maximum(row.policy_lipschitz) * span)
                    cl = state.no * end_lower[b] + state.hit * math_ops.box_min(kernel_lower, row.policy_lower, row.policy_upper)
                    ch = state.no * end_upper[b] + state.hit * math_ops.box_max(kernel_upper, row.policy_lower, row.policy_upper)
                    left_lower.append(math_ops.minimum([end_lower[b], cl]) if s == t else cl)
                    left_upper.append(math_ops.maximum([end_upper[b], ch]) if s == t else ch)
                test_lip += la * dt * math_ops.maximum(lip_growth)
                env = []
                for b, row in enumerate(c.row):
                    vals = []
                    for v in range(3):
                        val = [z] + [math_ops.maximum(state.a[i].edge[v][1]) * cross[b][i] + state.a[i].edge[v][2] * test_lip for i in range(1, len(row.policy_lower))]
                        vals.append(math_ops.box_max(val, row.policy_lower, row.policy_upper))
                    env.append(state.eta * math_ops.maximum(vals))
                rows.append((action_span, env))
                end_lower, end_upper = (left_lower, left_upper)
            terminal.append(rows[::-1])
        data.append(terminal)
    return data

def evalw(state, K, w):
    math_ops, z = (state.math_ops, state.z)
    future = [z] * 5
    past = []
    coord = []
    for i in reversed(range(len(w))):
        c = state.c[i]
        future = [a + state.la * state.h * b * w[i] for a, b in zip(future, c.forcing_span)]
        per = [[f * w[i] + h / state.e * future[b] for f, h in zip(row.instant_gain, row.gibbs_factor)] for b, row in enumerate(c.row)]
        past.append([math_ops.maximum(v) for v in per])
        coord.append(per)
    past.reverse()
    coord.reverse()
    out = []
    for t, terminal_cell in enumerate(state.c):
        val = []
        for j, rows in enumerate(K[t]):
            x = occ(math_ops, [v[j] if len(v) > j else z for v in coord[t]], terminal_cell.probs)
            for s, (action_span, env) in enumerate(rows):
                x += state.la * state.h * occ(math_ops, [a * p + b * w[s] for a, p, b in zip(action_span, past[s], env)], state.c[s].probs)
            val.append(x)
        out.append(math_ops.maximum(val))
    return out

def cert(name, eta, n, rig=False, rounds=2):
    box = ([0, 0, 0], [1000000] * 3)
    boxes = [box]
    for i in range(rounds):
        state = model(name, eta, n, rig, box)
        lo, hi = restrict(state)
        box = ([max(x, y) for x, y in zip(box[0], lo)], [min(x, y) for x, y in zip(box[1], hi)])
        boxes.append(box)
    state = model(name, eta, n, rig, box)
    K = kernels(state)
    # Find a positive witness in ordinary float irrespective of the verification backend.
    if rig:
        float_model = model(name, eta, n, False, box)
        float_kernels = [[[([float(v) for v in a], [float(v) for v in b]) for a, b in rr] for rr in R] for R in K]
    else:
        float_model = state
        float_kernels = K
    w = np.ones(n)
    for _ in range(45):
        y = np.array(evalw(float_model, float_kernels, w))
        w = y / y.max()
    w = [state.math_ops.n(f'{v:.12f}') for v in w]
    assert all((v > 0 for v in w))
    y = evalw(state, K, w)
    k = state.math_ops.maximum((b / a for a, b in zip(w, y)))
    unit = state.math_ops.maximum(evalw(state, K, [state.one] * n))
    result = dict(unweighted_bound=unit.str(14) if rig else unit, unweighted_below_one=bool(unit < 1), model=name, eta=str(eta), cells=n, arb=rig, precision_bits=ctx.prec if rig else None, bound=k.str(14) if rig else k, below_one=bool(k < 1), outer_boxes=boxes, weight=[round(float(v), 12) for v in w])
    if rig:
        # The compact Arb midpoint-radius display may extend above one even when
        # the internal ball proves k < 1. Save exact decimal upper ceilings too.
        def ceiling(x):
            v=int((x*10**12).upper().ceil().unique_fmpz())
            return f'{v//10**12}.{v%10**12:012d}'
        result.update(bound_upper=ceiling(k),unweighted_upper=ceiling(unit))
    return result
if __name__ == '__main__':
    name, eta, n, rig = sys.argv[1:5]
    res = cert(name, eta, int(n), bool(int(rig)), int(sys.argv[5]) if len(sys.argv) > 5 else 2)
    print(json.dumps(res, indent=1))
