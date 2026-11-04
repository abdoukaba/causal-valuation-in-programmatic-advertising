"""Reproduce Section 2.4's floating-point terminal-budget summary."""
import json
import sys
from pathlib import Path

from grid_certificate import Model


def compute():
    model = Model(21, 60, 0.45, 0.05, 1.0, 2.25)
    solution = model.solve(0, max_iter=800, damping=0.5)
    terminal_mass = solution['m'][-1].sum(axis=1)
    remaining = float(terminal_mass @ model.budgets)
    return {
        'belief_nodes': model.np,
        'time_steps': model.nt,
        'eta': model.eta,
        'entropy_weight': model.entropy,
        'expected_auctions': model.lam * model.T,
        'initial_budget': float(model.budgets[-1]),
        'mean_terminal_budget': remaining,
        'fraction_of_budget_remaining': remaining / float(model.budgets[-1]),
        'probability_budget_exhausted': float(terminal_mass[0]),
        'equilibrium_profit': model.monetary_policy_value(solution['q'], solution['pol']),
        'equilibrium_residual': solution['res'],
        'method': 'floating-point equilibrium on the stated grid; not interval-certified',
    }


if __name__ == '__main__':
    result = compute()
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('benchmark_budget.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
