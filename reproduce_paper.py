#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducibility code for:
"Two-sided stable matching for low-value recyclable collection under
multi-type fuzzy information and partially known criterion weights"

Run: python reproduce_paper.py
Dependencies: numpy, pandas, scipy
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ENTERPRISES = ["A1", "A2", "A3", "A4"]
COMMUNITIES = ["R1", "R2", "R3", "R4", "R5", "R6"]

A_DATA = {
    "A1": [
        (0, (0.74, 0.43), "L",  "VH", (0.38, 0.52)),
        (0, (0.90, 0.18), "VH", "H",  (0.44, 0.58)),
        (0, (0.82, 0.32), "M",  "H",  (0.44, 0.58)),
        (1, (0.82, 0.32), "L",  "VH", (0.22, 0.34)),
        (1, (0.86, 0.26), "L",  "H",  (0.56, 0.72)),
        (0, (0.90, 0.18), "L",  "M",  (0.28, 0.40)),
    ],
    "A2": [
        (0, (0.74, 0.43), "VH", "VH", (0.38, 0.52)),
        (1, (0.90, 0.18), "H",  "H",  (0.56, 0.72)),
        (1, (0.55, 0.62), "H",  "L",  (0.50, 0.66)),
        (1, (0.55, 0.62), "VH", "VH", (0.44, 0.58)),
        (0, (0.82, 0.32), "L",  "VH", (0.38, 0.52)),
        (1, (0.70, 0.48), "H",  "VH", (0.22, 0.34)),
    ],
    "A3": [
        (1, (0.66, 0.52), "H",  "L",  (0.32, 0.46)),
        (0, (0.70, 0.48), "L",  "H",  (0.32, 0.46)),
        (1, (0.55, 0.62), "L",  "H",  (0.56, 0.72)),
        (1, (0.55, 0.62), "VH", "M",  (0.28, 0.40)),
        (1, (0.66, 0.52), "VH", "L",  (0.18, 0.28)),
        (1, (0.60, 0.58), "M",  "L",  (0.38, 0.52)),
    ],
    "A4": [
        (1, (0.74, 0.43), "VH", "VH", (0.18, 0.28)),
        (0, (0.86, 0.26), "H",  "L",  (0.22, 0.34)),
        (1, (0.90, 0.18), "VH", "L",  (0.50, 0.66)),
        (0, (0.70, 0.48), "H",  "VH", (0.18, 0.28)),
        (1, (0.66, 0.52), "H",  "L",  (0.28, 0.40)),
        (1, (0.86, 0.26), "M",  "H",  (0.32, 0.46)),
    ],
}

R_DATA = {
    "R1": [
        ("L",  (0.85, 0.08), "VH", (0.55, 0.62)),
        ("VH", (0.85, 0.08), "M",  (0.55, 0.62)),
        ("M",  (0.60, 0.22), "M",  (0.86, 0.26)),
        ("L",  (0.85, 0.08), "H",  (0.60, 0.58)),
    ],
    "R2": [
        ("H", (0.72, 0.15), "H", (0.66, 0.52)),
        ("H", (0.72, 0.15), "H", (0.55, 0.62)),
        ("H", (0.78, 0.12), "H", (0.55, 0.62)),
        ("H", (0.48, 0.28), "H", (0.70, 0.48)),
    ],
    "R3": [
        ("H", (0.72, 0.15), "H",  (0.55, 0.62)),
        ("L", (0.54, 0.25), "M",  (0.70, 0.48)),
        ("M", (0.72, 0.15), "VH", (0.60, 0.58)),
        ("M", (0.42, 0.32), "H",  (0.78, 0.38)),
    ],
    "R4": [
        ("L",  (0.72, 0.15), "VH", (0.66, 0.52)),
        ("L",  (0.60, 0.22), "M",  (0.78, 0.38)),
        ("VH", (0.85, 0.08), "H",  (0.60, 0.58)),
        ("M",  (0.48, 0.28), "L",  (0.66, 0.52)),
    ],
    "R5": [
        ("L",  (0.72, 0.15), "VH", (0.90, 0.18)),
        ("H",  (0.60, 0.22), "VH", (0.90, 0.18)),
        ("H",  (0.48, 0.28), "M",  (0.55, 0.62)),
        ("VH", (0.60, 0.22), "M",  (0.55, 0.62)),
    ],
    "R6": [
        ("H",  (0.42, 0.32), "VH", (0.74, 0.43)),
        ("L",  (0.66, 0.18), "M",  (0.90, 0.18)),
        ("M",  (0.85, 0.08), "VH", (0.70, 0.48)),
        ("VH", (0.54, 0.25), "L",  (0.74, 0.43)),
    ],
}

PUBLISHED_Y = np.array([
    [0.3020, 0.4633, 0.3444, 0.7469, 0.6709, 0.3112],
    [0.6298, 0.7326, 0.5466, 0.7467, 0.4938, 0.7852],
    [0.6646, 0.3481, 0.5982, 0.7807, 0.7462, 0.5818],
    [0.8788, 0.4686, 0.6824, 0.6131, 0.6170, 0.7308],
], dtype=float)

PUBLISHED_B = np.array([
    [0.6293, 0.7008, 0.6813, 0.5928, 0.6754, 0.7396],
    [0.7024, 0.6740, 0.4675, 0.4732, 0.8581, 0.5330],
    [0.5835, 0.6870, 0.6948, 0.7779, 0.5668, 0.7403],
    [0.5715, 0.6594, 0.5982, 0.4290, 0.6575, 0.5843],
], dtype=float)

PUBLISHED_P = np.array([
    [0.4656, 0.5820, 0.5128, 0.6699, 0.6731, 0.5254],
    [0.6661, 0.7033, 0.5070, 0.6099, 0.6760, 0.6591],
    [0.6240, 0.5176, 0.6465, 0.7793, 0.6565, 0.6611],
    [0.7252, 0.5640, 0.6403, 0.5210, 0.6373, 0.6575],
], dtype=float)

PAPER_STABLE_MATCHING = (4, 5, 3, 0)
PAPER_UNCONSTRAINED_MATCHING = (4, 1, 3, 0)

PAPER_EXACT_STABLE = {
    "enterprise_total": 3.115551,
    "community_total": 2.557723,
    "balanced_total": 2.836637,
}
PAPER_EXACT_UNCONSTRAINED = {
    "enterprise_total": 3.062922,
    "community_total": 2.698797,
    "balanced_total": 2.880860,
}

LINGUISTIC = {
    "VL": (0.00, 0.00, 0.25),
    "L":  (0.00, 0.25, 0.50),
    "M":  (0.25, 0.50, 0.75),
    "H":  (0.50, 0.75, 1.00),
    "VH": (0.75, 1.00, 1.00),
}

MAX_EUCLIDEAN = {
    "binary": 1.0,
    "linguistic": math.sqrt((0.75**2 + 1.0**2 + 0.75**2) / 3.0),
    "ifs": math.sqrt(2.0 / 3.0),
    "pfs": math.sqrt(2.0 / 3.0),
    "interval": 1.0,
}

MAX_HAMMING = {
    "binary": 1.0,
    "linguistic": (0.75 + 1.0 + 0.75) / 3.0,
    "ifs": 2.0 / 3.0,
    "pfs": 2.0 / 3.0,
    "interval": 1.0,
}


def linguistic_distance(x: str, y: str, metric: str = "euclidean") -> float:
    a, b = LINGUISTIC[x], LINGUISTIC[y]
    if metric == "euclidean":
        return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)) / 3.0)
    if metric == "hamming":
        return sum(abs(u - v) for u, v in zip(a, b)) / 3.0
    raise ValueError(f"Unknown metric: {metric}")


def ifs_distance(x: Tuple[float, float], ideal=(1.0, 0.0),
                 metric: str = "euclidean") -> float:
    mu, nu = x
    mu0, nu0 = ideal
    pi, pi0 = 1.0 - mu - nu, 1.0 - mu0 - nu0
    a = (mu, nu, pi)
    b = (mu0, nu0, pi0)
    if metric == "euclidean":
        return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)) / 3.0)
    if metric == "hamming":
        return sum(abs(u - v) for u, v in zip(a, b)) / 3.0
    raise ValueError(f"Unknown metric: {metric}")


def pfs_distance(x: Tuple[float, float], ideal=(1.0, 0.0),
                 metric: str = "euclidean") -> float:
    u, v = x
    u0, v0 = ideal
    psi, psi0 = 1.0 - u*u - v*v, 1.0 - u0*u0 - v0*v0
    a = (u*u, v*v, psi)
    b = (u0*u0, v0*v0, psi0)
    if metric == "euclidean":
        return math.sqrt(sum((q - r) ** 2 for q, r in zip(a, b)) / 3.0)
    if metric == "hamming":
        return sum(abs(q - r) for q, r in zip(a, b)) / 3.0
    raise ValueError(f"Unknown metric: {metric}")


def interval_cost_to_benefit(x: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = x
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"Invalid interval: {x}")
    return (1.0 - hi, 1.0 - lo)


def interval_distance(x: Tuple[float, float], ideal=(1.0, 1.0),
                      metric: str = "euclidean") -> float:
    lo, hi = interval_cost_to_benefit(x)
    a = (lo, hi)
    if metric == "euclidean":
        return math.sqrt(sum((q - r) ** 2 for q, r in zip(a, ideal)) / 2.0)
    if metric == "hamming":
        return sum(abs(q - r) for q, r in zip(a, ideal)) / 2.0
    raise ValueError(f"Unknown metric: {metric}")


def normalized_satisfaction(distance: float, information_type: str,
                             metric: str = "euclidean") -> float:
    maxima = MAX_EUCLIDEAN if metric == "euclidean" else MAX_HAMMING
    value = 1.0 - distance / maxima[information_type]
    return float(min(1.0, max(0.0, value)))


def validate_fuzzy_domains() -> None:
    for actor, rows in A_DATA.items():
        for j, (binary, pfs, l3, l4, interval) in enumerate(rows):
            u, v = pfs
            assert binary in (0, 1)
            assert u*u + v*v <= 1.0 + 1e-12, (actor, j, pfs)
            assert l3 in LINGUISTIC and l4 in LINGUISTIC
            assert 0.0 <= interval[0] <= interval[1] <= 1.0
    for community, rows in R_DATA.items():
        for i, (l1, ifs, l3, pfs) in enumerate(rows):
            mu, nu = ifs
            u, v = pfs
            assert mu + nu <= 1.0 + 1e-12, (community, i, ifs)
            assert u*u + v*v <= 1.0 + 1e-12, (community, i, pfs)
            assert l1 in LINGUISTIC and l3 in LINGUISTIC


def enterprise_criterion_matrix(actor: str, metric: str = "euclidean") -> np.ndarray:
    rows = []
    for binary, pfs, l3, l4, interval in A_DATA[actor]:
        rows.append([
            normalized_satisfaction(abs(1.0 - binary), "binary", metric),
            normalized_satisfaction(pfs_distance(pfs, metric=metric), "pfs", metric),
            normalized_satisfaction(linguistic_distance(l3, "VH", metric), "linguistic", metric),
            normalized_satisfaction(linguistic_distance(l4, "VH", metric), "linguistic", metric),
            normalized_satisfaction(interval_distance(interval, metric=metric), "interval", metric),
        ])
    return np.asarray(rows, dtype=float)


def community_criterion_matrix(community: str, metric: str = "euclidean") -> np.ndarray:
    rows = []
    for l1, ifs, l3, pfs in R_DATA[community]:
        rows.append([
            normalized_satisfaction(linguistic_distance(l1, "VH", metric), "linguistic", metric),
            normalized_satisfaction(ifs_distance(ifs, metric=metric), "ifs", metric),
            normalized_satisfaction(linguistic_distance(l3, "VH", metric), "linguistic", metric),
            normalized_satisfaction(pfs_distance(pfs, metric=metric), "pfs", metric),
        ])
    return np.asarray(rows, dtype=float)


def all_criterion_matrices(metric: str = "euclidean"):
    alpha = {a: enterprise_criterion_matrix(a, metric) for a in ENTERPRISES}
    delta = {r: community_criterion_matrix(r, metric) for r in COMMUNITIES}
    return alpha, delta


def aggregate_satisfaction_matrices(alpha: dict, delta: dict,
                                    enterprise_weights: dict,
                                    community_weights: dict):
    Y = np.zeros((len(ENTERPRISES), len(COMMUNITIES)), dtype=float)
    B = np.zeros_like(Y)
    for i, actor in enumerate(ENTERPRISES):
        w = np.asarray(enterprise_weights[actor], dtype=float)
        if w.shape != (5,) or not np.isclose(w.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Invalid enterprise weight vector for {actor}")
        Y[i, :] = alpha[actor] @ w
    for j, community in enumerate(COMMUNITIES):
        w = np.asarray(community_weights[community], dtype=float)
        if w.shape != (4,) or not np.isclose(w.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Invalid community weight vector for {community}")
        B[:, j] = delta[community] @ w
    return Y, B


def maximize_deviation_weight_sets(alpha: dict, delta: dict,
                                   constraints: dict):
    if not all(actor in constraints.get("enterprise", {}) for actor in ENTERPRISES):
        raise ValueError("Complete actor-specific enterprise constraints are required.")
    if not all(actor in constraints.get("community", {}) for actor in COMMUNITIES):
        raise ValueError("Complete actor-specific community constraints are required.")
    wa = {
        actor: maximize_deviation_weights(alpha[actor],
                    constraints["enterprise"][actor])
        for actor in ENTERPRISES
    }
    wr = {
        community: maximize_deviation_weights(delta[community],
                    constraints["community"][community])
        for community in COMMUNITIES
    }
    return wa, wr


def dispersion_coefficients(criterion_matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(criterion_matrix, dtype=float)
    n_alt, n_crit = x.shape
    return np.array([
        sum(abs(x[j, c] - x[k, c])
            for j in range(n_alt) for k in range(n_alt))
        for c in range(n_crit)
    ], dtype=float)


def _constraint_arrays(n: int, constraints: Optional[dict],
                       default_bounds=(0.08, 0.40)):
    constraints = constraints or {}
    A_ub = constraints.get("A_ub")
    b_ub = constraints.get("b_ub")
    A_eq = constraints.get("A_eq")
    b_eq = constraints.get("b_eq")
    if constraints.get("include_sum_to_one", True):
        if A_eq is None:
            A_eq, b_eq = [], []
        A_eq = list(A_eq) + [[1.0] * n]
        b_eq = list(b_eq) + [1.0]
    bounds = constraints.get("bounds", [list(default_bounds)] * n)
    if len(bounds) != n:
        raise ValueError("The number of weight bounds must equal the number of criteria.")
    return (
        None if A_ub is None or len(A_ub) == 0 else np.asarray(A_ub, dtype=float),
        None if b_ub is None or len(b_ub) == 0 else np.asarray(b_ub, dtype=float),
        None if A_eq is None or len(A_eq) == 0 else np.asarray(A_eq, dtype=float),
        None if b_eq is None or len(b_eq) == 0 else np.asarray(b_eq, dtype=float),
        [tuple(v) for v in bounds],
    )


def maximize_deviation_weights(criterion_matrix: np.ndarray,
                               constraints: Optional[dict] = None) -> np.ndarray:
    d = dispersion_coefficients(criterion_matrix)
    A_ub, b_ub, A_eq, b_eq, bounds = _constraint_arrays(
        len(d), constraints, default_bounds=(0.08, 0.40)
    )
    res = linprog(
        c=-d, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs"
    )
    if not res.success:
        raise ValueError(f"Weight constraints infeasible or optimization failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def neutral_feasible_weights(criterion_count: int,
                             constraints: Optional[dict] = None) -> np.ndarray:
    n = criterion_count
    A_ub, b_ub, A_eq, b_eq, bounds = _constraint_arrays(
        n, constraints, default_bounds=(0.08, 0.40)
    )
    x0 = np.full(n, 1.0 / n)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    x0 = np.clip(x0, lo, hi)
    x0 = x0 / x0.sum()

    cons = []
    if A_eq is not None:
        cons.append({"type": "eq", "fun": lambda x: A_eq @ x - b_eq})
    if A_ub is not None:
        cons.append({"type": "ineq", "fun": lambda x: b_ub - A_ub @ x})
    res = minimize(
        fun=lambda x: float(np.sum((x - 1.0 / n) ** 2)),
        x0=x0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"ftol": 1e-12, "maxiter": 2000}
    )
    if not res.success:
        raise ValueError(f"Neutral feasible-weight optimization failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def load_weight_constraints(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def enumerate_partial_matchings(m: int = 4, n: int = 6):
    for assignment in product(range(-1, n), repeat=m):
        assigned = [j for j in assignment if j >= 0]
        if len(assigned) == len(set(assigned)):
            yield tuple(assignment)


def matching_score(assignment: Sequence[int], matrix: np.ndarray) -> float:
    return float(sum(matrix[i, j] for i, j in enumerate(assignment) if j >= 0))


def blocking_pairs(assignment: Sequence[int], Y: np.ndarray, B: np.ndarray,
                   tol: float = 1e-12) -> List[Tuple[int, int]]:
    m, n = Y.shape
    community_to_enterprise = {
        j: i for i, j in enumerate(assignment) if j >= 0
    }
    blocks = []
    for i in range(m):
        current_j = assignment[i]
        current_enterprise_value = 0.0 if current_j < 0 else Y[i, current_j]
        for j in range(n):
            if current_j == j:
                continue
            if Y[i, j] < -tol or B[i, j] < -tol:
                continue
            enterprise_strict = Y[i, j] > current_enterprise_value + tol
            current_i_for_j = community_to_enterprise.get(j, None)
            current_community_value = (
                0.0 if current_i_for_j is None else B[current_i_for_j, j]
            )
            community_strict = B[i, j] > current_community_value + tol
            if enterprise_strict and community_strict:
                blocks.append((i, j))
    return blocks


def is_stable(assignment: Sequence[int], Y: np.ndarray, B: np.ndarray) -> bool:
    return len(blocking_pairs(assignment, Y, B)) == 0


def enumerate_stable_matchings(Y: np.ndarray, B: np.ndarray):
    all_matchings = list(enumerate_partial_matchings(Y.shape[0], Y.shape[1]))
    stable = [a for a in all_matchings if is_stable(a, Y, B)]
    return all_matchings, stable


def best_unconstrained_matching(P: np.ndarray):
    all_matchings = list(enumerate_partial_matchings(P.shape[0], P.shape[1]))
    best = max(all_matchings, key=lambda a: matching_score(a, P))
    return best, matching_score(best, P)


def format_matching(assignment: Sequence[int]) -> str:
    pairs = [
        f"({ENTERPRISES[i]}, {COMMUNITIES[j]})"
        for i, j in enumerate(assignment) if j >= 0
    ]
    unmatched_enterprises = [
        ENTERPRISES[i] for i, j in enumerate(assignment) if j < 0
    ]
    used = {j for j in assignment if j >= 0}
    unmatched_communities = [COMMUNITIES[j] for j in range(len(COMMUNITIES)) if j not in used]
    return (
        ", ".join(pairs)
        + f" | unmatched enterprises: {unmatched_enterprises}"
        + f" | unmatched communities: {unmatched_communities}"
    )


def solve_stable_for_policy(Y: np.ndarray, B: np.ndarray,
                            omega_A: float, omega_R: float):
    if not (0.0 <= omega_A <= 1.0 and 0.0 <= omega_R <= 1.0):
        raise ValueError("Policy weights must lie in [0,1].")
    if abs(omega_A + omega_R - 1.0) > 1e-10:
        raise ValueError("Policy weights must sum to one.")
    all_matchings, stable = enumerate_stable_matchings(Y, B)
    objective = omega_A * Y + omega_R * B
    ranked = sorted(stable, key=lambda a: matching_score(a, objective), reverse=True)
    return all_matchings, ranked


def shift_linguistic(term: str, direction: int) -> str:
    order = ["VL", "L", "M", "H", "VH"]
    return order[min(4, max(0, order.index(term) + direction))]


def perturb_evaluations(direction: str, fuzzy_step: float = 0.05,
                        interval_step: float = 0.05):
    if direction not in {"optimistic", "pessimistic"}:
        raise ValueError("direction must be 'optimistic' or 'pessimistic'")
    sign = 1.0 if direction == "optimistic" else -1.0
    a_new, r_new = {}, {}

    for actor, rows in A_DATA.items():
        new_rows = []
        for binary, (u, v), l3, l4, (lo, hi) in rows:
            u2 = min(1.0, max(0.0, u + sign * fuzzy_step))
            v2 = max(0.0, v - sign * fuzzy_step)
            if u2*u2 + v2*v2 > 1.0:
                v2 = math.sqrt(max(0.0, 1.0 - u2*u2))
            l3n = shift_linguistic(l3, int(sign))
            l4n = shift_linguistic(l4, int(sign))
            if direction == "optimistic":
                lon = max(0.0, lo - interval_step)
                hin = max(lon, hi - interval_step)
            else:
                hin = min(1.0, hi + interval_step)
                lon = min(hin, lo + interval_step)
            new_rows.append((binary, (u2, v2), l3n, l4n, (lon, hin)))
        a_new[actor] = new_rows

    for community, rows in R_DATA.items():
        new_rows = []
        for l1, (mu, nu), l3, (u, v) in rows:
            mu2 = min(1.0, max(0.0, mu + sign * fuzzy_step))
            nu2 = max(0.0, nu - sign * fuzzy_step)
            if mu2 + nu2 > 1.0:
                nu2 = max(0.0, 1.0 - mu2)
            u2 = min(1.0, max(0.0, u + sign * fuzzy_step))
            v2 = max(0.0, v - sign * fuzzy_step)
            if u2*u2 + v2*v2 > 1.0:
                v2 = math.sqrt(max(0.0, 1.0 - u2*u2))
            new_rows.append((shift_linguistic(l1, int(sign)), (mu2, nu2),
                             shift_linguistic(l3, int(sign)), (u2, v2)))
        r_new[community] = new_rows
    return a_new, r_new


def save_matrix(matrix: np.ndarray, row_names: Sequence[str],
                col_names: Sequence[str], filename: str):
    pd.DataFrame(matrix, index=row_names, columns=col_names).to_csv(
        OUT / filename, encoding="utf-8-sig", float_format="%.8f"
    )


def main():
    validate_fuzzy_domains()

    alpha, delta = all_criterion_matrices("euclidean")
    for actor in ENTERPRISES:
        save_matrix(alpha[actor], COMMUNITIES,
                    ["E1_binary", "E2_PFS", "E3_linguistic", "E4_linguistic", "E5_interval"],
                    f"alpha_{actor}.csv")
    for community in COMMUNITIES:
        save_matrix(delta[community], ENTERPRISES,
                    ["T1_linguistic", "T2_IFS", "T3_linguistic", "T4_PFS"],
                    f"delta_{community}.csv")

    save_matrix(PUBLISHED_Y, ENTERPRISES, COMMUNITIES, "published_Y.csv")
    save_matrix(PUBLISHED_B, ENTERPRISES, COMMUNITIES, "published_B.csv")
    save_matrix(PUBLISHED_P, ENTERPRISES, COMMUNITIES, "published_P.csv")

    dispersion_rows = []
    for actor in ENTERPRISES:
        dispersion_rows.append({
            "side": "enterprise", "actor": actor,
            **dict(zip(["E1", "E2", "E3", "E4", "E5"],
                       dispersion_coefficients(alpha[actor])))
        })
    for community in COMMUNITIES:
        dispersion_rows.append({
            "side": "community", "actor": community,
            **dict(zip(["T1", "T2", "T3", "T4"],
                       list(dispersion_coefficients(delta[community])) + [np.nan]))
        })
    pd.DataFrame(dispersion_rows).to_csv(
        OUT / "weight_dispersion_coefficients.csv", index=False, encoding="utf-8-sig"
    )

    all_matchings, stable = enumerate_stable_matchings(PUBLISHED_Y, PUBLISHED_B)
    if len(all_matchings) != 1045:
        raise AssertionError(f"Expected 1045 partial matchings, got {len(all_matchings)}")
    if stable != [PAPER_STABLE_MATCHING]:
        raise AssertionError(f"Stable matching differs from manuscript: {stable}")
    best_unstable, best_value = best_unconstrained_matching(PUBLISHED_P)
    if best_unstable != PAPER_UNCONSTRAINED_MATCHING:
        raise AssertionError(f"Unconstrained matching differs from manuscript: {best_unstable}")

    stable_assignment = stable[0]
    stable_blocks = blocking_pairs(stable_assignment, PUBLISHED_Y, PUBLISHED_B)
    unconstrained_blocks = blocking_pairs(best_unstable, PUBLISHED_Y, PUBLISHED_B)

    policy_rows = []
    for omega_A in [0.0, 0.25, 0.50, 0.75, 1.0]:
        omega_R = 1.0 - omega_A
        _, ranked = solve_stable_for_policy(PUBLISHED_Y, PUBLISHED_B, omega_A, omega_R)
        policy_rows.append({
            "omega_A": omega_A, "omega_R": omega_R,
            "stable_matching": format_matching(ranked[0]),
            "stable_matching_count": len(ranked),
        })
    pd.DataFrame(policy_rows).to_csv(
        OUT / "policy_weight_sensitivity_template.csv", index=False, encoding="utf-8-sig"
    )

    alpha_h, delta_h = all_criterion_matrices("hamming")
    for actor in ENTERPRISES:
        save_matrix(alpha_h[actor], COMMUNITIES,
                    ["E1_binary", "E2_PFS", "E3_linguistic", "E4_linguistic", "E5_interval"],
                    f"alpha_hamming_{actor}.csv")
    for community in COMMUNITIES:
        save_matrix(delta_h[community], ENTERPRISES,
                    ["T1_linguistic", "T2_IFS", "T3_linguistic", "T4_PFS"],
                    f"delta_hamming_{community}.csv")

    match_rows = []
    for label, assignment in [
        ("stable", stable_assignment),
        ("unconstrained", best_unstable),
    ]:
        match_rows.append({
            "solution": label,
            "matching": format_matching(assignment),
            "enterprise_satisfaction_from_published_matrix": matching_score(assignment, PUBLISHED_Y),
            "community_satisfaction_from_published_matrix": matching_score(assignment, PUBLISHED_B),
            "balanced_score_from_published_P": matching_score(assignment, PUBLISHED_P),
            "blocking_pairs": str([
                (ENTERPRISES[i], COMMUNITIES[j])
                for i, j in blocking_pairs(assignment, PUBLISHED_Y, PUBLISHED_B)
            ]),
        })
    pd.DataFrame(match_rows).to_csv(
        OUT / "matching_results.csv", index=False, encoding="utf-8-sig"
    )

    print("=" * 76)
    print("Reproduction checks")
    print("=" * 76)
    print(f"Partial one-to-one assignments enumerated: {len(all_matchings)} (expected 1045)")
    print(f"Number of stable matchings: {len(stable)} (expected 1)")
    print("Stable matching:", format_matching(stable_assignment))
    print("Stable matching blocking pairs:", stable_blocks)
    print("Unconstrained best matching:", format_matching(best_unstable))
    print("Unconstrained blocking pairs:", [
        (ENTERPRISES[i], COMMUNITIES[j]) for i, j in unconstrained_blocks
    ])
    print(f"Unconstrained objective from displayed P matrix: {best_value:.6f}")
    print()
    print("Manuscript's higher-precision reported totals:")
    print("Stable:", PAPER_EXACT_STABLE)
    print("Unconstrained:", PAPER_EXACT_UNCONSTRAINED)
    print()
    print("Criterion-level matrices were recalculated from Table 2 and saved.")
    print("Published Y/B/P matrices were saved separately and used for matching.")
    print("IMPORTANT: exact actor-specific weight optimization cannot be reproduced")
    print("until the omitted Omega_i / Lambda_j constraints and full weights are supplied.")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
