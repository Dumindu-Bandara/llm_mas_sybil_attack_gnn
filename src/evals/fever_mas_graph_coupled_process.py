"""
Graph-coupled Ornstein-Uhlenbeck process / Linear-Gaussian State-Space Model
(LG-SSM) for detecting the naive replacement Sybil attack simulated in
`fever_mas_naive_sybil.py`.
=============================================================================

Model
-----
Each of the n=4 agent personas (LiteralMatcher, Skeptic, ContextReasoner,
DevilsAdvocate) is treated as a node of the fully-connected communication
graph K_n used by the MAS (all-to-all `observe()`/broadcast_recap). A node's
continuous "belief state" at deliberation round t is a d=2 feature vector
built from its structured Verdict:

    channel 0 (signed polarity) = confidence * (+1 SUPPORTS / -1 REFUTES / 0 NOT_ENOUGH_INFO)
    channel 1 (raw confidence)  = confidence, unsigned

Honest consensus dynamics are modeled as a multivariate Ornstein-Uhlenbeck
(OU) process coupled through the graph Laplacian L of K_n:

    dx(t) = -(theta*I + kappa*L) (x(t) - mu) dt + Sigma^{1/2} dW(t)

`theta` is each node's private mean reversion, `kappa` is how strongly a
node's belief is pulled toward its neighbors' beliefs (the consensus term:
(L x)_i = sum_{j in N(i)} (x_i - x_j)), and Sigma is the diffusion. Because
the drift is linear and the noise additive-Gaussian, one round -> round
transition is *exactly* Gaussian:

    x(t+1) | x(t) ~ N(mu + F (x(t) - mu), Q)

with F = expm(A) and Q solved from the associated Lyapunov integral, via the
Van Loan (1978) block-matrix trick (see `discretize_ou`). This turns the
per-claim vote trajectory into a standard linear-Gaussian state-space model
that a Kalman filter can score exactly:

    x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)      (transition = graph-OU)
    y_t = H_t x_t + v_t,     v_t ~ N(0, R)       (observation = the votes actually cast)

Naive Sybils are, by construction (see `fever_mas_naive_sybil.py`), agents
that fixate on a fixed target label/confidence and never actually update in
response to peers. That is exactly a violation of the fitted honest-graph-OU
transition law, so it shows up as a large, persistent Kalman *innovation*
(observed vs. one-step-ahead predicted) at that node -- this is the standard
innovation-based fault/anomaly-detection use of a Kalman filter.

Pipeline in this file
----------------------
1. `load_dataset`      - parse `fever_mas_naive_sybil_results.jsonl` into
                          per-claim (T, n, d) trajectories + is_sybil labels.
2. `fit_null_model`     - fit (theta, kappa, sigma2, r2) by maximizing the
                          Kalman-filter marginal log-likelihood, using ONLY
                          the ground-truth-honest nodes at every round (the
                          Sybil nodes are masked out, i.e. treated as
                          missing observations) -- this estimates what
                          *honest* consensus dynamics look like on this
                          graph, independent of any particular claim's
                          Sybils.
3. `score_trajectories` - run the fitted (F, Q, R) Kalman filter over the
                          FULL 4-node trajectory (nobody masked this time)
                          and turn each node's standardized innovations into
                          a scalar `sybil_score`.
4. `evaluate_auc`       - compare `sybil_score` against the true `is_sybil`
                          label with ROC-AUC / average precision.
5. `simulate_naive_sybil_dataset` - generate synthetic labeled trajectories
                          directly from the (fitted or hand-set) graph-OU
                          model, with Sybil nodes clamped to a fixed target
                          state instead of following the dynamics -- an
                          LLM-free stand-in for `fever_mas_naive_sybil.py`,
                          useful for unit-testing this pipeline and for
                          augmenting the labeled dataset for the downstream
                          Sybil-detection GNN mentioned in that file's
                          docstring.

Run directly for a report:

    ml conda && conda activate mas_framework_test
    python fever_mas_graph_coupled_process.py --max-claims 1500
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover - sklearn is expected to be present
    roc_auc_score = None
    average_precision_score = None


AGENT_NAMES = ["LiteralMatcher", "Skeptic", "ContextReasoner", "DevilsAdvocate"]
N_AGENTS = len(AGENT_NAMES)
N_CHANNELS = 2  # [signed polarity, raw confidence]

LABEL_POLARITY = {
    "SUPPORTS": 1.0,
    "REFUTES": -1.0,
    "NOT_ENOUGH_INFO": 0.0,
}


# ---------------------------------------------------------------------
# 1. Data loading: results jsonl -> per-claim (T, n, d) trajectories
# ---------------------------------------------------------------------
@dataclass
class ClaimTrajectory:
    index: int
    claim: str
    names: list[str]
    is_sybil: np.ndarray  # (n,) bool
    X: np.ndarray  # (T, n, d) float -- per-round, per-agent feature vector
    ground_truth_label: Optional[str] = None
    target_label: Optional[str] = None
    sybil_count: int = 0


def vote_to_features(vote: dict) -> np.ndarray:
    """Structured Verdict -> continuous 2D belief-state feature vector."""
    confidence = float(vote["confidence"])
    polarity = LABEL_POLARITY.get(vote["label"], 0.0)
    return np.array([polarity * confidence, confidence], dtype=float)


def load_dataset(path: str, max_claims: Optional[int] = None) -> list[ClaimTrajectory]:
    """Parse a `fever_mas_naive_sybil.py`-style results jsonl into
    per-claim trajectories. Rows that don't have the expected shape
    (parse failures, aborted claims, etc.) are silently skipped.
    """
    trajectories: list[ClaimTrajectory] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_claims is not None and len(trajectories) >= max_claims:
                break
            try:
                row = json.loads(line)
                roster = row["roster"]
                names = [entry["name"] for entry in roster]
                is_sybil = np.array([bool(entry["is_sybil"]) for entry in roster])
                vote_history = row["vote_history"]
                n = len(names)
                T = len(vote_history)
                if T == 0 or n == 0:
                    continue
                X = np.zeros((T, n, N_CHANNELS))
                for t, round_votes in enumerate(vote_history):
                    for j, name in enumerate(names):
                        X[t, j] = vote_to_features(round_votes[name])
                trajectories.append(
                    ClaimTrajectory(
                        index=i,
                        claim=row.get("claim", ""),
                        names=names,
                        is_sybil=is_sybil,
                        X=X,
                        ground_truth_label=row.get("ground_truth_label"),
                        target_label=row.get("target_label"),
                        sybil_count=int(row.get("sybil_count", int(is_sybil.sum()))),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return trajectories


# ---------------------------------------------------------------------
# 2. Graph-coupled OU -> exact linear-Gaussian discretization (Van Loan)
# ---------------------------------------------------------------------
def laplacian_complete_graph(n: int) -> np.ndarray:
    """Unweighted graph Laplacian of K_n (the MAS's all-to-all recap
    topology): degree n-1 on the diagonal, -1 off-diagonal.
    """
    return n * np.eye(n) - np.ones((n, n))


def discretize_ou(A: np.ndarray, Sigma: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact discretization of dx = A x dt + Sigma^{1/2} dW over one step
    of length `dt`, via Van Loan's (1978) block-matrix method (see e.g.
    Sarkka, "Bayesian Filtering and Smoothing", eq. 4.11-4.13):

        Phi = expm( dt * [[A, Sigma], [0, -A.T]] )  = [[C11, C12], [0, C22]]
        F = C22.T
        Q = C22.T @ C12

    Returns (F, Q) such that x_{t+1} = F x_t + w_t, w_t ~ N(0, Q).
    """
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = A
    M[:n, n:] = Sigma
    M[n:, n:] = -A.T
    Phi = expm(M * dt)
    C12 = Phi[:n, n:]
    C22 = Phi[n:, n:]
    F = C22.T
    Q = F @ C12
    Q = 0.5 * (Q + Q.T)  # symmetrize away numerical drift
    return F, Q


def build_F_Q(theta: float, kappa: float, sigma2: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    L = laplacian_complete_graph(n)
    A = -(theta * np.eye(n) + kappa * L)
    Sigma = sigma2 * np.eye(n)
    return discretize_ou(A, Sigma, dt=1.0)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def unpack_params(raw: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Unconstrained -> positive-constrained (theta, kappa, sigma2, r2[d])."""
    theta = float(softplus(raw[0])) + 1e-4
    kappa = float(softplus(raw[1]))
    sigma2 = float(softplus(raw[2])) + 1e-6
    r2 = softplus(raw[3 : 3 + N_CHANNELS]) + 1e-6
    return theta, kappa, sigma2, r2


# ---------------------------------------------------------------------
# 3. Kalman filter with missing-data masking (used both for masked-honest
#    fitting and for full-graph scoring)
# ---------------------------------------------------------------------
def kalman_run(
    Y: np.ndarray,
    mask: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R_diag: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run a Kalman filter over a single-channel (T, n) trajectory.

    `mask[t, i]` = True if node i's value at round t is treated as observed
    (False = missing, e.g. a Sybil masked out during null-model fitting).

    Returns:
        loglik      - total observed-data marginal log-likelihood
        innov_full  - (T, n) innovation e_t,i = y_t,i - predicted, NaN
                      where not observed
        svar_full   - (T, n) innovation variance S_t,ii, NaN where not
                      observed
    """
    T, n = Y.shape
    x = x0.copy()
    P = P0.copy()
    loglik = 0.0
    innov_full = np.full((T, n), np.nan)
    svar_full = np.full((T, n), np.nan)
    eye_n = np.eye(n)

    for t in range(T):
        if t > 0:
            x = F @ x
            P = F @ P @ F.T + Q

        obs_idx = np.where(mask[t])[0]
        if obs_idx.size > 0:
            H = eye_n[obs_idx]  # (k, n) selection matrix
            y = Y[t, obs_idx]
            R = np.diag(R_diag[obs_idx])

            y_pred = H @ x
            e = y - y_pred
            S = H @ P @ H.T + R
            S = 0.5 * (S + S.T)

            L_chol = np.linalg.cholesky(S)
            alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, e))
            loglik += -0.5 * (
                obs_idx.size * np.log(2 * np.pi)
                + 2.0 * np.sum(np.log(np.diag(L_chol)))
                + e @ alpha
            )

            K = np.linalg.solve(S.T, (P @ H.T).T).T  # P H^T S^-1, stable
            x = x + K @ e
            IKH = eye_n - K @ H
            P = IKH @ P @ IKH.T + K @ R @ K.T

            innov_full[t, obs_idx] = e
            svar_full[t, obs_idx] = np.diag(S)

    return loglik, innov_full, svar_full


# ---------------------------------------------------------------------
# 4. Fit the honest null model (mask Sybil nodes out of the likelihood)
# ---------------------------------------------------------------------
def negloglik(
    raw: np.ndarray,
    trajectories: list[ClaimTrajectory],
    n: int,
    mask_honest: bool,
    P0_scale: float,
) -> float:
    theta, kappa, sigma2, r2 = unpack_params(raw)
    F, Q = build_F_Q(theta, kappa, sigma2, n)
    P0 = P0_scale * np.eye(n)
    x0 = np.zeros(n)

    total = 0.0
    for traj in trajectories:
        T = traj.X.shape[0]
        if mask_honest:
            mask = np.tile(~traj.is_sybil, (T, 1))
        else:
            mask = np.ones((T, n), dtype=bool)
        for c in range(N_CHANNELS):
            Y = traj.X[:, :, c]
            R_diag = np.full(n, r2[c])
            ll, _, _ = kalman_run(Y, mask, F, Q, R_diag, x0, P0)
            total += ll
    return -total


def fit_null_model(
    trajectories: list[ClaimTrajectory],
    n: int = N_AGENTS,
    init: Optional[np.ndarray] = None,
    P0_scale: float = 1.0,
    maxiter: int = 100,
    verbose: bool = True,
) -> dict:
    """Maximum-likelihood fit of the honest graph-OU dynamics, using only
    the ground-truth-honest nodes (Sybils masked out) at every round.
    """
    if init is None:
        init = np.zeros(3 + N_CHANNELS)
    result = minimize(
        negloglik,
        init,
        args=(trajectories, n, True, P0_scale),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "disp": verbose},
    )
    theta, kappa, sigma2, r2 = unpack_params(result.x)
    return {
        "theta": theta,
        "kappa": kappa,
        "sigma2": sigma2,
        "r2": r2,
        "P0_scale": P0_scale,
        "raw": result.x,
        "opt_result": result,
    }


# ---------------------------------------------------------------------
# 5. Score every node under the fitted honest model (innovation-based
#    anomaly / Sybil score)
# ---------------------------------------------------------------------
def score_trajectories(
    trajectories: list[ClaimTrajectory],
    params: dict,
    n: int = N_AGENTS,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run the fitted (F, Q, R) Kalman filter over the FULL trajectory
    (all n nodes observed, Sybils included) and turn standardized
    innovations into a per-node sybil_score.

    Under the honest null model, each standardized innovation
    z_{t,c,i} = e_{t,c,i} / sqrt(S_{t,c,ii}) should behave like a standard
    Gaussian, so sum_z2 = sum_{t,c} z_{t,c,i}^2 over a node's k = T*d terms
    should behave like a chi-square(k) variate (mean k, variance 2k).
    A naive Sybil can violate this in *either* direction: a large,
    evidence-driven belief swing inflates sum_z2 above k, while the
    naive-Sybil's actual failure mode -- clamping to one target and barely
    moving round to round -- makes it *more* predictable than an honest
    node and drives sum_z2 *below* k. So the anomaly score is the
    two-sided, variance-normalized deviation from the chi-square(k) null:

        sybil_score_i = |sum_z2_i - k_i| / sqrt(2 * k_i)

    Returns (scores, labels, per_claim) where `scores`/`labels` are flat
    arrays pooled over every (claim, node) pair, and `per_claim` keeps the
    per-claim breakdown for inspection/export.
    """
    F, Q = build_F_Q(params["theta"], params["kappa"], params["sigma2"], n)
    P0 = params["P0_scale"] * np.eye(n)
    x0 = np.zeros(n)

    all_scores, all_labels, per_claim = [], [], []
    for traj in trajectories:
        T = traj.X.shape[0]
        mask_all = np.ones((T, n), dtype=bool)
        z2_sum = np.zeros(n)
        n_terms = np.zeros(n)
        for c in range(N_CHANNELS):
            Y = traj.X[:, :, c]
            R_diag = np.full(n, params["r2"][c])
            _, innov, svar = kalman_run(Y, mask_all, F, Q, R_diag, x0, P0)
            z2 = innov**2 / svar
            z2_sum += np.nansum(z2, axis=0)
            n_terms += np.sum(~np.isnan(z2), axis=0)

        k = np.maximum(n_terms, 1)
        node_scores = np.abs(z2_sum - k) / np.sqrt(2 * k)
        per_claim.append(
            {
                "index": traj.index,
                "claim": traj.claim,
                "names": traj.names,
                "is_sybil": traj.is_sybil.tolist(),
                "sybil_score": node_scores.tolist(),
            }
        )
        all_scores.extend(node_scores.tolist())
        all_labels.extend(traj.is_sybil.astype(int).tolist())

    return np.array(all_scores), np.array(all_labels), per_claim


def evaluate_auc(scores: np.ndarray, labels: np.ndarray) -> dict:
    if roc_auc_score is None:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    if labels.min() == labels.max():
        # degenerate (all-honest or all-Sybil) split -- AUC undefined
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


# ---------------------------------------------------------------------
# 6. LLM-free synthetic simulator of the naive-Sybil attack, generated
#    directly from the graph-OU model (for unit-testing this pipeline
#    and for augmenting the labeled dataset for a downstream GNN).
# ---------------------------------------------------------------------
def simulate_naive_sybil_dataset(
    n_claims: int,
    F: np.ndarray,
    Q: np.ndarray,
    r2: np.ndarray,
    n: int = N_AGENTS,
    T: int = 4,
    sybil_count: int = 2,
    target_confidence: float = 0.9,
    sybil_jitter: float = 0.02,
    seed: int = 0,
) -> list[ClaimTrajectory]:
    """Generate `n_claims` synthetic trajectories: honest nodes follow the
    graph-OU transition x_t = F x_{t-1} + w_t exactly; Sybil nodes instead
    stay clamped at a fixed (target-label, high-confidence) state plus tiny
    jitter every round, regardless of what their neighbors do -- the
    defining trait of the "naive" Sybil strategy in
    `fever_mas_naive_sybil.py` (never actually updates on peer evidence).
    """
    rng = np.random.default_rng(seed)
    trajectories = []
    for i in range(n_claims):
        is_sybil = np.zeros(n, dtype=bool)
        if sybil_count > 0:
            sybil_idx = rng.choice(n, size=min(sybil_count, n), replace=False)
            is_sybil[sybil_idx] = True

        target_sign = rng.choice([1.0, -1.0])
        target_state = np.array([target_sign * target_confidence, target_confidence])

        X = np.zeros((T, n, N_CHANNELS))
        x_prev = np.zeros((n, N_CHANNELS))
        for t in range(T):
            if t == 0:
                x = rng.normal(scale=0.3, size=(n, N_CHANNELS))
            else:
                x = np.zeros((n, N_CHANNELS))
                for c in range(N_CHANNELS):
                    w = rng.multivariate_normal(np.zeros(n), Q)
                    x[:, c] = F @ x_prev[:, c] + w

            # Sybils ignore the graph-OU dynamics entirely: clamp to a
            # fixed target state with only tiny jitter, every round.
            x[is_sybil] = target_state + rng.normal(
                scale=sybil_jitter, size=(is_sybil.sum(), N_CHANNELS)
            )

            obs_noise = rng.normal(size=(n, N_CHANNELS)) * np.sqrt(r2)[None, :]
            X[t] = x + obs_noise
            x_prev = x

        trajectories.append(
            ClaimTrajectory(
                index=10_000_000 + i,
                claim=f"<synthetic-{i}>",
                names=list(AGENT_NAMES[:n]),
                is_sybil=is_sybil,
                X=X,
                ground_truth_label="SYNTHETIC",
                target_label=None,
                sybil_count=int(is_sybil.sum()),
            )
        )
    return trajectories


# ---------------------------------------------------------------------
# 7. CLI report: fit on real data, score real held-out claims, and
#    sanity-check the whole pipeline against a synthetic ground truth.
# ---------------------------------------------------------------------
def train_test_split(trajectories: list[ClaimTrajectory], test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trajectories))
    n_test = int(len(idx) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train = [trajectories[i] for i in train_idx]
    test = [trajectories[i] for i in test_idx]
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="fever_mas_naive_sybil_results.jsonl",
        help="Path to fever_mas_naive_sybil.py results jsonl.",
    )
    parser.add_argument("--max-claims", type=int, default=1500)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--simulate-n", type=int, default=500)
    parser.add_argument("--out-scores", default=None, help="Optional jsonl path to dump per-claim scores.")
    args = parser.parse_args()

    print(f"Loading up to {args.max_claims} claims from {args.data} ...")
    trajectories = load_dataset(args.data, max_claims=args.max_claims)
    print(f"Loaded {len(trajectories)} usable claim trajectories.")
    sybil_counts = [t.sybil_count for t in trajectories]
    print(f"Sybil-count distribution: {np.bincount(sybil_counts)}")

    train, test = train_test_split(trajectories, args.test_frac, args.seed)
    print(f"\nTrain claims: {len(train)}  |  Test claims: {len(test)}")

    print("\nFitting honest null graph-OU model on TRAIN (Sybil nodes masked out) ...")
    params = fit_null_model(train, maxiter=args.maxiter)
    print(
        f"\nFitted params: theta={params['theta']:.4f}  kappa={params['kappa']:.4f}  "
        f"sigma2={params['sigma2']:.4f}  r2={params['r2']}"
    )

    print("\nScoring TEST claims (all 4 nodes visible, including Sybils) ...")
    test_scores, test_labels, test_per_claim = score_trajectories(test, params)
    metrics = evaluate_auc(test_scores, test_labels)
    print(
        f"Real-data held-out detection: ROC-AUC={metrics['roc_auc']:.4f}  "
        f"AP={metrics['average_precision']:.4f}  "
        f"(n_nodes={len(test_labels)}, n_sybil={int(test_labels.sum())})"
    )

    if args.simulate_n > 0:
        print(
            f"\nSanity check: simulating {args.simulate_n} synthetic naive-Sybil claims "
            "directly from the fitted graph-OU model ..."
        )
        F, Q = build_F_Q(params["theta"], params["kappa"], params["sigma2"], N_AGENTS)
        synth = simulate_naive_sybil_dataset(
            args.simulate_n, F, Q, params["r2"], seed=args.seed + 1
        )
        synth_scores, synth_labels, _ = score_trajectories(synth, params)
        synth_metrics = evaluate_auc(synth_scores, synth_labels)
        print(
            f"Synthetic sanity-check detection: ROC-AUC={synth_metrics['roc_auc']:.4f}  "
            f"AP={synth_metrics['average_precision']:.4f}  "
            "(should be near-perfect since the synthetic Sybils are maximally naive "
            "by construction; a real-data AUC well below this indicates real naive "
            "Sybils partially blend into the honest-consensus null model)."
        )

    if args.out_scores:
        with open(args.out_scores, "w") as f:
            for row in test_per_claim:
                f.write(json.dumps(row) + "\n")
        print(f"\nWrote per-claim test scores to {args.out_scores}")


if __name__ == "__main__":
    main()
