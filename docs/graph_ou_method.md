Your Section 3 is still a placeholder, so this explanation uses Eq. 1 and the descriptions in the Abstract, Introduction and §2. Where the draft doesn't give formulas (discretisation, estimation, calibration, reweighting), I've filled in the standard derivation that matches what the draft describes, and marked it as mine. I also checked `src/graph_ou/`: it has the agent, graph and experiment scaffolding, but the OU detector isn't implemented yet.

---

## Graph OU method

### 1. The idea
Most graph defenses (G-Safeguard, XG-Guard, BlindGuard, Guardian…) learn what an attack looks like. Graph OU does the opposite: it models what **benign collaboration** looks like as a dynamical process. Anything that process can't explain is flagged.

The observation behind it: in a clean multi-agent system, each agent's answer drifts toward a weighted mix of what its neighbours said. It then wobbles around that consensus, with the wobble coming from decoding randomness. That pattern, pulled toward a moving target plus bounded noise, is exactly an **Ornstein–Uhlenbeck (mean-reverting) process**.

### 2. The model (Eq. 1)
Let $X_i(t) \in \mathbb{R}^d$ be the embedding of agent $i$'s utterance at round $t$.

$$dX_i = -\Theta\big(X_i - \mu_i(t)\big)\,dt + \Sigma\, dW_i, \qquad \mu_i(t) = \sum_{j \in \mathcal{N}(i)} a_{ij} X_j(t)$$

| Symbol | Meaning |
|---|---|
| $\mu_i(t)$ | **Attractor**: the neighbourhood consensus. It's a graph diffusion over in-neighbours, so the topology enters the model directly. |
| $a_{ij}$ | Edge weights (e.g. row-normalised adjacency). These are also what soft reweighting changes later. |
| $\Theta$ | **Contraction rate**: how quickly an agent moves toward its neighbours. |
| $\Sigma$ | **Benign fluctuation scale**: normal disagreement and sampling noise. |
| $dW_i$ | Brownian noise. |

The "graph-coupled" part is that each agent's target depends on the other agents' states, so the whole system is one coupled stochastic process on the communication graph.

### 3. Closed-form one-step prediction *(my derivation)*
Debate runs in discrete rounds. Hold $\mu_i$ fixed over one step $\Delta$ and the OU process has an exact Gaussian transition:

$$X_i(t{+}1) \mid \mathcal{F}_t \sim \mathcal{N}\Big(\underbrace{\mu_i(t) + e^{-\Theta\Delta}\big(X_i(t) - \mu_i(t)\big)}_{\hat X_i(t+1)},\; Q\Big), \quad Q = \int_0^\Delta e^{-\Theta s}\Sigma\Sigma^\top e^{-\Theta^\top s}ds$$

In the scalar/isotropic case ($\Theta = \theta I$, $\Sigma = \sigma I$), $Q = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta\Delta})\,I$.

So before agent $i$ speaks, you can already predict where its next utterance should land and how much spread is normal. In discrete form it is a VAR(1)-style update:

$$X_i(t{+}1) = A\,X_i(t) + (I - A)\,\mu_i(t) + \varepsilon, \qquad A = e^{-\Theta\Delta}$$

### 4. Innovation residual = the detection score
When agent $i$ emits a message, embed it and compare it with the prediction:

$$r_i(t{+}1) = X_i(t{+}1) - \hat X_i(t{+}1), \qquad s_i(t{+}1) = r_i^\top Q^{-1} r_i \quad \text{(Mahalanobis)}$$

- **Benign:** the agent moved roughly toward its neighbours, $r$ is noise-sized, and $s$ is small.
- **Injection:** the agent jumped in a direction its in-neighbourhood can't explain, leaving a large "innovation" and a large $s$.

**Timing:** the score is computed at emission time, before downstream agents read the message. That is the "per message" entry in Table 1 and the contrast with post-hoc backprop and attribution methods.

A useful property you could make explicit in the paper: a benign agent that gets *persuaded* by an attacker is still following OU dynamics, because its attractor now contains the attacker's state. The residual therefore points at the **source** of the deviation, not at everyone it later infects.

### 5. Estimating Θ and Σ without training *(my derivation)*
There's no offline training. For each deployment (query × topology), fit from the system's own early rounds:
- Regress the increments $X_i(t{+}1) - X_i(t)$ on $(X_i(t) - \mu_i(t))$. Least squares gives $\hat A$, and from it $\hat\Theta$.
- Take the covariance of the regression residuals to get $\hat Q$ (and so $\hat\Sigma$).

$d$ is the size of a sentence embedding (hundreds of dimensions), so in practice you'll want a restricted form: scalar or diagonal $\Theta$, a shrinkage covariance for $Q$, or a PCA projection first.

### 6. Conformal threshold instead of a global ε
Collect benign calibration scores $s_1, \dots, s_n$ for this query and topology, choose a target false-alarm rate $\alpha$, and set

$$\tau = \text{the } \lceil (n+1)(1-\alpha) \rceil\text{-th smallest } s_k$$

If benign residuals are exchangeable, then $P(s_{\text{new}} > \tau) \le \alpha$. Easy, convergent queries get a tight threshold. Hard, contested queries (where real disagreement is large) get a loose one. Equivalently, report a conformal p-value $p = \frac{1 + \#\{s_k \ge s_{\text{new}}\}}{n+1}$.

### 7. Soft edge reweighting
Because $s$ is continuous, you don't have to delete the agent. Instead, scale down the influence of its outgoing edges, for example

$$a_{ji} \leftarrow a_{ji} \cdot \exp\!\big(-\lambda \max(0,\, s_i - \tau)\big)$$

or use $p_i$ as a multiplicative weight. This affects what neighbours are shown or told to trust, and it also feeds back into the attractor $\mu_j$ for the next prediction. A suspicious agent is down-weighted only while it deviates, and its benign contributions in other rounds are kept.

### 8. Per-round loop
```
round 0: agents answer independently → embeddings X_i(0)
early rounds: fit Θ̂, Q̂; build conformal calibration set
for each round t, for each message from agent i:
    μ_i  = Σ_j a_ij X_j(t)
    X̂_i  = μ_i + Â (X_i(t) − μ_i)
    s_i  = Mahalanobis(embed(msg) − X̂_i; Q̂)
    if s_i > τ (or p_i < α): flag, decay a_·i ∝ (s_i − τ)
    deliver message with updated edge weights
```

---

## Gaps reviewers will likely probe (for when you write §3)
1. **Clean warm-up assumption.** Θ, Σ and the calibration set come from early rounds. If the attacker injects in round 0 or 1, you're calibrating on contaminated data. You need a robust estimator (median or trimmed fits, leave-one-agent-out) or a cross-query prior.
2. **Exchangeability is not automatic.** Under contraction, the spread shrinks round by round, so raw residuals across rounds aren't identically distributed. Normalising by $Q$ helps, or calibrate per round index. State this assumption precisely, since the conformal guarantee depends on it.
3. **Stealthy attackers.** An attacker that moves slowly toward the wrong answer can stay inside $\tau$ each round. Consider a cumulative score (CUSUM over residuals), or flag agents whose estimated per-agent $\hat\theta_i \approx 0$, meaning "stubborn" agents that never revert.
4. **Embedding choice.** The method is only as good as the embedding's sensitivity to semantic and answer changes. An ablation across encoders would help.
5. **Small n.** With 3–8 agents and a few rounds, the calibration set per query is tiny, and conformal with $n \approx 10$ gives a coarse threshold. Pooling across agents within the query is probably required.

If you want, I can draft Section 3 in LaTeX from this structure, or start implementing the OU scorer in `src/graph_ou/` alongside `agent.py`.