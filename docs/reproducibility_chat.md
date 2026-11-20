To make the comparison look fair to reviewers, show that every method saw the same inputs, had the same budget, and was scored on the same metrics. Then separate out which part of your gain comes from which part of your method. Here is a protocol built around that.

## 1. One shared harness, with trajectories logged once
- Run every attack scenario once in your own framework and log the full trajectory: messages, embeddings, edges and round index. Then **replay the identical logs to every detector**. This removes LLM sampling noise from the comparison and is the strongest fairness argument you can make.
- Fix everything else across methods:
  - the backbone LLM, the prompts and the seeds
  - the topology and the attacker placement
  - the injection text
  - the sentence encoder, wherever a method uses embeddings
- Run online, end-to-end only for the intervention experiments in step 3, using the same seeds.

## 2. Reproduce first, then compare
Before running anything in your setting, reproduce each baseline **in its own reported setting**. For example, run G-Safeguard on CSQA, MMLU and GSM8K with its own topology and attacker ratio. Report a small table: "reported vs. our reproduction".
- Use the authors' code and default hyperparameters.
- Give every method the same tuning budget.
- If no code exists (likely for SAI-Guard), reimplement it, mark it † in tables, and release your implementation.

## 3. Evaluate detection and intervention separately
The baselines combine different detectors with different interventions (GAT plus edge cut, flagging, corrective agent, and so on). If you only compare end-to-end, reviewers can't tell whether you won on the score or on the intervention.

| Eval | Measures | Metrics |
|---|---|---|
| **A. Detection** (offline, replayed logs) | Score quality | AUROC, AUPRC, **TPR at a fixed FPR (e.g. 5%)**, at both message level and agent level |
| **B. Defense** (online) | Outcome | Accuracy under attack, attack success rate, and **accuracy with no attack** to measure collateral damage |
| **C. Cross-product** | Where the gain comes from | Each detector × {hard node removal, edge cut, your soft reweighting} |

Eval C is what convinces reviewers. If baseline detector plus soft reweighting beats baseline detector plus hard removal, your intervention contribution stands on its own. If your score plus hard removal still beats the baselines, your detector contribution stands on its own.

## 4. Neutralise the threshold advantage
Your conformal threshold is one of your contributions, so comparing your adaptive threshold against their fixed one mixes two effects. Do three things:
- Report **threshold-free metrics** (AUROC/AUPRC) for all methods.
- Give each baseline an **oracle-tuned global threshold** chosen on a validation split. This is their best case.
- Add a "**baseline + conformal**" variant: apply your calibration to their scores. Then report your method's realised FPR against the target α, which shows the guarantee holds in practice.

If you still win when baselines get conformal calibration, the gain is from the OU score itself. If they improve a lot with it, that's still a win: the calibration contribution is general.

## 5. Match the information budget per supervision type
- **Supervised (G-Safeguard, XG-Guard):** report two regimes.
  - *In-distribution:* train on the same dataset, topology and backbone as the test set. This is their upper bound, and you may lose here. Say so honestly.
  - *Transfer:* train on, say, star / 8 agents / backbone A and test on random / 20–50 agents / backbone B. This is where your "no retraining" claim actually lives.
  - Optionally add a curve of performance against the number of labelled trajectories.
- **Unsupervised (BlindGuard, Guardian):** give them **exactly the same benign warm-up data** your method uses to estimate Θ̂, Σ̂ and the calibration set. No more, no less.
- **Training-free (ARGUS, Node backprop.):** no fitting is needed. Make sure all three methods see the same rounds.

## 6. Make the timing difference measurable, not just claimed
Per-message, per-round and post-hoc methods decide at different moments, so add:
- **Detection latency:** the round or message at which the attacker is first flagged.
- **Infection count:** the number of benign agents that consumed the injected message before it was flagged.

To stay fair to post-hoc methods (Node backprop.), also run a **final-answer-only** evaluation in which every method may decide at the end. This shows you aren't winning only because of the timing rule.

## 7. Cover the baselines' home turf and your weak spots
- **Scale:** run 8, 20 and 50 agents so you meet BlindGuard and SAI-Guard on their scale.
- **Attacker ratio:** sweep it as G-Safeguard does, e.g. 1/8, 2/8 and 3/8.
- **All five topologies × all four datasets.**
- **An adaptive attacker that knows the OU model.** Reviewers will ask for this. Include:
  - slow drift that stays below τ each round
  - injection in round 0, which contaminates the warm-up
  - an attacker that mimics the attractor

  Report all baselines under these attacks too.

## 8. Cost and statistics
- **Cost:** report LLM calls, tokens and wall-clock per query. ARGUS (a corrective agent) and SAI-Guard (regeneration) need extra LLM calls, while your method is a matrix operation per message. This is a legitimate secondary advantage.
- **Statistics:** use at least 3 seeds and report 95% confidence intervals. Use paired tests across queries, because every method sees the same queries.

## Suggested table layout for the paper
1. **Main table:** datasets × topologies, with AUROC and accuracy under attack for {G-Safeguard, XG-Guard, BlindGuard, Guardian, ARGUS, SAI-Guard, Node backprop., Ours}.
2. **Transfer table:** supervised baselines in-distribution vs. transferred, next to Ours.
3. **Ablation table:** detector × intervention cross-product, and baseline ± conformal.
4. **Latency/infection plot** and a **scale plot** (accuracy against agent count).
5. **Adaptive attacker table.**
6. **Appendix:** the reproduction table and a cost table.

## Order of work
Build trajectory logging with embeddings into your existing agent-graph framework first. It already logs messages, so this is mostly adding the embedding and edge metadata. After that, reproduce G-Safeguard: once your harness matches its reported numbers, every later comparison inherits that credibility.