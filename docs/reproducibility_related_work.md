Your `related_work/` folder has 6 of the 7 medium-to-strongest baselines. The design I proposed works with them, with one main change: **use G-Safeguard's `dataset.json` as the shared trajectory format** instead of the schema I invented. G-Safeguard, BlindGuard and XG-Guard all read that format directly, so those three need almost no adapter code. I inspected the code but haven't run any of these repos yet.

## What's in `related_work/`

| Repo | Paper | Usable as baseline? |
|---|---|---|
| `G-safeguard` | G-Safeguard (strongest) | ✅ |
| `BlindGuard` | BlindGuard (strong) | ✅ It also includes **Dominant, TAM and PREM**, three extra unsupervised graph anomaly-detection baselines you get for free. |
| `XG-Guard` | XG-Guard (medium) | ✅ |
| `GUARDIAN` | Guardian (strong) | ✅ |
| `ARGUS` | ARGUS (strong) | ⚠️ The scoring logic has to be pulled out of its simulator (see below). |
| `BPD` | Node backprop. (medium) | ✅ |
| `STAR` | Sentence rectification (weak) | Optional, cheap to add |
| `NetSafe` | Analysis only | Not a detector. Reuse its topology generator and the `fact.jsonl` dataset. |
| `SADOG-Demo` | DID-based Sybil resistance on a blockchain | ❌ Unrelated to prompt injection. Leave it out. |
| — | **SAI-Guard** | ❌ **Missing.** You'd have to reimplement it, and its citation is still TODO. |

## Compatibility per repo

| | Input it expects | Encoder | Continuous score available? | Built-in decision rule | Intervention | Adapter effort |
|---|---|---|---|---|---|---|
| **G-Safeguard** | `dataset.json` | MiniLM-L6 (384-d) | Yes: `sigmoid(gnn(...))`, but only the thresholded result is saved | ≥ 0.5 | Drops a flagged agent's messages for the rest of the run | **Small** |
| **BlindGuard** (+Dominant/TAM/PREM) | same | MiniLM-L6 | Yes (`message`, `anomaly_scores`), but discarded | **top-k with k=3** | same | **Small** |
| **XG-Guard** | same, plus `summary` and `keywords` fields for training | MiniLM-L6, sentence **and token** embeddings | Yes (`get_score_overall`), but discarded | top-k with k=3 | same | **Medium** |
| **GUARDIAN** | a list of `Data(text=[...], edge_index, adj)` per round | `bert-base-uncased`, trained per query | Yes: `detect()` returns scores | argmax: **always removes 1 node** | Deactivates the node | **Small** |
| **BPD** | Its own `EdgeGraph` (flat/hierarchy only) | LLM judge per edge | Yes: agent scores | ε = 1.5 | Re-runs with the detected agent "switched to benign" | **Medium** |
| **STAR** | Text of each reply | LLM sentence factuality check | Yes: `analysis_score` | ≥ 0.3 | Rewrites the message | **Small** |
| **ARGUS** | Its own sandbox (921-line `agent.py`, MisinfoTask) | OpenAI embeddings API plus an LLM goal-inference step | Yes: edge scores in `edge_finder.py` | top-k edges | Corrective agent | **Medium–Large** |

## Incompatibilities to handle (these are what would make a comparison silently unfair)

1. **The adjacency matrix is transposed.** G-Safeguard, BlindGuard and XG-Guard use `A[src, dst]`, so in-edges are a **column** (`adj_matrix[:, idx]`). Your `AgentGraph.neighbors` uses `A[i][j] != 0 ⇒ j → i`, so in-edges are a **row**. Convert in exactly one function and cover it with a unit test. On symmetric topologies (chain, tree, complete) a bug here is invisible; it only shows up on directed random graphs.

2. **The attack model is different.** G-Safeguard's "prompt injection" is an **LLM attacker with an adversarial system prompt** that writes persuasive arguments. Your `InjectionAgent` is **scripted** and uses `fake_system` / `ignore_previous` payloads. To reproduce their numbers you need their attacker. For your own story you need both, so make attacker type a run dimension and add an LLM-persuasion attacker to your framework.

3. **Message format.** Theirs is a raw `"<REASON>: … <ANSWER>: X"` string. Yours is JSON `{"reason", "answer"}`. Pick one rendering for the embedded text, for example `"<REASON>: {reason} <ANSWER>: {answer}"`, and use it for **every** method, including yours.

4. **Fixed encoder and dimensions.** All three GNN repos hard-code MiniLM-L6 (`in_channels=384`). BlindGuard and XG-Guard also pad edge features to 1536 = 4 turns × 384, so **at most 4 turns (initial + 3 rounds)**. Use MiniLM-L6 and ≤ 3 debate rounds for the main table. An encoder ablation then means retraining those baselines.

5. **Most decision rules use information a real defender wouldn't have.**
   - BlindGuard and XG-Guard use top-k with k=3, and their test sets have exactly 3 attackers, so k is the true attacker count.
   - GUARDIAN always removes the top-1 node, even when no attacker is present.
   - BPD's repair switches the detected agent to benign mode, which isn't possible against a real attacker.

   That's why the harness must **save the continuous scores** and apply one shared threshold policy (oracle global and conformal) to every method.

6. **BlindGuard's AUROC isn't a real AUROC.** `cal_mean_AUROC` puts **binary top-k predictions** into `roc_auc_score`. Recompute AUROC for every method from the saved scores.

7. **G-Safeguard trains on different node features than it tests on.**
   - In training, `x` = embeddings of the **system prompts** (`gen_training_dataset.py` → `data.py`), and the attacker's system prompt differs from everyone else's. That leaks the label.
   - Online, `x` = the mean of the turn-0 message embeddings.

   Reproduce it as-is first. Then also run a consistent message-based-`x` version, and report which one your main table uses.

8. **XG-Guard training needs a `summary` and `keywords` per sample, and the repo has no script that generates them.** Its pre-generated `datasets_local` and `datasets_online` cover CSQA, MMLU and GSM8K. For FEVER, write a small LLM prompt that produces them. Its online inference doesn't need them.

9. **Topology assumptions.**
   - GUARDIAN builds a fully connected graph internally. Pass your own `edge_index`/`adj` instead.
   - BPD only supports flat/hierarchy structures. Build its `EdgeGraph` from your trajectory: round r, sender j → round r+1, receiver i.
   - G-Safeguard's test script covers random, chain, tree and star. "Complete" = random with sparsity 1.0.

10. **ARGUS is tied to its sandbox.** Its scoring is betweenness centrality (0.2) + message frequency (0.2) + relevance (0.6). Relevance is the similarity between message sentences and an **LLM-inferred** "hypothesized misinfo goal", so the goal is not given by the dataset. `get_embedding` calls the OpenAI embeddings API, which your vLLM gpt-oss server won't serve. Reimplement those three terms (about 100 lines) over your trajectories, copy the goal-inference prompt from `sandbox/agent.py`, use MiniLM for relevance, and mark it † in tables.

11. **FEVER setups differ.** GUARDIAN's FEVER uses the claim only, with no evidence, as an A/B/C choice over 100 random dev claims. Yours supplies evidence. Use yours, and say so in the paper.

12. **Environments clash.** ARGUS uses Python 3.9, STAR 3.8, G-Safeguard 3.10, and XG-Guard pins torch 2.5.1 / PyG 2.6.1. Don't merge them into `mas_framework_test`. Give each baseline its own conda env and have adapters talk to them **through files**: trajectories JSON in, scores JSONL out.

13. **Every repo, and your code, uses `temperature=0`.** Same issue as before: seeds won't produce different trajectories.

## What needs to be implemented

### In your repo (`src/graph_ou/`)

| # | Item | Notes | Size |
|---|---|---|---|
| 1 | **Trajectory export in G-Safeguard `dataset.json` format** | Save full replies in `count()`. Write `adj_matrix` (transposed to their convention), `attacker_idxes`, `system_prompts`, `question`, `correct_answer`, `wrong_answer`, `communication_data`, plus your extra metadata (dataset, topology, seed, attacker type). | ~80 lines |
| 2 | **LLM-persuasion attacker** | Port G-Safeguard's attacker prompt (`gen_graph.py`, `agents.py`) as a second attacker type next to `InjectionAgent`. | ~40 |
| 3 | **Task interface plus CSQA, MMLU and GSM8K tasks** | Reuse G-Safeguard's `gen_csqa.py`, `gen_mmlu.py` and `gen_gsm8k.py` loaders and splits. | ~150 |
| 4 | **Topology generator** | chain, star, tree, complete, random. Copy `get_adj_matrix` (G-Safeguard) or `generate_adj` (NetSafe) **so the topologies are identical to theirs**. | ~40 |
| 5 | `embed.py` | MiniLM-L6 sentence embeddings, plus token embeddings for XG-Guard. | ~50 |
| 6 | `detectors/graph_ou.py` | Your method. | your core work |
| 7 | `scores` table, `metrics.py`, `thresholds.py` | AUROC, AUPRC, TPR@FPR, latency; oracle and conformal thresholds. | ~120 |
| 8 | Edge-weight hook `W` in `AgentGraph`, plus `interventions.py` | hard-remove, edge-cut, soft-reweight. | ~60 |
| 9 | Unit tests | Adjacency transpose round-trip, prefix-invariance (causality), schema validation. | ~80 |

### Adapters (one file each, run in that baseline's own env)

| Baseline | What the adapter does | Patch to the baseline repo? |
|---|---|---|
| G-Safeguard | Load the checkpoint, call `embeddings2graph` + `gnn` per round prefix, write `sigmoid` scores | None: import its functions |
| BlindGuard (SCL) + Dominant/TAM/PREM | Same; write `-message` or `anomaly_scores` per round | None |
| XG-Guard | Same, using `run_ours_inference`; for FEVER training, generate `summary`/`keywords` with an LLM (~40 lines) | None |
| GUARDIAN | Build a per-round `Data(text, edge_index, adj)` list from your trajectory, call `train_model` + `detect` per prefix | None |
| BPD | Build an `EdgeGraph` from your trajectory, call `get_edge_score` per edge (LLM calls), then `backward_propagate` and `aggregate_agent_scores`; replace the MMLU-only `CHOICE_RE` answer extractor | Small: generalize answer extraction |
| STAR | Call `analyze_sentences_factuality` per message and take the max `analysis_score` per agent | Point `llm_api.py` at your vLLM |
| ARGUS | **Reimplement** the edge scoring (betweenness, frequency, relevance to the LLM-inferred goal) over trajectories; aggregate edge scores to source-node scores | No reuse beyond prompts (†) |
| SAI-Guard | Reimplement from the paper once the citation is verified | n/a (†) |

Each adapter is about 50–100 lines. The rule is: import the baseline's own functions, never edit its model code, and save continuous scores. For the online defense runs, the adapters plug into your `AgentGraph` loop through the same interface, so you don't reuse their `main_defense*.py` scripts. You use those scripts only for the **reproduction check** in their own setting.

### Reproduction runs (no new code)
Run G-Safeguard, BlindGuard and XG-Guard as-is on XG-Guard's shipped `datasets_online/PI/{csqa,mmlu,gsm8k}` test sets. Then run the same samples through your adapters and confirm the numbers match. Note that BlindGuard already ships checkpoints under `PI/checkpoint*`.

## Decisions you need to make
1. **Backbone LLM.** The baselines' published numbers use gpt-4o-mini. Your runs use gpt-oss-20b on vLLM. Either reproduce with gpt-4o-mini, which needs an API key, or reproduce with gpt-oss-20b and say clearly that this is not their reported setting.
2. **Main attacker type.** LLM persuasion (their setting), scripted injection (yours), or both.
3. **SAI-Guard.** Reimplement it, or soften Table 1's claim to "the only training-free per-message method".

I'd start with items 1, 4 and the G-Safeguard adapter. Once your FEVER trajectories load into G-Safeguard's own `embeddings2graph` without errors, the format question is settled for three baselines at once.