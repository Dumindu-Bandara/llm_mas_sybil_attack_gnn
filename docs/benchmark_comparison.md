The draft lists four benchmarks: prompt injection on CSQA, MMLU, GSM8K and FEVER, across five topologies with varying attacker ratios (§1). It promises supervised, unsupervised and training-free baselines. Everything below is judged only on what the draft says about each related-work paper. Where the draft doesn't say which datasets a paper used, I've rated it on its method and threat model and marked that as an inference.

## Overall relevance as a baseline (weakest to strongest)

| Tier | Paper | Why (from the draft) |
|---|---|---|
| **Not a baseline** | NetSafe | It's "analysis only" with no detector, so there's nothing to compare against. It is still the reference for your setup: the same topologies, plus CSQA, GSM8K and FACT. |
| **Not a baseline** | SentinelAgent | An LLM oversight agent shown on an email system built with Magentic-One. It's a different task, and it isn't even in Table 1. |
| **Not a baseline** | OpenEvoShield | Its threat model is a drifting attacker handled by continual co-evolution, at 4–5 agents. It answers a different question from detecting a single injection. |
| **Weak** | InfaGuard | Its intervention is replacing and rehabilitating agents, so it would be scored on recovery rather than detection. |
| **Weak** | Sentence rectification | It targets cooperative attacks and uses an LLM verifier, which is a different cost class. It also only acts post hoc. |
| **Weak–Medium** | PropGuard | It's a strong RL-trained system at 8 agents with 3 attackers, but recovering subgraphs is a different output from per-message scores. The citation is still a TODO. |
| **Weak–Medium** | SentinelNet | It's trained on adversarial debate trajectories and removes the bottom-K agents. It's comparable on detection, but the hard removal makes the intervention comparison uneven. |
| **Medium** | XG-Guard | It's a supervised-style (contrastive) graph detector that flags per round. It's the natural second supervised baseline after G-Safeguard. |
| **Medium** | Node backprop. | It's training-free like yours but post hoc. It's useful to show that detecting earlier matters, but it's a weak accuracy competitor at 5 agents with 1 attacker. |
| **Strong** | Guardian | It's conceptually the closest: it uses earlier timesteps as the normality reference and scores incrementally. Your §2.3 says you unify its idea, so reviewers will expect a head-to-head. |
| **Strong** | BlindGuard | It's the main label-free baseline, and at 20–50 agents with 3 attackers it's the scale benchmark. It directly tests your claim that you need neither attack labels nor training. |
| **Strong** | ARGUS | It's the only other training-free method that decides per round. Your §2.3 calls it "closest in spirit", so it's the key comparison for your training-free claim. |
| **Strong** | SAI-Guard | It's the only other method marked **per message** in Table 1, at 8–80 agents with 3 attackers. It directly challenges the caption's claim that you are "the only" per-message method. The citation is still a TODO. |
| **Strongest** | G-Safeguard | It has the same attack type (prompt injection), three of your four datasets (CSQA, MMLU, GSM8K), varying attacker ratios and topological intervention. It is the de facto benchmark setup you're inheriting. |

## Per benchmark (weakest to strongest)

### CSQA
Weakest first: OpenEvoShield, InfaGuard, sentence rectification, PropGuard, SentinelNet, Node backprop., XG-Guard, ARGUS, Guardian, SAI-Guard, BlindGuard, **G-Safeguard**.
- **G-Safeguard** is strongest because the draft explicitly says it evaluates prompt injection on CSQA, so you can reproduce its exact setup.
- NetSafe also used CSQA, so use its topology/attack protocol as a sanity reference rather than as a baseline.
- **BlindGuard and SAI-Guard** come next because CSQA is multiple choice. An injection flips a discrete option, a setting where agent-level graph anomaly detectors are known to work well, so these are the label-free competitors you must beat.
- ARGUS sits lower here because its stated focus is misinformation (MISINFOTASK), not commonsense multiple choice.

### MMLU
Weakest first: OpenEvoShield, InfaGuard, sentence rectification, PropGuard, Node backprop., XG-Guard, SentinelNet, ARGUS, Guardian, SAI-Guard, BlindGuard, **G-Safeguard**.
- **G-Safeguard** is strongest because it's the only paper in the draft stated to use MMLU. NetSafe did not use MMLU.
- **SentinelNet** moves up relative to CSQA because it's built on debate trajectories, and multi-choice knowledge debate is the setting debate-style systems are usually designed for (inferred from its method description).
- MMLU is broad in subject, so per-query conformal calibration should help most here: "hard contested" queries vary a lot. That makes it the benchmark where your **fixed-threshold** argument against the per-round flaggers (XG-Guard, BlindGuard) is strongest.

### GSM8K
Weakest first: OpenEvoShield, InfaGuard, PropGuard, Node backprop., ARGUS, SentinelNet, Guardian, BlindGuard, SAI-Guard, sentence rectification, XG-Guard, **G-Safeguard**.
- **G-Safeguard** is strongest because the draft lists it as evaluating on GSM8K, and NetSafe used GSM8K too.
- **XG-Guard and sentence rectification move up sharply.** In math, an injected wrong number can be a tiny change in sentence-embedding space. XG-Guard's **token-level** encoding and sentence rectification's step-by-step verifier can catch that change, while a sentence-embedding OU residual may miss it. I expect GSM8K to be your weakest benchmark, and these two are the most dangerous competitors there.
- ARGUS drops because it's built around misinformation, not arithmetic.

### FEVER
Weakest first: OpenEvoShield, InfaGuard, PropGuard, BlindGuard, XG-Guard, Guardian, Node backprop., SAI-Guard, G-Safeguard, SentinelNet, sentence rectification, **ARGUS**.
- **No paper in the draft is stated to evaluate on FEVER**, so you'll have to run every baseline yourself. Your current code (`non_attack_fever.py`, `prompt_injection_fever.py`) already targets it.
- **ARGUS** is strongest because injected claims on FEVER behave like misinformation, which is exactly its stated target.
- NetSafe's FACT dataset is the closest prior setup.
- **Sentence rectification** is next because a sentence-level verifier is a natural fit for checking claims against evidence.
- **SentinelNet** ranks high because its factual-alignment loss is designed for factual consistency.
- G-Safeguard stays relevant to keep results comparable with the other three benchmarks.
- FEVER is where you can argue that the method generalises beyond the G-Safeguard suite. It's also where reviewers will ask why ARGUS's MISINFOTASK wasn't used as well.

## Practical recommendation
The smallest set that covers the "supervised, unsupervised and training-free" promise in §1:
- **Supervised:** G-Safeguard, plus XG-Guard (the latter needed mainly for GSM8K).
- **Unsupervised:** BlindGuard and Guardian.
- **Training-free:** ARGUS, plus Node backprop. to show the cost of deciding post hoc.
- **Per message:** SAI-Guard. You need to verify its citation first, because Table 1's "only entry" claim rests on how it compares.

Two caveats from the draft itself:
- Six references (PropGuard, InfaGuard, SAI-Guard, OpenEvoShield, SentinelAgent, sentence rectification) are still `TODO` and unverified. Don't commit to them as baselines until the citations and code availability are confirmed.
- Their per-dataset coverage above is inferred from method descriptions. I didn't read the original papers, so it's worth checking.