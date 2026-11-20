# Graph OU LLM-MAS

## SOTA

Cateogry 1: Needs training data. Supervised/Unsupervised methods

| Method | Method | Metrics |
| --- | --- | --- |
| G-Safeguard | Graph attention on MA utterence graph, edge pruning | |
| BlindGuard | Removing supervision via direction noise injection, contrastive learning | |
| XG-Guard | Bi-level token/sentence level encoding, contrastive score fusion | |
| Prop-Guard | Dual-view spatio temperal graph, RL inspector | |
| SentilNet | Credit-based detector for augmented debate trajectories, elimination of bottom K agents | |
| Guardian | Treats as discrete time temporal graph, unsupervised encoder-decoder with information bottleneck abstraction | |

Category 2: Attribute and repair

| Method | Method | Metrics |
| --- | --- | --- |
| Node contribution backprop | Signed DAG backprop | |
| Sentence level verification | splits responses into sentences and audits with an LLM verifier | |
| ARGUS | Training free, scored edges by topology importance, channel frequency, information relevance | |
| InfaGuard | Rehabilitate Infected Agents | |
| SAIGuard | Simulate and santise communication state | |
| OpenEvoShield | Co-evolve against a drifting attacker. | |

Attacks

- Prompt Injection
- Memory Poison
- Tool Injection

Datasets

1. Extract datasets,(Script based).
2. Mini datasets for each type for quick testing.
3. Test accuracies on mini datasets.
4. Graph OU Method Implementation as a defense.
5. Improvement on task accuracy.
