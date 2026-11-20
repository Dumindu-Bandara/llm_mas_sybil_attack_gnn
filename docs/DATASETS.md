# Datasets

We use the following four datasets in our experiments.

1. Measuring Massive Multitask Language Understanding - MMLU
2. CommonsenseQA - CSQA
3. Grade School Math 8K - GSM8K
4. Fact Extraction and VERification - FEVER

## Measuring Massive Multitask Language Understanding - MMLU

**Website:** [github.com/hendrycks/test](https://github.com/hendrycks/test), [huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu)

**Description:** MMLU is a multiple-choice benchmark of a text model's accuracy across 57 subjects. Each question has four answer choices (A–D) and one correct answer.

**Splits:** Counts are from the `all` configuration on Hugging Face.

$\color{red}{\text{Check the split/question count. Seems incorrect}}$

| Split | Questions | Purpose |
|---|---:|---| 
| Dev | 285 | Few-shot examples (5 per subject) |
| Validation | 1,531 | Validation |
| Test | 14,042 | Evaluation |

## CommonsenseQA - CSQA

**Website:** [github.com/jonathanherzig/commonsenseqa](https://github.com/jonathanherzig/commonsenseqa), [huggingface.co/datasets/tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)

**Description:** CommonsenseQA is a multiple-choice question answering dataset that targets commonsense knowledge. Target concepts that share a semantic relation with a source concept were drawn from ConceptNet, and crowd-workers wrote questions that mention the source concept. Each question has five answer choices (A–E), one correct and four distractors. The paper reports 12,247 questions; the release on Hugging Face contains 12,102.

**Splits:** Counts are for the random split, the main evaluation split. The dataset also has a question token split.

| Split | Questions | Notes |
|---|---:|---|
| Train | 9,741 | |
| Validation | 1,221 | |
| Test | 1,140 | Answers withheld |

## Grade School Math 8K - GSM8K1

**Website:** [github.com/openai/grade-school-math](https://github.com/openai/grade-school-math), [huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)

**Description:** GSM8K contains about 8.5K human-written grade school math word problems. Each takes 2 to 8 steps to solve using basic arithmetic. Each record has a `question` and a step-by-step `answer`, which includes calculator annotations (`<<...>>`) and ends with the final numeric answer after `####`.

**Splits:**

| Split | File | Problems |
|---|---|---:|
| Train | `train.jsonl` | 7,473 |
| Test | `test.jsonl` | 1,319 |

`train_socratic.jsonl` and `test_socratic.jsonl` contain the same number of problems in a variant format, where each solution step is preceded by an automatically generated subquestion. `example_model_solutions.jsonl` contains model solutions for the 1,319 test problems.

## Fact Extraction and VERification - FEVER

**Website:** [fever.ai/dataset/fever.html](https://fever.ai/dataset/fever.html)

**Description:** FEVER is a fact-verification dataset of 185,445 claims created by altering sentences from Wikipedia. Annotators, who did not have access to the source sentence, labelled each claim as `SUPPORTS`, `REFUTES` or `NOT ENOUGH INFO`. For `SUPPORTS` and `REFUTES` claims, they also recorded the evidence sentences that justify the label. Inter-annotator agreement was 0.6841 (Fleiss κ). The evidence comes from a pre-processed June 2017 Wikipedia dump.

**Splits:**

| Split | File | Claims | `SUPPORTS` | `REFUTES` | `NOT ENOUGH INFO` |
|---|---|---:|---:|---:|---:|
| Train | `train.jsonl` | 145,449 | 80,035 | 29,775 | 35,639 |
| Shared task dev | `shared_task_dev.jsonl` | 19,998 | 6,666 | 6,666 | 6,666 |
| Shared task test (blind) | `shared_task_test.jsonl` | 19,998 | Withheld | Withheld | Withheld |

`shared_task_dev.jsonl` is the concatenation of `paper_dev.jsonl` and `paper_test.jsonl`, the original paper splits. Each has 9,999 claims, with 3,333 per label.

## Citations for Datasets

```
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@inproceedings{talmor-etal-2019-commonsenseqa,
    title = "{C}ommonsense{QA}: A Question Answering Challenge Targeting Commonsense Knowledge",
    author = "Talmor, Alon  and
      Herzig, Jonathan  and
      Lourie, Nicholas  and
      Berant, Jonathan",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1421",
    doi = "10.18653/v1/N19-1421",
    pages = "4149--4158",
    archivePrefix = "arXiv",
    eprint        = "1811.00937",
    primaryClass  = "cs",
}

@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}

@inproceedings{Thorne18Fever,
    author = {Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
    title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VERification}},
    booktitle = {NAACL-HLT},
    year = {2018}
}
```