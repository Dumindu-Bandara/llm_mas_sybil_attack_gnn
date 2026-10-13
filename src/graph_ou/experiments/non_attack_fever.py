import asyncio
import json
from collections import Counter
from typing import Literal, TypedDict

import numpy as np
from pydantic import BaseModel

from graph_ou.agent import Agent, AgentGraph

# Next action items summary.
# TODO 1: Input, output, docstrings
# TODO 2: Tests using pytest.
# TODO 3: Logging functionality.
# TODO 4: Sybils vs Naive attack success rates.


def build_recap_prompt(graph: AgentGraph, i: int, verdicts: list[dict]) -> str:
    def format_response(v: dict) -> str:
        answer = v["answer"] if v["answer"] is not None else "INVALID"
        return f"answer={answer}; reason={v['reason']}"

    peers = (
        "\n".join(
            f"- Agent {j}: {format_response(verdicts[j])}" for j in graph.neighbors(i)
        )
        or "(no neighbours)"
    )

    return (
        f"Your previous verdict: {format_response(verdicts[i])}\n\n"
        f"Verdicts from the agents you are connected to:\n{peers}\n\n"
        "Reconsider the claim against the evidence. Your peers' verdicts may be wrong; "
        "change your answer only if their reasoning is supported by the evidence. "
        "Give your updated verdict using the structured schema."
    )


class FeverVerdict(BaseModel):
    reason: str
    answer: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]


class FeverSample(TypedDict):
    id: int
    claim: str
    label: Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    evidence_sets: list[list[list[str | int]]]
    evidence_text: list[str]


class FeverDataset:
    def __init__(self, data_path: str) -> None:
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> FeverSample:
        return self.data[idx]


class FeverClaimVerifier:
    def __init__(
        self, data_path: str, agentgraph: AgentGraph, log_path: str | None = None
    ) -> None:
        self.dataset = FeverDataset(data_path)

        self.agentgraph = agentgraph
        self.log_path = log_path
        # TODO: create a logger
        # TODO: Add these parameters to the class constructer.
        self.max_rounds = 3
        self.quorum = 3

    async def eval_fever(self):

        correct = 0

        for i in range(len(self.dataset)):
            sample = self.dataset[i]

            result = await self.verify_claim(sample)

            if result["label"] == sample["label"]:
                correct += 1

        return {"success_rate": correct / len(self.dataset)}

    async def verify_claim(self, sample: FeverSample):

        # TODO: Add logging (print if logging true)
        claim = sample["claim"]
        evidence = " ".join(sample["evidence_text"])

        task_prompt = (
            f"Claim: {claim}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Give your independent verdict using the structured schema."
        )

        round0_replies = await self.agentgraph.broadcast(task_prompt)
        round0_replies = [
            self.agentgraph.parse_verdict(reply[0]) for reply in round0_replies
        ]
        tally = self.count(round0_replies)
        winner = self.check_quorum(tally)

        round_no = 0
        prev_replies = round0_replies

        while winner is None and round_no < self.max_rounds:
            round_no += 1

            replies = await self.agentgraph.recap(prev_replies)
            replies = [self.agentgraph.parse_verdict(reply[0]) for reply in replies]
            tally = self.count(replies)
            winner = self.check_quorum(tally)

            prev_replies = replies

        if winner is None:
            ranked = tally.most_common()
            top_label, top_count = ranked[0]
            if len(ranked) > 1 and ranked[1][1] == top_count:
                winner = "NOT_ENOUGH_INFO"
            else:
                winner = top_label

        return {
            "label": winner,
            "num_rounds": round_no,
            "tally": ranked,
        }

    def count(self, replies: list[dict]) -> Counter:
        return Counter(r["answer"] for r in replies)

    def check_quorum(self, tally: Counter) -> str | None:
        if not tally:
            return None
        top_label, top_count = tally.most_common(1)[0]
        return top_label if top_count >= self.quorum else None


def main():
    fever_jsonl_path = "/blue/prabhat/duminduaelamurem/wd/2026_fall/llm_mas_sybil_attack_gnn/llm_mas_sybil_attack_gnn/datasets/FEVER/processed_shared_task_dev.jsonl"

    fever_dataset = FeverDataset(fever_jsonl_path)
    print("FEVER dataset loaded")

    adjacency_matrix = np.array(
        [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]]
    )

    system_prompt = (
        "You are a fact verification agent. "
        "Task: verify a claim against evidence sentences from Wikipedia "
        "(FEVER-style fact verification). Always answer using the "
        "required structured schema. Base your verdict ONLY on the "
        "supplied evidence text, not on prior/world knowledge about "
        "whether the claim is true in reality."
    )

    agents = [
        Agent(system_prompt, "openai/gpt-oss-20b", FeverVerdict) for _ in range(4)
    ]

    agentgraph = AgentGraph(
        adjacency_matrix=adjacency_matrix,
        agents=agents,
        build_recap_prompt=build_recap_prompt,
    )

    fever_verifier = FeverClaimVerifier(fever_jsonl_path, agentgraph)

    asyncio.run(fever_verifier.eval_fever())


if __name__ == "__main__":
    main()
