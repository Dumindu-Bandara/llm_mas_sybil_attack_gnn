import json
import numpy as np
import asyncio
from typing import Optional
# Adjacency Matrix
# Initialize agent (memory, role, system_prompt, model_type)
# Initial round


# Initialize agents
# Initial response
# Votes
# Tally votes
# If winner exists?

# Rounds (if winner is none)
# Broadcast the recap
# Updated verdict
# Votes
# Count and tally votes
# winner (Loop back)

# If winner is not none after max rounds
# Plurality winner

# return {claim, final_label, final_output, rounds_run, vote_history, final_tally}


# AgentGraph
# Adjacency matrix (1-N)
# List of Agents (1-N)
# Broadcast
# Collect responses

# Fever_Verify_Claim
# take agent graph
# Braodcaset 1 (task)
# collect -> tally - quoram -> winner? or not

# if not recap -> tally -> quoram recap for N rounds

# fianly count winners


# def neighbour_recap(graph: AgentGraph, i: int, verdicts: list[dict]) -> str:
#     lines = [f"- Agent {j}: answer={verdicts[j]['answer']}; reason={verdicts[j]['reason']}"
#              for j in graph.neighbors(i)]
#     peers = "\n".join(lines) or "(no neighbours)"
#     return (f"Verdicts from agents you are connected to:\n{peers}\n\n"
#             f"Your previous verdict: answer={verdicts[i]['answer']}; reason={verdicts[i]['reason']}\n"
#             "Give your updated verdict.")


# def majority_only_recap(graph: AgentGraph, i: int, verdicts: list[dict]) -> str:
#     from collections import Counter
#     counts = Counter(verdicts[j]["answer"] for j in graph.neighbors(i))
#     return f"Your neighbours voted: {dict(counts)}. Give your updated verdict."


# main
# Load fever json file  data = [json.loads(line) for line in f]
# correct = 0 at initially
# Result file initialization.
# Create the evidence claim
# evidence_text = " ".join(sample["evidence_text"])
# print(f"\nClaim {i+1}/{len(data)}: {claim}")
# print(f"Number of evidence sentences: {len(sample['evidence_text'])}")
# print(f"Evidence: {evidence_text}")
# print(f"Ground truth label: {sample['label']}")
# results run_verify_claim
# Get the final output. If matches correct else incorrect.
# Write to the result file
# out = {
#     "claim": claim,
#     "evidence": evidence_text,
#     "ground_truth_label": gt_label,
#     "final_label": final_label,
#     "final_output": result['final_output'],
#     "rounds_run": result['rounds_run'],
#     "final_tally": result['final_tally'],
#     "correct": final_label == gt_label
# }
# My suggestion:
# Take and agent graph as the input.
# Create a dataloader class
# Run on the agentgraph.
# quoram, tally, pluraity
# claim at the end

from graph_ou.agent import Agent, AgentGraph
from pydantic import BaseModel
from typing import Literal
from collections import Counter

def build_recap_prompt(graph: AgentGraph, i: int, verdicts: list[dict]) -> str:
    def format_response(v: dict) -> str:
        answer = v["answer"] if v["answer"] is not None else "INVALID"
        return f"answer={answer}; reason={v['reason']}"

    peers = "\n".join(
        f"- Agent {j}: {format_response(verdicts[j])}" for j in graph.neighbors(i)
    ) or "(no neighbours)"

    return (
        f"Your previous verdict: {format_response(verdicts[i])}\n\n"
        f"Verdicts from the agents you are connected to:\n{peers}\n\n"
        "Reconsider the claim against the evidence. Your peers' verdicts may be wrong; "
        "change your answer only if their reasoning is supported by the evidence. "
        "Give your updated verdict using the structured schema."
    )

class FeverVerdict(BaseModel):
    reason: str  # brief public justification shared with agents.
    answer: Literal[
        "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"
    ]  # fixed output label set for FEVER

class FeverDataset:
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FeverClaimVerifier:
    def __init__(self, data_path: str, agentgraph: AgentGraph, log_path: str|None = None) -> None:
        self.dataset = FeverDataset(data_path)
        self.correct = 0
        self.agentgraph = agentgraph
        self.log_path = log_path
        # create a logger
        # Add these parameters
        self.max_rounds = 3
        self.quorum = 3

    async def eval_fever(self):

        for i in range(len(self.dataset)):
            sample = self.dataset[i]

            result = await self.verify_claim(sample)

            # TODO: Add Ground Truth Label Comparison and Metrics here.

        # if result == gt correct +1 else 0
        pass

    async def verify_claim(self, sample):
        # add logging (print if logging true)
        claim = sample["claim"]
        evidence = " ".join(sample["evidence_text"])

        task_prompt = (
            f"Claim: {claim}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Give your independent verdict using the structured schema."
        )

        round0_replies = await self.agentgraph.broadcast(task_prompt)
        # recap format???
        round0_replies = [self.agentgraph.parse_verdict(reply[0]) for reply in round0_replies]
        tally = self.count(round0_replies)
        winner = self.check_quorum(tally)

        round_no = 0
        prev_replies = round0_replies

        # NOTE: Testing
        # winner = None

        while winner is None and round_no < self.max_rounds:
            round_no += 1

            # prev_round replies
            replies = await self.agentgraph.recap(prev_replies)
            replies = [self.agentgraph.parse_verdict(reply[0]) for reply in replies]
            tally = self.count(replies)
            winner = self.check_quorum(tally)

            prev_replies = replies

            # NOTE: Testing
            # winner = None

        if winner is None: 
            ranked = tally.most_common()
            top_label, top_count = ranked[0]
            if len(ranked) > 1 and ranked[1][1] == top_count:
                winner = "NOT_ENOUGH_INFO"
            else:
                winner = top_label

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

    adjacency_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0]
    ])

    system_prompt = (
        "You are a fact verification agent. "
        "Task: verify a claim against evidence sentences from Wikipedia "
        "(FEVER-style fact verification). Always answer using the "
        "required structured schema. Base your verdict ONLY on the "
        "supplied evidence text, not on prior/world knowledge about "
        "whether the claim is true in reality."
    )

    agents = [Agent(system_prompt, "openai/gpt-oss-20b", FeverVerdict) for _ in range(4)]
    
    agentgraph = AgentGraph(
        adjacency_matrix=adjacency_matrix,
        agents=agents,
        build_recap_prompt=build_recap_prompt
    )

    fever_verifier = FeverClaimVerifier(fever_jsonl_path, agentgraph)

    asyncio.run(fever_verifier.eval_fever())

if __name__ == "__main__":
    main()


