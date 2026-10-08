import asyncio
import json
import os
from collections import Counter
from typing import Literal, Optional

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "openai/gpt-oss-20b")
VLLM_CONTEXT_SIZE = int(os.environ.get("VLLM_CONTEXT_SIZE", "32768"))
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")


class Verdict(BaseModel):
    label: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"] = Field(
        description="Your verdict on the claim, given ONLY the evidence provided."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Your confidence in this verdict, from 0 to 1.",
    )
    reasoning: str = Field(
        description="1-3 sentences justifying the verdict, citing the evidence directly.",
    )


LABEL_TO_OUTPUT = {
    "SUPPORTS": "True",
    "REFUTES": "False",
    "NOT_ENOUGH_INFO": "Not enough info",
}


# Distinct personas reduce correlated errors -> a better ensemble than asking
# the same prompt 4 times.
PERSONAS = {
    "LiteralMatcher": (
        "You are a literal fact-checker. You only accept a claim as SUPPORTS "
        "or REFUTES if the evidence states it explicitly or through direct, "
        "unambiguous entailment. You are strict about exact wording."
    ),
    "Skeptic": (
        "You are a skeptical verifier. Actively look for ways the evidence "
        "might be insufficient, ambiguous, or only partially related to the "
        "claim before agreeing to a confident SUPPORTS/REFUTES verdict."
    ),
    "ContextReasoner": (
        "You are a contextual reasoner. You may combine multiple evidence "
        "sentences and use ordinary world knowledge to fill small "
        "inferential gaps, but you must never invent facts absent from the "
        "evidence."
    ),
    "DevilsAdvocate": (
        "You are a devil's advocate. For any verdict the discussion seems "
        "to be converging on, actively search for the strongest "
        "counter-argument or missing piece of evidence before accepting it."
    ),
}


def build_agent(name: str, persona: str) -> Agent:
    return Agent(
        name=name,
        system_prompt=(
            f"{persona}\n\n"
            "Task: verify a claim against evidence sentences from Wikipedia "
            "(FEVER-style fact verification). Always answer using the "
            "required structured schema. Base your verdict ONLY on the "
            "supplied evidence text, not on prior/world knowledge about "
            "whether the claim is true in reality, unless your persona "
            "explicitly allows it."
        ),
        model=OpenAIChatModel(
            credential=OpenAICredential(
                api_key=VLLM_API_KEY,
                base_url=VLLM_BASE_URL,
            ),
            model=VLLM_MODEL_NAME,
            stream=False,
            context_size=VLLM_CONTEXT_SIZE,
        ),
        toolkit=Toolkit(),  # no external tools needed; evidence is given directly
    )


def tally(votes: dict[str, Verdict]) -> Counter:
    return Counter(v.label for v in votes.values())


def check_quorum(counts: Counter, quorum: int = 3) -> Optional[str]:
    """Byzantine-style quorum: n=4 agents tolerate f=1 bad vote (n >= 3f+1)."""
    label, n = counts.most_common(1)[0]
    return label if n >= quorum else None


def format_votes(votes: dict[str, Verdict]) -> str:
    lines = []
    for name, v in votes.items():
        lines.append(f"- {name}: {v.label} (confidence {v.confidence:.2f}) - {v.reasoning}")
    return "\n".join(lines)


async def broadcast_recap(agents: list[Agent], text: str) -> None:
    """Agents can only `observe()` plain user/assistant text (raw replies
    carrying tool-call blocks are rejected), so cross-agent visibility is
    achieved via a plain-text recap built from each agent's structured
    verdict, rather than by re-feeding agents' raw reply messages to each
    other.
    """
    recap = UserMsg("Moderator", text)
    await asyncio.gather(*(agent.observe(recap) for agent in agents))


async def verify_claim(
    claim: str,
    evidence: str,
    max_rounds: int = 3,
    quorum: int = 3,
) -> dict:
    agents = [build_agent(name, persona) for name, persona in PERSONAS.items()]

    task_prompt = (
        f"Claim: {claim}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Give your independent verdict using the structured schema."
    )

    # Round 0: no recap is broadcast before this, so votes are fully independent.
    msg0 = UserMsg("user", task_prompt)
    round0_replies = await asyncio.gather(
        *(agent.reply(msg0, structured_schema=Verdict) for agent in agents)
    )
    votes = {
        agent.name: Verdict(**reply.structured_output)
        for agent, reply in zip(agents, round0_replies)
    }

    history = [dict(votes)]
    counts = tally(votes)
    winner = check_quorum(counts, quorum)

    round_no = 0
    if winner is None:
        await broadcast_recap(
            agents,
            "Round 0 (independent) votes:\n"
            + format_votes(votes)
            + "\n\nYou will now see each other's verdicts and reasoning. "
            "Reconsider your own verdict in light of your peers' "
            "arguments, but only change your mind if their "
            "evidence-based reasoning is genuinely more convincing than "
            "your own.",
        )

        while winner is None and round_no < max_rounds:
            round_no += 1
            deliberate_msg = UserMsg(
                "Moderator",
                f"Deliberation round {round_no}: give your updated verdict "
                "using the required schema.",
            )
            replies = await asyncio.gather(
                *(
                    agent.reply(deliberate_msg, structured_schema=Verdict)
                    for agent in agents
                )
            )
            votes = {
                agent.name: Verdict(**reply.structured_output)
                for agent, reply in zip(agents, replies)
            }
            history.append(dict(votes))
            counts = tally(votes)
            winner = check_quorum(counts, quorum)

            if winner is None and round_no < max_rounds:
                await broadcast_recap(
                    agents,
                    f"Round {round_no} votes:\n{format_votes(votes)}\n\n"
                    "No quorum yet (need 3/4 agreement). One more round.",
                )

    # No quorum: fall back to plurality. A tie means irreducible disagreement,
    # so default to NOT_ENOUGH_INFO.
    if winner is None:
        ranked = counts.most_common()
        top_label, top_n = ranked[0]
        if len(ranked) > 1 and ranked[1][1] == top_n:
            winner = "NOT_ENOUGH_INFO"
        else:
            winner = top_label

    return {
        "claim": claim,
        "final_label": winner,
        "final_output": LABEL_TO_OUTPUT[winner],
        "rounds_run": round_no,
        "vote_history": history,
        "final_tally": dict(counts),
    }


# Expects already-resolved evidence text; raw FEVER rows only contain
# (Wiki_Page, sent_id) references and need a retrieval step first.
async def example_main():
    sample = {
        "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
        "evidence": (
            "Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish "
            "actor. He then played Detective John Amsterdam in the "
            "short-lived Fox television series New Amsterdam, as well as "
            "appearing as Faaland in the 2009 Fox television film "
            "Virtuality, originally intended as a pilot."
        ),
    }

    result = await verify_claim(sample["claim"], sample["evidence"])

    print(f"\nClaim: {result['claim']}")
    print(f"Final verdict: {result['final_output']}  (raw label: {result['final_label']})")
    print(f"Rounds needed: {result['rounds_run']}")
    print(f"Final tally: {result['final_tally']}")


def main():
    fever_jsonl_file = "/blue/prabhat/duminduaelamurem/wd/2026_fall/llm_mas_sybil_attack_gnn/llm_mas_sybil_attack_gnn/datasets/FEVER/processed_shared_task_dev.jsonl"

    # TODO: Test with new processed_shared_task_dev.jsonl file.
    with open(fever_jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # NOTE: FEVER data is structured as:
    # data: List[dict]
    # Each sample has:
    # {
    #     "id": str,
    #     "claim": str,
    #     "label": str,  # SUPPORTS | REFUTES | NOT ENOUGH INFO
    #     "evidence_sets": List[List[List[Wiki_Page, sentence_ID]]],
    #     "evidence_text": List[str],
    # }

    # See: https://fever.ai/dataset/fever.html#:~:text=HLT%7D%2C%0A%20%20%20%20year%20%3D%20%7B2018%7D%0A%7D-,Data%20Format,-The%20data%20is for original data format.

    print(len(data), "claims loaded from", fever_jsonl_file)

    correct = 0

    if os.path.exists("ferver_mas_consensus_results.jsonl"):
        os.remove("ferver_mas_consensus_results.jsonl")

    for i, sample in enumerate(data):
        claim = sample['claim']
        # evidence_parts = [
        #     e[2] for e in sample['evidence'][0] if e[2] is not None
        # ]
        # evidence_text = " ".join(evidence_parts)
        evidence_text = " ".join(sample["evidence_text"])
        print(f"\nClaim {i+1}/{len(data)}: {claim}")
        print(f"Number of evidence sentences: {len(sample['evidence_text'])}")
        print(f"Evidence: {evidence_text}")
        print(f"Ground truth label: {sample['label']}")

        result = asyncio.run(verify_claim(claim, evidence_text))

        # LABEL_TO_OUTPUT = {
        #     "SUPPORTS": "True",
        #     "REFUTES": "False",
        #     "NOT_ENOUGH_INFO": "Not enough info",
        # }

        final_label = result['final_label'].upper()
        gt_label =  "_".join(sample['label'].split(" "))

        print(f"Final label: {final_label}")
        print(f"Ground Truth label: {gt_label}")

        if final_label == gt_label:
            print("Correct!")
            correct += 1
        else:
            print("Incorrect!")

        with open("/blue/prabhat/duminduaelamurem/wd/2026_fall/llm_mas_sybil_attack_gnn/llm_mas_sybil_attack_gnn/results/sep7_sep13/ferver_mas_consensus_results.jsonl", "a") as f:
            out = {
                "claim": claim,
                "evidence": evidence_text,
                "ground_truth_label": gt_label,
                "final_label": final_label,
                "final_output": result['final_output'],
                "rounds_run": result['rounds_run'],
                "final_tally": result['final_tally'],
                "correct": final_label == gt_label
            }
            f.write(json.dumps(out) + "\n")

    accuracy = correct / len(data)
    print(f"\nAccuracy on FEVER dev set: {accuracy:.2%} ({correct}/{len(data)})")


if __name__ == "__main__":
    # asyncio.run(main())
    main()
