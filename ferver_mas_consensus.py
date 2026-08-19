"""
Multi-Agent System (MAS) for FEVER Claim Verification with AgentScope
========================================================================

Architecture
------------
- 4 LLM agents, fully connected (all-to-all) via explicit broadcast.
  The installed AgentScope version (see `dependencies/agentscope`) exposes a
  single unified `Agent` class with no `MsgHub`/`InMemoryMemory` any more, so
  the old "auto-broadcast inside a MsgHub" trick is replaced with explicit
  `Agent.observe()` calls that inject a Moderator recap into every agent's
  own context.
- Round 0: each agent independently produces a first verdict via its own
  `reply()` call, with no recap injected beforehand, so there is no
  cross-talk contaminating the initial vote.
- If no supermajority forms, agents enter deliberation rounds: after each
  round a Moderator recap (every agent's label/confidence/reasoning) is
  broadcast to all 4 agents via `observe()`, so each agent revises its vote
  in light of its peers' arguments on the next round.
- Consensus rule: Byzantine-style quorum. With n=4 agents you can only
  safely tolerate f=1 unreliable/hallucinating vote (n >= 3f+1), so a
  verdict is finalized only once >= 3 of 4 agents agree.
- If no quorum forms after `max_rounds`, fall back to plurality; a
  genuine 2-2 split degrades to NOT ENOUGH INFO, since irreducible
  disagreement among independent verifiers is itself evidence that the
  claim is not cleanly decidable from the evidence given.

FEVER uses the labels SUPPORTS / REFUTES / NOT ENOUGH INFO internally;
we map those to True / False / "Not enough info" at the very end.

Install:
    The `dependencies/agentscope` checkout is used directly (editable/local
    install), not the PyPI `agentscope` package - its `Agent`/`OpenAIChatModel`
    API has diverged significantly from the old ReActAgent/MsgHub API.

LLM backend: a local vLLM server exposing an OpenAI-compatible API, e.g.:
    vllm serve Qwen/Qwen3-32B-FP8 \
        --port 8000 \
        --max-model-len 32768 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --reasoning-parser qwen3

Since vLLM's server is OpenAI-compatible, we keep AgentScope's
OpenAIChatModel and just point its OpenAICredential at the local server via
`base_url` (see VLLM_BASE_URL/VLLM_MODEL_NAME below) instead of
api.openai.com. Structured output is produced via tool-calling (a
`_GenerateStructuredOutput` tool AgentScope injects automatically), which is
why vLLM needs `--enable-auto-tool-choice --tool-call-parser hermes`. No
real API key is needed - vLLM doesn't check it by default, so a placeholder
value is used.
"""

import asyncio
import os
from collections import Counter
from typing import Literal, Optional

from pydantic import BaseModel, Field

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit


# ---------------------------------------------------------------------
# 0. Local vLLM server config (OpenAI-compatible endpoint)
#    Started with e.g.:
#      vllm serve Qwen/Qwen3-32B-FP8 --port 8000 --max-model-len 32768 \
#          --enable-auto-tool-choice --tool-call-parser hermes \
#          --reasoning-parser qwen3
# ---------------------------------------------------------------------
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3-32B-FP8")
VLLM_CONTEXT_SIZE = int(os.environ.get("VLLM_CONTEXT_SIZE", "32768"))
# vLLM doesn't check the API key by default; the OpenAI client just needs
# a non-empty string. Override VLLM_API_KEY if you've configured vLLM
# with --api-key.
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")


# ---------------------------------------------------------------------
# 1. Structured verdict schema - every agent must answer in this shape
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 2. Four agents with distinct verification personas
#    (diversity reduces correlated errors -> a better ensemble than
#    asking the same prompt 4 times)
# ---------------------------------------------------------------------
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
            # Point the OpenAI client at the local vLLM server instead of
            # api.openai.com.
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


# ---------------------------------------------------------------------
# 3. Consensus utilities
# ---------------------------------------------------------------------
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
    """Inject a Moderator recap into every agent's own context.

    Stands in for the old MsgHub auto-broadcast: agents can only `observe()`
    plain user/assistant text (raw replies carrying tool-call blocks are
    rejected), so cross-agent visibility is achieved via a plain-text recap
    built from each agent's structured verdict, rather than by re-feeding
    agents' raw reply messages to each other.
    """
    recap = UserMsg("Moderator", text)
    await asyncio.gather(*(agent.observe(recap) for agent in agents))


# ---------------------------------------------------------------------
# 4. The verification pipeline
# ---------------------------------------------------------------------
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

    # ---- Round 0: fully independent votes (no recap yet -> no cross-talk) ----
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

    # ---- Deliberation rounds: all-to-all via explicit recap broadcast ----
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
            # Sent identically to all 4 agents -> a synchronous voting
            # round.
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

    # ---- Fallback: plurality, tie -> NOT_ENOUGH_INFO ----
    if winner is None:
        ranked = counts.most_common()
        top_label, top_n = ranked[0]
        if len(ranked) > 1 and ranked[1][1] == top_n:
            winner = "NOT_ENOUGH_INFO"  # genuine disagreement -> safest default
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


# ---------------------------------------------------------------------
# 5. Example run on a FEVER-style sample
#    (real FEVER JSONL rows look like:
#     {"id": ..., "claim": "...", "label": "SUPPORTS",
#      "evidence": [[[annotation_id, evidence_id, "Wiki_Page", sent_id], ...]]}
#     -- you still need a retrieval step to turn (Wiki_Page, sent_id)
#     pairs into evidence TEXT; this pipeline assumes that's already done
#     and you're passing in resolved evidence sentences as a string.)
# ---------------------------------------------------------------------
async def main():
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


if __name__ == "__main__":
    asyncio.run(main())
