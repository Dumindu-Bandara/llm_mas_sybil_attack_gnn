"""
Multi-Agent System (MAS) for FEVER Claim Verification with AgentScope
========================================================================

Architecture
------------
- 4 LLM agents, fully connected (all-to-all) via AgentScope's MsgHub.
- Round 0: each agent independently produces a first verdict, OUTSIDE the
  MsgHub, so there is no cross-talk contaminating the initial vote.
- If no supermajority forms, agents enter deliberation rounds INSIDE a
  MsgHub: every agent's reply is auto-broadcast to every other agent
  (true all-to-all, no manual message routing code needed), so each
  agent revises its vote in light of its peers' arguments.
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
    pip install agentscope

Requires an LLM API key in the environment, e.g.:
    export OPENAI_API_KEY=...
(swap OpenAIChatModel/OpenAIChatFormatter below for DashScopeChatModel,
AnthropicChatModel, etc. if you use a different provider - AgentScope
supports many out of the box.)
"""

import asyncio
import os
from collections import Counter
from typing import Literal, Optional

from pydantic import BaseModel, Field

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub
from agentscope.tool import Toolkit


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


def build_agent(name: str, persona: str) -> ReActAgent:
    return ReActAgent(
        name=name,
        sys_prompt=(
            f"{persona}\n\n"
            "Task: verify a claim against evidence sentences from Wikipedia "
            "(FEVER-style fact verification). Always answer using the "
            "required structured schema. Base your verdict ONLY on the "
            "supplied evidence text, not on prior/world knowledge about "
            "whether the claim is true in reality, unless your persona "
            "explicitly allows it."
        ),
        model=OpenAIChatModel(
            model_name="gpt-4o-mini",  # swap for any AgentScope-supported model
            api_key=os.environ["OPENAI_API_KEY"],
            stream=False,
        ),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
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

    # ---- Round 0: fully independent votes (no MsgHub -> no cross-talk) ----
    msg0 = Msg("user", task_prompt, "user")
    round0_replies = await asyncio.gather(
        *(agent(msg0, structured_model=Verdict) for agent in agents)
    )
    votes = {agent.name: Verdict(**reply.metadata) for agent, reply in zip(agents, round0_replies)}

    history = [dict(votes)]
    counts = tally(votes)
    winner = check_quorum(counts, quorum)

    # ---- Deliberation rounds: all-to-all via MsgHub ----
    round_no = 0
    if winner is None:
        async with MsgHub(
            participants=agents,
            announcement=Msg(
                "Moderator",
                "Round 0 (independent) votes:\n"
                + format_votes(votes)
                + "\n\nYou will now see each other's verdicts and reasoning. "
                "Reconsider your own verdict in light of your peers' "
                "arguments, but only change your mind if their "
                "evidence-based reasoning is genuinely more convincing than "
                "your own.",
                "system",
            ),
        ) as hub:
            while winner is None and round_no < max_rounds:
                round_no += 1
                deliberate_msg = Msg(
                    "Moderator",
                    f"Deliberation round {round_no}: give your updated verdict "
                    "using the structured schema.",
                    "system",
                )
                # Sent identically to all 4 agents -> a synchronous voting
                # round. Their replies auto-broadcast to each other via
                # MsgHub as soon as each is generated.
                replies = await asyncio.gather(
                    *(agent(deliberate_msg, structured_model=Verdict) for agent in agents)
                )
                votes = {agent.name: Verdict(**reply.metadata) for agent, reply in zip(agents, replies)}
                history.append(dict(votes))
                counts = tally(votes)
                winner = check_quorum(counts, quorum)

                if winner is None and round_no < max_rounds:
                    # Explicit recap keeps peer votes legible even though
                    # they were already auto-broadcast as structured replies.
                    await hub.broadcast(
                        Msg(
                            "Moderator",
                            f"Round {round_no} votes:\n{format_votes(votes)}\n\n"
                            "No quorum yet (need 3/4 agreement). One more round.",
                            "system",
                        )
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
    import agentscope

    agentscope.init()  # optional: sets up logging/studio

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