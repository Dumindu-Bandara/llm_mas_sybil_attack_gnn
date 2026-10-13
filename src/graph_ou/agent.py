import asyncio
import os
from collections.abc import Callable
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

import numpy as np


class Verdict(BaseModel):
    reason: str  # brief public justification shared with agents.
    answer: Literal[
        "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"
    ]  # fixed output label set for FEVER


async def async_vllm_invoke(
    prompt: list[dict[str, str]], model_type: str, verdict: type[BaseModel]
) -> tuple[str | None, str | None]:
    async_openai_client = AsyncOpenAI(
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    response = await async_openai_client.chat.completions.create(
        model=model_type,
        messages=prompt,
        seed=0,  # NOTE temperature=1.0 and top_p=1.0 recommended by OpenAI
        temperature=0,
        max_tokens=4096,
        reasoning_effort="medium",  # low | medium | high
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": verdict.model_json_schema()},
        },
    )

    msg = response.choices[0].message
    thinking = getattr(msg, "reasoning", None)

    return msg.content, thinking


class Agent:
    def __init__(
        self,
        system_prompt: str,
        model_type: str,
        verdict: type[BaseModel],
        role: Literal["normal", "attacker"] = "normal",
    ) -> None:
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.role = role
        self.verdict = verdict

        self.memory: list[dict[str, str]] = []
        self.memory.append({"role": "system", "content": self.system_prompt})

    def set_role(self, role: Literal["normal", "attacker"]) -> None:
        self.role = role

    def get_role(self) -> Literal["normal", "attacker"]:
        return self.role

    def reset(self) -> None:
        self.memory = []
        self.memory.append({"role": "system", "content": self.system_prompt})

    async def async_chat(self, prompt: str) -> tuple[str | None, str | None]:
        user_msg = {"role": "user", "content": prompt}
        self.memory.append(user_msg)
        response = await async_vllm_invoke(self.memory, self.model_type, self.verdict)
        return response


RecapPromptBuilder = Callable[["AgentGraph", int, list[dict]], str]


class AgentGraph:
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        agents: list[Agent],
        build_recap_prompt: RecapPromptBuilder,
    ) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.agents = agents
        self.build_recap_prompt = build_recap_prompt

        assert self.adjacency_matrix.shape[0] == len(agents)

    async def broadcast(self, prompt: str) -> list[tuple[str | None, str | None]]:
        tasks = [asyncio.create_task(agent.async_chat(prompt)) for agent in self.agents]
        # gather returns results in input order, so responses[k] belongs to self.agents[k]
        return await asyncio.gather(*tasks)

    async def recap(self, responses: list[dict]) -> list[tuple[str | None, str | None]]:
        assert len(responses) == len(self.agents)
        # verdicts = [self.parse_verdict(content) for content, _thinking in responses]

        tasks = [
            asyncio.create_task(
                agent.async_chat(self.build_recap_prompt(self, i, responses))
            )
            for i, agent in enumerate(self.agents)
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    def parse_verdict(content: str | None) -> dict[str, str | None]:
        try:
            return Verdict.model_validate_json(content).model_dump()
        except Exception:
            return {"answer": None, "reason": str(content)}

    def neighbors(self, i: int) -> list[int]:
        # Row i = the agents that agent i listens to (A[i][j] != 0 means j -> i) (incoming edges)
        n = len(self.agents)
        return [j for j in range(n) if j != i and self.adjacency_matrix[i][j] != 0]

    def reset(self) -> None:
        for agent in self.agents:
            agent.reset()
