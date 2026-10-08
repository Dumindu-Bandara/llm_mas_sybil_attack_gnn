import os
import re
import json
import asyncio
from abc import abstractmethod
from pydantic import BaseModel

from openai import AsyncOpenAI
from collections.abc import Callable
from typing import Literal

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

class Verdict(BaseModel):
    reason: str # brief public justification shared with agents.
    answer: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"] # fixed output label set for FEVER

async def async_vllm_invoke(prompt: list[dict], model_type: str):
    async_openai_client = AsyncOpenAI(
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    response = await async_openai_client.chat.completions.create(
            model=model_type,
            messages=prompt,
            seed=0, #NOTE temperature=1.0 and top_p=1.0 recommended by OpenAI 
            temperature=0,
            max_tokens=4096,
            reasoning_effort="medium", # low | medium | high
            response_format={
                "type": "json_schema",
                "json_schema" : {"name": "verdict", "schema": Verdict.model_json_schema()}
            }
        )

    msg = response.choices[0].message
    thinking = getattr(msg, "reasoning", None)

    return msg.content, thinking

class Agent: 
    def __init__(self, system_prompt, model_type): 
        self.model_type = model_type
        self.system_prompt = system_prompt 
        self.memory = []
        self.memory.append({"role": "system", "content": system_prompt})
        self.role = "normal"
    
    def set_role(self, role: Literal["normal", "attacker"]): 
        self.role = role
    
    def get_role(self):
        return self.role
    
    async def async_chat(self, prompt): 
        user_msg = {"role": "user", "content": prompt}
        self.memory.append(user_msg)
        response = await async_vllm_invoke(self.memory, self.model_type)
        return response


RecapPromptBuilder = Callable[["AgentGraph", int, list[dict]], str]
class AgentGraph:

    def __init__(self, adjacency_matrix, agents: list[Agent], build_recap_prompt: RecapPromptBuilder):
        self.adjacency_matrix = adjacency_matrix
        self.agents = agents
        self.build_recap_prompt = build_recap_prompt

        assert self.adjacency_matrix.shape[0] == len(agents)

    def neighbors(self, i: int) -> list[int]:
        # Row i = the agents that agent i listens to (A[i][j] != 0 means j -> i) (incoming edges)
        n = len(self.agents)
        return [j for j in range(n) if j != i and self.adjacency_matrix[i][j] != 0]

    @staticmethod
    def parse_verdict(content: str) -> dict:
        try:
            return Verdict.model_validate_json(content).model_dump()
        except Exception:
            return {"answer": None, "reason": str(content)}

    async def broadcast(self, prompt):
        tasks = [asyncio.create_task(agent.async_chat(prompt)) for agent in self.agents]
        # gather returns results in input order, so responses[k] belongs to self.agents[k]
        return await asyncio.gather(*tasks)

    async def recap(self, responses):
        assert len(responses) == len(self.agents)
        verdicts = [self.parse_verdict(content) for content, _thinking in responses]

        tasks = [
            asyncio.create_task(agent.async_chat(self.build_recap_prompt(self, i, verdicts)))
            for i, agent in enumerate(self.agents)
        ]
        return await asyncio.gather(*tasks)

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