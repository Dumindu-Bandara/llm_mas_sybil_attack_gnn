import os
import re
import json
from pydantic import BaseModel

from openai import AsyncOpenAI
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

    # def parser(self, response):
    #     # Regex patter is used to catch <REASON>: ... / <ANSWER>: ... style responses
    #     splits = re.split(r'<[A-Z_ ]+>: ', str(response).strip())
    #     splits = [s for s in splits if s]
    #     if len(splits) == 2:
    #         answer = splits[-1].strip()
    #         reason = splits[-2].strip()
    #         self.last_response = {"answer": answer, "reason": reason}

    #     else:
    #         self.last_response = {"answer": None, "reason": response}
    
    def set_role(self, role: Literal["normal", "attacker"]): 
        self.role = role
    
    def get_role(self):
        return self.role
    
    async def async_chat(self, prompt): 
        user_msg = {"role": "user", "content": prompt}
        self.memory.append(user_msg)
        response = await async_vllm_invoke(self.memory, self.model_type)
        # self.parser(response)
        # ai_msg = {"role": "assistant", "content": response}
        # self.memory.append(ai_msg)
        
        return response

