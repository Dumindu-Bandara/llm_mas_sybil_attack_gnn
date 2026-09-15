import asyncio
import os
import re
from openai import OpenAI, AsyncOpenAI
from typing import Literal

def vllm_invoke(prompt, model_type: str):
    client = OpenAI(
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    response = client.chat.completions.create(
            model=model_type,
            messages=prompt,
            temperature=0,
            max_tokens=4096
        ).choices[0].message.content

    return response

async def avllm_invoke(prompt, model_type: str):
    aclient = AsyncOpenAI(
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    response = await aclient.chat.completions.create(
            model=model_type,
            messages=prompt,
            temperature=0,
            max_tokens=4096
        )

    return response.choices[0].message.content

class Agent: 
    def __init__(self, system_prompt, model_type): 
        self.model_type = model_type
        self.system_prompt = system_prompt 
        self.memory = []
        self.memory.append({"role": "system", "content": system_prompt})
        self.role = "normal"

    def parser(self, response):
        # Regex patter is used to catch <REASON>: ... / <ANSWER>: ... style responses
        splits = re.split(r'<[A-Z_ ]+>: ', str(response).strip())
        splits = [s for s in splits if s]
        if len(splits) == 2:
            answer = splits[-1].strip()
            reason = splits[-2].strip()
            self.last_response = {"answer": answer, "reason": reason}

        else:
            self.last_response = {"answer": None, "reason": response}
 
    def chat(self, prompt): 
        user_msg = {"role": "user", "content": prompt}
        self.memory.append(user_msg)
        response = vllm_invoke(self.memory, self.model_type)
        self.parser(response)
        ai_msg = {"role": "assistant", "content": response}
        self.memory.append(ai_msg)
        
        return response
    
    def set_role(self, role: Literal["normal", "attacker"]): 
        self.role = role
    
    def get_role(self):
        return self.role
    
    async def achat(self, prompt): 
        user_msg = {"role": "user", "content": prompt}
        self.memory.append(user_msg)
        response = await avllm_invoke(self.memory, self.model_type)
        self.parser(response)
        ai_msg = {"role": "assistant", "content": response}
        self.memory.append(ai_msg)
        
        return response

async def main():
    system_prompt = "You are a critical thinking agent which analyses set of evidence statements and accept/reject a claim."
    agent = Agent(system_prompt, "openai/gpt-oss-20b")

    agent.set_role("normal")
    print(agent.get_role())

    question_prompt = "1+1 == 3 is this true or false?"
    response = await agent.achat(question_prompt)
    print(response)

if __name__  == "__main__":
    asyncio.run(main())