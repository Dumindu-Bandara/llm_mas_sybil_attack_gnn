import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graph_ou_method
from graph_ou_method import Agent


def test_init_sets_system_prompt_and_defaults():
    agent = Agent(system_prompt="You are a fact checker.", model_type="test-model")

    assert agent.model_type == "test-model"
    assert agent.system_prompt == "You are a fact checker."
    assert agent.memory == [{"role": "system", "content": "You are a fact checker."}]
    assert agent.get_role() == "normal"


def test_set_role():
    agent = Agent("sys", "test-model")

    agent.set_role("attacker")
    assert agent.get_role() == "attacker"

    agent.set_role("normal")
    assert agent.get_role() == "normal"


def test_async_chat_appends_user_message_and_calls_llm(monkeypatch):
    calls = []

    # async def fake_invoke(prompt, model_type):
    #     calls.append((list(prompt), model_type))
    #     return '{"reason": "r", "answer": "SUPPORTS"}', "thinking"

    # monkeypatch.setattr(graph_ou_method, "async_vllm_invoke", fake_invoke)

    # agent = Agent("sys", "test-model")
    agent = Agent("sys", "openai/gpt-oss-20b")

    response = asyncio.run(agent.async_chat("Is the sky blue?"))

    assert response == ('{"reason": "r", "answer": "SUPPORTS"}', "thinking")
    assert agent.memory == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Is the sky blue?"},
    ]
    assert calls == [(agent.memory, "test-model")]
