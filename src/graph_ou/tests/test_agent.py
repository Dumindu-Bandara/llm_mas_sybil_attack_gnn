import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graph_ou_method
from graph_ou_method import Agent, Verdict


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

    async def fake_invoke(prompt, model_type):
        calls.append((list(prompt), model_type))
        return '{"reason": "r", "answer": "SUPPORTS"}', "thinking"

    monkeypatch.setattr(graph_ou_method, "async_vllm_invoke", fake_invoke)

    agent = Agent("sys", "test-model")
    response = asyncio.run(agent.async_chat("Is the sky blue?"))

    assert response == ('{"reason": "r", "answer": "SUPPORTS"}', "thinking")
    assert agent.memory == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Is the sky blue?"},
    ]
    assert calls == [(agent.memory, "test-model")]


@pytest.mark.skipif(
    os.getenv("RUN_VLLM_TESTS") != "1",
    reason="needs a running vLLM server; set RUN_VLLM_TESTS=1 to enable",
)
def test_async_chat_with_live_vllm():
    agent = Agent("sys", os.getenv("VLLM_MODEL", "openai/gpt-oss-20b"))
    content, _thinking = asyncio.run(agent.async_chat("Is the sky blue?"))

    verdict = Verdict.model_validate_json(content)
    assert verdict.answer in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
    assert agent.memory == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Is the sky blue?"},
    ]
