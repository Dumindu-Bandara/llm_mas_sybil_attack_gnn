import asyncio
import os

import pytest

import graph_ou.agent
from graph_ou.agent import Agent, Verdict


def test_init_sets_system_prompt_and_defaults():
    agent = Agent(
        system_prompt="You are a fact checker.",
        model_type="test-model",
        verdict=Verdict,
    )

    assert agent.model_type == "test-model"
    assert agent.system_prompt == "You are a fact checker."
    assert agent.memory == [{"role": "system", "content": "You are a fact checker."}]
    assert agent.get_role() == "normal"


def test_set_role():
    agent = Agent("sys", "test-model", Verdict)

    agent.set_role("attacker")
    assert agent.get_role() == "attacker"

    agent.set_role("normal")
    assert agent.get_role() == "normal"


def test_async_chat_appends_user_message_and_calls_llm(monkeypatch):
    calls = []

    async def fake_invoke(prompt, model_type, verdict):
        calls.append((list(prompt), model_type, verdict))
        return '{"reason": "r", "answer": "SUPPORTS"}', "thinking"

    monkeypatch.setattr(graph_ou.agent, "async_vllm_invoke", fake_invoke)

    agent = Agent("sys", "test-model", Verdict)
    response = asyncio.run(agent.async_chat("Is the sky blue?"))

    assert response == ('{"reason": "r", "answer": "SUPPORTS"}', "thinking")
    assert agent.memory == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Is the sky blue?"},
    ]
    assert calls == [(agent.memory, "test-model", Verdict)]


@pytest.mark.skipif(
    os.getenv("RUN_VLLM_TESTS") != "1",
    reason="needs a running vLLM server; set RUN_VLLM_TESTS=1 to enable",
)
def test_async_chat_with_live_vllm():
    agent = Agent("sys", os.getenv("VLLM_MODEL", "openai/gpt-oss-20b"), Verdict)
    content, _thinking = asyncio.run(agent.async_chat("Is the sky blue?"))

    verdict = Verdict.model_validate_json(content)
    assert verdict.answer in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
    assert agent.memory == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Is the sky blue?"},
    ]
