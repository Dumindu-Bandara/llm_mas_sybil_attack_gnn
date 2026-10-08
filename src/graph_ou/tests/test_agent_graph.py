import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_ou_method import AgentGraph


class FakeAgent:
    """Stands in for Agent: records prompts and returns a fixed verdict without calling vLLM."""

    def __init__(self, answer, delay=0.0):
        self.answer = answer
        self.delay = delay
        self.prompts = []

    async def async_chat(self, prompt):
        self.prompts.append(prompt)
        await asyncio.sleep(self.delay)
        return json.dumps({"reason": f"because {self.answer}", "answer": self.answer}), "thinking"


def neighbour_answers_prompt(graph, i, verdicts):
    return f"agent {i} sees " + ",".join(verdicts[j]["answer"] for j in graph.neighbors(i))


# 0 <- 1, 0 <- 2, 1 <- 0, 2 has no incoming edges; the self-loop on 2 must be ignored.
ADJ = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [0, 0, 1],
])


def make_graph(builder=neighbour_answers_prompt):
    agents = [FakeAgent("SUPPORTS"), FakeAgent("REFUTES"), FakeAgent("NOT_ENOUGH_INFO")]
    return AgentGraph(ADJ, agents, build_recap_prompt=builder), agents


def test_init_rejects_mismatched_adjacency_matrix():
    with pytest.raises(AssertionError):
        AgentGraph(np.zeros((2, 2)), [FakeAgent("SUPPORTS")] * 3, build_recap_prompt=neighbour_answers_prompt)


def test_neighbors_uses_incoming_edges_and_skips_self_loops():
    graph, _ = make_graph()

    assert graph.neighbors(0) == [1, 2]
    assert graph.neighbors(1) == [0]
    assert graph.neighbors(2) == []


def test_parse_verdict_valid_and_invalid():
    assert AgentGraph.parse_verdict('{"reason": "r", "answer": "REFUTES"}') == {"reason": "r", "answer": "REFUTES"}
    assert AgentGraph.parse_verdict("not json") == {"answer": None, "reason": "not json"}


def test_broadcast_sends_same_prompt_and_keeps_agent_order():
    agents = [FakeAgent("SUPPORTS", delay=0.03), FakeAgent("REFUTES", delay=0.0), FakeAgent("NOT_ENOUGH_INFO", delay=0.01)]
    graph = AgentGraph(ADJ, agents, build_recap_prompt=neighbour_answers_prompt)

    responses = asyncio.run(graph.broadcast("claim"))

    assert [json.loads(content)["answer"] for content, _ in responses] == ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    assert all(agent.prompts == ["claim"] for agent in agents)


def test_recap_sends_each_agent_its_own_prompt_from_the_builder():
    graph, agents = make_graph()
    previous = [
        ('{"reason": "a", "answer": "SUPPORTS"}', None),
        ('{"reason": "b", "answer": "REFUTES"}', None),
        ('{"reason": "c", "answer": "NOT_ENOUGH_INFO"}', None),
    ]

    responses = asyncio.run(graph.recap(previous))

    assert len(responses) == 3
    assert agents[0].prompts == ["agent 0 sees REFUTES,NOT_ENOUGH_INFO"]
    assert agents[1].prompts == ["agent 1 sees SUPPORTS"]
    assert agents[2].prompts == ["agent 2 sees "]


def test_recap_uses_the_builder_given_to_the_instance():
    graph, agents = make_graph(builder=lambda graph, i, verdicts: f"custom {i}")

    asyncio.run(graph.recap([('{"reason": "r", "answer": "SUPPORTS"}', None)] * 3))

    assert [agent.prompts for agent in agents] == [["custom 0"], ["custom 1"], ["custom 2"]]


def test_recap_rejects_wrong_number_of_responses():
    graph, _ = make_graph()

    with pytest.raises(AssertionError):
        asyncio.run(graph.recap([('{"reason": "r", "answer": "SUPPORTS"}', None)]))
