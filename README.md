# Sybil Attack Defense using Federated GNN on LLM-MAS

## Setup

For this experiments, we will be using `Agentscope` multi agent framework. 
Use `python=3.12` for your environment.

```bash
conda create -n mas_framework_test python=3.12
conda activate mas_framework_test
pip install vllm

# Login to Huggingface and download Qwen/Qwen3-32B-FP8
hf auth login
hf download Qwen/Qwen3-32B-FP8
```

Launch vLLM.
```bash
vllm serve Qwen/Qwen3-32B-FP8 \
  --port 8000 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
```

Other options for vLLM serve.

- `--max-model-len` — cap context to fit comfortably in the B200's 183GB (raise/lower as needed; 32B-FP8 weights are ~32GB so you have plenty of headroom for KV cache).
- `--enable-auto-tool-choice --tool-call-parser hermes` — needed if your agents will call tools (AgentScope/CAMEL toolkits use OpenAI-style function calling).
- `--reasoning-parser qwen3` — separates Qwen3's `<think>` reasoning traces from the final answer in the API response; drop it if you don't want thinking mode.

Verify vLLM running

Once the server logs show `Uvicorn running on http://0.0.0.0:8000`, check it from another terminal on the same node.

Health check:

```bash
curl http://localhost:8000/health
```

List loaded models:

```bash
curl http://localhost:8000/v1/models
```

Send a chat completion: Use jq in pipe for JSON parsing.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }' | jq
```

Test tool calling (relied on by the haggle agents):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "What is the weather in Boston?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }]
  }'
```

A working setup returns a `tool_calls` array instead of plain text.

GPU sanity check — confirm vLLM loaded weights onto the GPU:

```bash
nvidia-smi
```

You should see a process using tens of GB of the B200's memory (matching the ~32GB FP8 weights plus KV cache).

## Datasets

### Preprocessing FEVER Dataset

```bash
python fever_preprocessing.py --wiki-dir "/blue/prabhat/duminduaelamurem/wd/2026_fall/llm_mas_sybil_attack_gnn/llm_mas_sybil_attack_gnn/datasets/FEVER/wiki-pages" --claims "/blue/prabhat/duminduaelamurem/wd/2026_fall/llm_mas_sybil_attack_gnn/llm_mas_sybil_attack_gnn/datasets/FEVER/shared_task_dev.jsonl" --out "/blue/prabhat/duminduaelamurem/wd/2026_fall/llm_mas_sybil_attack_gnn/llm_mas_sybil_attack_gnn/datasets/FEVER/processed_shared_task_dev.jsonl" --index-db "/blue/prabhat/duminduaelamurem/wd/2026_fall/llm_mas_sybil_attack_gnn/llm_mas_sybil_attack_gnn/datasets/FEVER/wiki_index.sqlite"
```