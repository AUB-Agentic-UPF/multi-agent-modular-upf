
# Multi-Agent Modular UPF Control Framework (Evaluation)

This framework implements a multi-agent pipeline that translates high-level operator intent into:

1. functional requirements (FR) and non-functional requirements (NFR) extracted via the Intent Handling Function (IHF), also referred to as the Intent Handler,
2. PCC rules via the Policy Creator,
3. a selected module chain + profiles via the Selector,
4. PFCP-inspired data-plane rules (PDR/FAR/QER/URR structures)
5. a deployment plan via the Executor,
6. an assessment of whether the intent requirements are satisfied via the Monitor.
   
The default `main.py` script runs a fixed set of intents for evaluation and prints each agent’s JSON output plus upstream metrics (time + token usage).

Each agent operates as an independent decision module with structured JSON interfaces, enabling traceability across the intent-to-deployment pipeline.

---

## Requirements

* Python 3.10+ (recommended)
* `langchain-openai`
* `langchain-core`

If you don’t have a `requirements.txt` yet, you can install the main dependencies with:

```bash
pip install langchain-openai langchain-core
```

---

## Setup: API Key (required)

You must export your OpenAI API key before running.

**macOS / Linux:**

```bash
export OPENAI_API_KEY="your_key_here"
```

**Windows (PowerShell):**

```powershell
setx OPENAI_API_KEY "your_key_here"
```

Close and reopen the terminal after `setx` so the environment variable is available.

---

## Run

From the project folder:

```bash
python main.py
```

---

## Model Selection (change if you want)

Each agent defines its own model inside its Python file:

* `IHF.py`
* `policy_creator.py`
* `selector.py`
* `configurator.py`
* `executor.py`
* `monitor.py`

  
To change a model, edit the line that looks like:

```python
ChatOpenAI(model="...", temperature=0.0)
```

Examples:

* `model="gpt-4.1"`
* `model="gpt-5-mini"`


---

## Monitor Agent (separate evaluation)

The Monitor agent is evaluated separately and is **not** included in `main.py`.

* Monitor evaluation logic lives in: `monitor_evaluation.py`
* This is intentional: `main.py` reports upstream stages only:
  **IHF → Policy Creator → Selector → Configurator → Executor**

---

## Project Structure

* `main.py` — runs the evaluation loop over all intents (excluding Monitor)
* `IHF.py` — parses raw intent into structured JSON containing FR and NFR
* `policy_creator.py` — builds PCC rules (UL/DL/bidirectional)
* `selector.py` — selects modules + deployment profiles
* `configurator.py` — creates PFCP-like rules (PDR/FAR/QER/URR)
* `executor.py` — produces a deployment plan for a cloud orchestrator.
* `monitor.py` — Monitor agent (checks NFR compliance, orchestrator feedback, etc.)
* `monitor_evaluation.py` — Monitor agent evaluation (separate from `main.py`)
* `usage_utils.py` — token usage extraction helpers


---

## Notes

* Each agent is expected to output **valid JSON**.
* `main.py` enforces a confidence threshold (default: 65). If confidence is below the threshold, it will ask whether to continue.
