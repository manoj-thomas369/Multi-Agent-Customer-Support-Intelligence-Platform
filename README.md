# Multi-Agent Customer Support Intelligence Platform

An AI-powered multi-agent system that automatically understands, classifies,
responds to, and routes e-commerce customer support tickets (delivery,
refunds, payments, product issues, account access).

## Architecture

```
User Ticket
   |
   v
Intake Agent          -> cleans text, extracts intent / sentiment / entities
   |
   v
Classification Agent  -> predicts category + priority (TF-IDF + Logistic Regression)
   |
   v
Retrieval Agent (RAG) -> embeds the ticket and searches a FAISS knowledge base
   |
   v
Response Agent        -> drafts a grounded reply from the retrieved resolution
   |
   v
Escalation Agent       -> routes low-confidence / angry / unresolved tickets to a human
   |
   v
Logging + Learning Agent -> persists the outcome and folds confident resolutions
                             back into the knowledge base
```

`src/orchestrator.py` runs these agents in sequence and is called from a
FastAPI backend (`src/api/main.py`); a Streamlit chat UI (`app.py`) is the
front end.

### Design choices

- **No dataset was supplied for this project.** `scripts/generate_dataset.py`
  synthesizes a labeled ticket dataset (`data/tickets_dataset.csv`) from
  category-specific templates plus a small FAQ knowledge base
  (`data/faq_knowledge_base.csv`), so the whole pipeline is trainable and
  runnable end-to-end without external data.
- **Embeddings**: the Retrieval Agent uses `sentence-transformers`
  (`all-MiniLM-L6-v2`) over FAISS. If the model can't be downloaded (no
  internet), it automatically falls back to a TF-IDF embedding so the system
  still runs fully offline.
- **Response generation** is retrieval-grounded templating by default, so no
  LLM API key is required to run the project. If `OPENAI_API_KEY` is set (see
  `.env.example`), the Response Agent will use it to polish the draft's tone.
- **Continuous learning**: confident, non-escalated resolutions are folded
  back into the FAISS index at runtime (`LearningAgent`), so the knowledge
  base grows from real resolved tickets over time, per the "Learning Agent"
  requirement in the brief.
- **Orchestration** is a simple, explicit Python pipeline rather than
  CrewAI/LangGraph, since each agent's interface (`Agent.run(TicketContext)`)
  is a thin, testable unit — a CrewAI/LangGraph wrapper could be dropped in
  around the same agents later without changing their internals.

## Project layout

```
app.py                     Streamlit chat UI
src/
  agents/                  IntakeAgent, ClassificationAgent, RetrievalAgent,
                            ResponseAgent, EscalationAgent, LearningAgent
  api/                      FastAPI app + request/response schemas
  db/                       SQLAlchemy models + session management (SQLite)
  ml/                       TF-IDF + Logistic Regression training/inference
  nlp/                      Sentiment (VADER) + rule-based entity/intent extraction
  retrieval/                Embedding backends + FAISS-backed vector store
  orchestrator.py           Wires the agents into one pipeline
  state.py                  TicketContext: the object threaded through agents
scripts/
  generate_dataset.py       Synthesizes tickets_dataset.csv + faq_knowledge_base.csv
  train_classifier.py       Trains + saves the category/priority models
  build_vector_index.py     Builds the initial FAISS index from the FAQ dataset
tests/                      pytest suite (fully offline, no trained artifacts required)
```

## Opening in VS Code

The `.vscode/` folder ships run/debug configurations, so once you've created
the venv and installed dependencies (below), open this folder in VS Code and
use the **Run and Debug** panel (`Ctrl+Shift+D`) instead of typing commands:

- **Backend: FastAPI (uvicorn)** - starts the API on port 8000
- **Frontend: Streamlit** - starts the chat UI
- **Full stack (backend + frontend)** - starts both together
- **Scripts: generate_dataset.py / train_classifier.py / build_vector_index.py** - reruns the one-time setup steps
- **Tests: pytest** - runs the test suite

VS Code will prompt to install the recommended Python extension if it isn't
already installed, and is pre-configured to use `.venv` and discover tests
under `tests/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to configure `API_BASE_URL` or an
optional `OPENAI_API_KEY`.

### 1. Generate data, train models, build the knowledge base

```bash
python scripts/generate_dataset.py
python scripts/train_classifier.py
python scripts/build_vector_index.py
```

### 2. Run the backend

```bash
uvicorn src.api.main:app --reload
```

### 3. Run the frontend (in a second terminal)

```bash
streamlit run app.py
```

Open the Streamlit URL, type a ticket the way a customer would (e.g. *"My
order #ORD12345 for the wireless earbuds hasn't arrived in 10 days, this is
unacceptable!"*), and watch each agent's decision in the expandable "Agent
decision log".

## API

| Method | Path             | Description                              |
| ------ | ---------------- | ----------------------------------------- |
| POST   | `/ticket`        | Submit raw ticket text, returns `ticket_id` |
| POST   | `/process`       | Runs the full agent pipeline for a `ticket_id` |
| GET    | `/ticket/{id}`   | Fetch the stored ticket + its outcome     |
| GET    | `/logs/{id}`     | Fetch the per-agent decision log          |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests are fully offline/hermetic: they train tiny classifiers and build a
tiny FAISS index (forced to the TF-IDF embedding backend) in a temp
directory per test, so they don't touch the real `models/`, `vector_index/`,
or `support.db`, and don't require network access.

## Evaluation

`scripts/train_classifier.py` writes precision/recall/F1/accuracy for both
the category and priority classifiers to `models/training_report.json`.

## Tech stack

Python, FastAPI, Streamlit, scikit-learn, sentence-transformers, FAISS,
VADER sentiment, SQLAlchemy (SQLite).
