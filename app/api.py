"""
api.py
A REST API over the same engine streamlit_app.py uses (profiler.py,
visualizer.py, groq_agent.py, groq_insight_generator.py) — proof that the
actual analysis logic was never tied to Streamlit specifically, just
driven by it. Useful for anyone who wants to call EDAgent from a script,
another service, or a different frontend entirely.

Run with:
    cd app
    python3 api.py            # dev server on http://localhost:5000
    # or: flask --app api run

Endpoints:
    GET    /                              - the web UI (templates/index.html)
    GET    /health                        - liveness check
    POST   /datasets                      - upload a CSV (multipart 'file' field), returns dataset_id + profile
    GET    /datasets/<id>                 - re-fetch a dataset's profile
    GET    /datasets/<id>/charts          - auto-generated Plotly charts as JSON specs
    POST   /datasets/<id>/report          - {"mode": "agentic"|"single_shot"} -> generated report
    POST   /datasets/<id>/ask             - {"question": "..."} -> grounded answer
    DELETE /datasets/<id>                 - free the in-memory dataset

Known limitation: datasets are held in an in-memory dict, not a database.
Fine for local use, a demo, or a single-process deployment — but it means
data is lost on restart and NOT shared across multiple worker processes
(e.g. `gunicorn -w 4`). If this needs to survive restarts or scale beyond
one process, swap DATASET_STORE for Redis or a real database; nothing
else in this file would need to change, since it's only ever accessed
through get_dataset()/store_dataset()/delete_dataset() below.
"""

import io
import json
import os
import threading
import uuid

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from profiler import profile_dataframe
from visualizer import generate_charts
from sandbox import UnsafeQueryError

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-memory dataset store — see the module docstring's Known limitation.
# ---------------------------------------------------------------------------

DATASET_STORE: dict[str, dict] = {}
_STORE_LOCK = threading.Lock()

# Overridable per-deployment via env vars — no code change needed to trade
# investigation depth for speed. Defaults match generate_insights_agentic's/
# answer_question's own defaults; set lower on a rate-limited or free-tier
# host (each iteration is a real network call, and Groq's free tier can
# throttle shared-IP hosting providers much harder than a home connection —
# a real deployment hit a 5-minute investigation from retries stacking up).
AGENTIC_MAX_ITERATIONS = int(os.environ.get("EDAGENT_MAX_ITERATIONS", "7"))
ASK_MAX_ITERATIONS = int(os.environ.get("EDAGENT_ASK_MAX_ITERATIONS", "6"))


def store_dataset(df: pd.DataFrame, profile: dict) -> str:
    dataset_id = uuid.uuid4().hex
    with _STORE_LOCK:
        DATASET_STORE[dataset_id] = {"df": df, "profile": profile}
    return dataset_id


def get_dataset(dataset_id: str) -> dict | None:
    with _STORE_LOCK:
        return DATASET_STORE.get(dataset_id)


def delete_dataset(dataset_id: str) -> bool:
    with _STORE_LOCK:
        return DATASET_STORE.pop(dataset_id, None) is not None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """The web UI (templates/index.html + static/app.js) — a small
    vanilla-JS frontend that drives the same JSON endpoints below.
    Previously this path 404'd since api.py only exposed JSON routes;
    this makes the API self-documenting/usable in a browser, not just
    from curl or a script."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/datasets", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send a multipart/form-data request with a 'file' field."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    try:
        df = pd.read_csv(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Couldn't parse that as a CSV: {e}"}), 400

    if df.empty:
        return jsonify({"error": "CSV parsed but has no rows."}), 400

    profile = profile_dataframe(df)
    dataset_id = store_dataset(df, profile)
    return jsonify({"dataset_id": dataset_id, "profile": profile}), 201


@app.route("/datasets/<dataset_id>", methods=["GET"])
def get_dataset_profile(dataset_id):
    dataset = get_dataset(dataset_id)
    if dataset is None:
        return jsonify({"error": f"No dataset with id '{dataset_id}'."}), 404
    return jsonify({"dataset_id": dataset_id, "profile": dataset["profile"]})


@app.route("/datasets/<dataset_id>/charts", methods=["GET"])
def get_dataset_charts(dataset_id):
    dataset = get_dataset(dataset_id)
    if dataset is None:
        return jsonify({"error": f"No dataset with id '{dataset_id}'."}), 404

    charts = generate_charts(dataset["df"], dataset["profile"])
    # fig.to_json() is Plotly's own encoder (handles numpy/pandas types
    # jsonify's default encoder can't) — json.loads it back so the whole
    # response goes through jsonify() as one normal JSON-safe dict rather
    # than hand-splicing raw JSON strings into a response body.
    charts_json = {name: json.loads(fig.to_json()) for name, fig in charts.items()}
    return jsonify({"dataset_id": dataset_id, "charts": charts_json})


@app.route("/datasets/<dataset_id>/report", methods=["POST"])
def generate_report(dataset_id):
    dataset = get_dataset(dataset_id)
    if dataset is None:
        return jsonify({"error": f"No dataset with id '{dataset_id}'."}), 404

    if not os.environ.get("GROQ_API_KEY"):
        return jsonify({"error": "GROQ_API_KEY is not set on the server."}), 503

    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "agentic")
    model = body.get("model")
    if mode not in ("agentic", "single_shot"):
        return jsonify({"error": "mode must be 'agentic' or 'single_shot'."}), 400

    try:
        if mode == "agentic":
            from groq_agent import generate_insights_agentic
            result = generate_insights_agentic(dataset["df"], dataset["profile"], model=model, max_iterations=AGENTIC_MAX_ITERATIONS, verbose=False)
            return jsonify({"mode": mode, "report": result["report"], "tool_calls": result["tool_calls"], "iterations": result["iterations"]})
        else:
            from groq_insight_generator import generate_insights
            report = generate_insights(dataset["profile"], chart_names=list(generate_charts(dataset["df"], dataset["profile"]).keys()), model=model)
            return jsonify({"mode": mode, "report": report})
    except UnsafeQueryError as e:
        return jsonify({"error": f"Sandbox rejected a query unexpectedly: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Report generation failed: {e}"}), 502


@app.route("/datasets/<dataset_id>/ask", methods=["POST"])
def ask_question(dataset_id):
    dataset = get_dataset(dataset_id)
    if dataset is None:
        return jsonify({"error": f"No dataset with id '{dataset_id}'."}), 404

    if not os.environ.get("GROQ_API_KEY"):
        return jsonify({"error": "GROQ_API_KEY is not set on the server."}), 503

    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required."}), 400
    model = body.get("model")

    try:
        from groq_agent import answer_question
        result = answer_question(dataset["df"], dataset["profile"], question, model=model, max_iterations=ASK_MAX_ITERATIONS, verbose=False)
        return jsonify({"question": question, "answer": result["answer"], "tool_calls": result["tool_calls"], "iterations": result["iterations"]})
    except UnsafeQueryError as e:
        return jsonify({"error": f"Sandbox rejected a query unexpectedly: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Couldn't answer that: {e}"}), 502


@app.route("/datasets/<dataset_id>", methods=["DELETE"])
def remove_dataset(dataset_id):
    if not delete_dataset(dataset_id):
        return jsonify({"error": f"No dataset with id '{dataset_id}'."}), 404
    return "", 204


if __name__ == "__main__":
    # use_reloader=False is deliberate: Werkzeug's file-watcher can end up
    # monitoring far more than the project directory on some setups
    # (observed on Windows: it started watching files inside pandas'
    # own site-packages installation and kept restarting the server every
    # few seconds, killing any in-progress request each time). debug=True
    # is kept for the interactive error pages; just the auto-restart-on-
    # file-change behavior is disabled. Restart manually (Ctrl+C, rerun)
    # after editing this file.
    app.run(debug=True, port=5000, use_reloader=False)
