"""
server.py — Flask backend for LexiAssist (Ollama/llama3 version)
"""

import traceback
import ollama
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from retriever import Retriever
from query_expander import expand_query
from generator import generate_answer

app = Flask(__name__, static_folder=".")
CORS(app)

print("Starting LexiAssist server — loading retriever...")
retriever = Retriever()
print("Server ready.\n")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Missing 'query' field."}), 400

    try:
        sub_queries = expand_query(query, n=3)
        retrieved = retriever.retrieve(query, sub_queries=sub_queries)
        result = generate_answer(query, retrieved)
        result["sub_queries"] = sub_queries
        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Pipeline error. Check server logs."}), 500


@app.route("/api/simplify", methods=["POST"])
def api_simplify():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Missing 'text' field."}), 400

    if len(text) > 8000:
        return jsonify({"error": "Text too long (max 8000 characters)."}), 400

    try:
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "Simplify legal text into plain English. Be concise."},
                {"role": "user", "content": text}
            ]
        )
        simplified = response.message.content
        return jsonify({"simplified": simplified})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Simplification failed."}), 500


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "ok",
        "chunks_indexed": len(retriever.chunks),
        "embed_model": "all-MiniLM-L6-v2",
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "llm": "llama3"
    })


@app.route("/")
def serve_gui():
    return send_from_directory(".", "gui.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)