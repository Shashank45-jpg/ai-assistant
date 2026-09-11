"""Production-ready Flask AI Assistant application using Groq."""
from __future__ import annotations
import os
import time
from typing import Any
from flask import Flask, jsonify, render_template, request
from groq import Groq
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "12000"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PROMPT_LIBRARY: dict[str, list[dict[str, str]]] = {
    "answer": [
        {"id": "concise", "label": "Concise factual answer", "system": "You are a precise research assistant. Answer factually in 2-3 sentences. If a fact may be uncertain or time-sensitive, say so."},
        {"id": "explained", "label": "Explained with context", "system": "You are a knowledgeable teacher. Answer the question, explain why it matters, and use a friendly approachable tone."},
        {"id": "detailed", "label": "Detailed with examples", "system": "You are a subject-matter expert. Give a structured answer with useful detail and at least one concrete example when appropriate."},
    ],
    "summarize": [
        {"id": "brief", "label": "One-paragraph brief", "system": "Summarize the supplied text in one tight paragraph of 3-4 sentences. Preserve important facts and avoid adding information."},
        {"id": "bullets", "label": "Key-points bullets", "system": "Summarize the supplied text as 4-6 concise bullet points in the order the main ideas appear. Preserve important facts and avoid invention."},
        {"id": "executive", "label": "Executive overview", "system": "Write an executive-style summary: one sentence describing the subject, followed by 2-3 sentences covering key takeaways and implications supported by the source text."},
    ],
    "generate": [
        {"id": "story", "label": "Short story", "system": "You are a creative fiction writer. Write a vivid short story of roughly 200-300 words with a clear beginning, middle, and end based on the user's idea."},
        {"id": "poem", "label": "Poem", "system": "You are a poet. Write an evocative poem of 8-16 lines based on the user's prompt, using imagery and rhythm."},
        {"id": "idea", "label": "Concept / idea pitch", "system": "You are a creative concept developer. Pitch one original creative concept in 3-5 sentences and explain what makes it interesting."},
    ],
    "advice": [
        {"id": "quick_tips", "label": "Quick tips list", "system": "You are a practical coach. Give 3-5 concise, actionable tips as a numbered list."},
        {"id": "step_by_step", "label": "Step-by-step plan", "system": "You are a planning coach. Give a logical 4-6 step plan the user can follow to make progress."},
        {"id": "encouraging", "label": "Encouraging & personal", "system": "You are a warm, encouraging mentor. Acknowledge the challenge briefly, then give 2-3 concrete, practical suggestions."},
    ],
}
def get_groq_api_key() -> str | None:
    """Read the Groq API key when the Flask application is created."""
    return os.getenv("GROQ_API_KEY")
def _error_status(error: Exception) -> int | None:
    """Extract an HTTP status from a Groq SDK error when available."""
    for name in ("status_code", "status"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    return None
def rate_limit_message(error: Exception) -> str:
    """Return an actionable message for Groq quota/rate-limit failures."""
    message = str(error).lower()
    if any(
        token in message
        for token in (
            "quota",
            "rate limit",
            "rate_limit",
            "billing",
            "too many requests",
        )
    ):
        return (
            "Groq API rate limit or quota reached. "
            "Please wait a moment and try again, or check your Groq API usage."
        )
    return (
        "The Groq AI service is temporarily rate-limiting requests. "
        "Please wait a moment and try again."
    )
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
    api_key = get_groq_api_key()
    client = Groq(api_key=api_key) if api_key else None
    @app.get("/")
    def index():
        return render_template("index.html", prompt_library=PROMPT_LIBRARY)
    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "ai_configured": client is not None,
                "model": MODEL,
            }
        )
    @app.post("/api/run")
    def run_function():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(
                {"error": "Request body must be valid JSON."}
            ), 400
        function = data.get("function")
        prompt_id = data.get("prompt_id")
        raw_input = data.get("input")
        if not isinstance(function, str) or function not in PROMPT_LIBRARY:
            return jsonify({"error": "Unknown function."}), 400
        if not isinstance(prompt_id, str):
            return jsonify(
                {"error": "A prompt variant is required."}
            ), 400
        if not isinstance(raw_input, str):
            return jsonify({"error": "Input must be text."}), 400
        user_input = raw_input.strip()
        if not user_input:
            return jsonify(
                {"error": "Please enter some input."}
            ), 400
        if len(user_input) > MAX_INPUT_LENGTH:
            return jsonify(
                {
                    "error": (
                        f"Input is too large. Please keep it under "
                        f"{MAX_INPUT_LENGTH:,} characters."
                    )
                }
            ), 413
        variant = next(
            (
                item
                for item in PROMPT_LIBRARY[function]
                if item["id"] == prompt_id
            ),
            None,
        )
        if variant is None:
            return jsonify(
                {"error": "Unknown prompt variant."}
            ), 400
        if client is None:
            return jsonify(
                {
                    "error": (
                        "AI service is not configured yet. "
                        "Add GROQ_API_KEY to the server environment."
                    )
                }
            ), 503
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": variant["system"],
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    },
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            content = (
                response.choices[0].message.content
                if response.choices
                else None
            )
            if not content:
                return jsonify(
                    {
                        "error": (
                            "The Groq AI service returned "
                            "an empty response."
                        )
                    }
                ), 502
        except Exception as error:
            status = _error_status(error)
            message = str(error).lower()
            if (
                status in {401, 403}
                or "api key" in message
                or "authentication" in message
            ):
                return jsonify(
                    {
                        "error": (
                            "The Groq API key is invalid or not authorized. "
                            "Check the GROQ_API_KEY in the server environment."
                        )
                    }
                ), 502
            if (
                status == 429
                or "rate limit" in message
                or "rate_limit" in message
                or "quota" in message
            ):
                return jsonify(
                    {"error": rate_limit_message(error)}
                ), 429
            if (
                status in {408, 504}
                or "timeout" in message
            ):
                return jsonify(
                    {
                        "error": (
                            "The Groq AI service took too long "
                            "to respond. Please try again."
                        )
                    }
                ), 504
            if (
                "connection" in message
                or "connect" in message
            ):
                return jsonify(
                    {
                        "error": (
                            "Unable to reach the Groq AI service. "
                            "Please try again."
                        )
                    }
                ), 502
            app.logger.exception(
                "Unexpected Groq request failure"
            )
            return jsonify(
                {
                    "error": (
                        "Something went wrong while "
                        "generating the response."
                    )
                }
            ), 500
        return jsonify(
            {
                "result": content.strip(),
                "function": function,
                "prompt_used": variant["label"],
                "prompt_id": prompt_id,
                "elapsed_seconds": round(
                    time.perf_counter() - started,
                    2,
                ),
            }
        )
    @app.post("/api/feedback")
    def feedback():
        data: Any = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(
                {"error": "Request body must be valid JSON."}
            ), 400
        if not isinstance(data.get("helpful"), bool):
            return jsonify(
                {"error": "Feedback value must be true or false."}
            ), 400
        if (
            not isinstance(data.get("function"), str)
            or data["function"] not in PROMPT_LIBRARY
        ):
            return jsonify({"error": "Invalid function."}), 400
        if (
            not isinstance(data.get("input"), str)
            or not data["input"].strip()
        ):
            return jsonify(
                {"error": "Feedback input is required."}
            ), 400
        if (
            not isinstance(data.get("output"), str)
            or not data["output"].strip()
        ):
            return jsonify(
                {"error": "Feedback output is required."}
            ), 400
        return jsonify({"status": "accepted"}), 202
    @app.get("/api/feedback/stats")
    def feedback_stats():
        return jsonify(
            {
                "total": 0,
                "helpful": 0,
                "not_helpful": 0,
                "persistence": "stateless",
            }
        )
    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify(
            {"error": "Request is too large."}
        ), 413
    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify(
                {"error": "Endpoint not found."}
            ), 404
        return (
            render_template(
                "index.html",
                prompt_library=PROMPT_LIBRARY,
            ),
            404,
        )
    @app.errorhandler(500)
    def server_error(_error):
        return jsonify(
            {"error": "Internal server error."}
        ), 500
    return app
app = create_app()
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )

       
                   
           
        
