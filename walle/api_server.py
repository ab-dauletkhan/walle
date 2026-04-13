"""
WALL-E API Server
Lightweight Flask bridge for web interface and external control.
Runs on a separate port (default 5001) alongside the main orchestrator.
"""

import logging
import re
import threading
from typing import Optional

from flask import Flask, jsonify, request

_log = logging.getLogger("walle.api")


def create_app(serial_manager, llm_client=None, vision_service=None):
    """Create Flask app with references to shared subsystems."""
    app = Flask(__name__)
    _chat_lock = threading.Lock()

    @app.route("/command", methods=["POST"])
    def send_command():
        """Send a raw serial command to the Arduino."""
        data = request.get_json(silent=True) or {}
        cmd = data.get("command", "").strip()
        if not cmd:
            return jsonify({"error": "Missing 'command' field"}), 400
        if len(cmd) > 20 or not re.match(r"^[A-Za-z0-9\n\s?-]+$", cmd):
            return jsonify({"error": "Invalid command format"}), 400
        result = serial_manager.send_command(cmd)
        return jsonify({"status": result})

    @app.route("/status", methods=["GET"])
    def get_status():
        """Return robot connection status."""
        return jsonify(
            {
                "connected": serial_manager.is_connected(),
                "simulation": serial_manager.simulation,
            }
        )

    @app.route("/chat", methods=["POST"])
    def chat():
        """Send text to the LLM chat pipeline, return response."""
        if llm_client is None:
            return jsonify({"error": "LLM not available"}), 503
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400
        if not _chat_lock.acquire(timeout=60):
            return jsonify({"error": "Chat is busy, try again later"}), 429
        try:
            response = llm_client.chat(text)
            return jsonify({"response": response or "(no response)"})
        except Exception:
            _log.exception("Chat error")
            return jsonify({"error": "Internal server error"}), 500
        finally:
            _chat_lock.release()

    @app.route("/vision", methods=["GET"])
    def get_vision():
        """Return latest vision context as JSON."""
        if llm_client is None:
            return jsonify({"error": "LLM not available"}), 503
        ctx = llm_client.context_manager
        visual = ctx.visual
        if visual is None:
            return jsonify({"faces": [], "objects": [], "scene": "", "description": ""})
        return jsonify(
            {
                "faces": visual.faces_detected,
                "objects": visual.objects_detected,
                "scene": visual.scene_description,
                "description": visual.to_natural_language(),
                "timestamp": visual.timestamp.isoformat(),
            }
        )

    return app


class APIServer:
    """Runs the Flask API in a background thread."""

    def __init__(
        self, serial_manager, llm_client=None, vision_service=None, port: int = 5001
    ):
        self._port = port
        self._app = create_app(serial_manager, llm_client, vision_service)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the API server in a daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="api-server",
        )
        self._thread.start()
        _log.info("Server started on http://localhost:%s", self._port)

    def _run(self) -> None:
        # Suppress Flask's default startup banner
        import logging

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)
        self._app.run(host="0.0.0.0", port=self._port, threaded=True)

    def stop(self) -> None:
        # Flask dev server doesn't have a clean shutdown from another thread,
        # but since the thread is a daemon it will be killed on process exit.
        _log.info("Server stopping")
