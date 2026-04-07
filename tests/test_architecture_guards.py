"""Architecture guard tests — verify contracts survive refactoring."""

import types

import pytest

from walle.memory.context_manager import ContextManager, VisualContext
from walle.memory.vector_index import FAISSManager
from walle.tools.registry import (
    ToolSuiteFacade, ToolRegistry, CommunicationToolProvider, SchemaExecutorToolProvider,
)
from walle.vision.service import VisionService


class FakeExecutor:
    method_prefix = "_"

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def execute(self, name: str, args: dict) -> str:
        self.calls.append((name, dict(args)))
        return self.responses.get(name, f"ok:{name}")


class FakeLLMClient:
    def __init__(self):
        self.messages = []
        self.context_manager = ContextManager()

    def chat(self, text: str) -> str:
        self.messages.append(text)
        return f"echo:{text}"


class StubVisionService(VisionService):
    def _create_backend(self):
        return types.SimpleNamespace(close=lambda: None)


def test_api_server_uses_public_chat_interface():
    pytest.importorskip("flask")
    from walle.api_server import create_app

    app = create_app(
        serial_manager=types.SimpleNamespace(
            send_command=lambda cmd: "OK",
            is_connected=lambda: False,
            simulation=True,
        ),
        llm_client=FakeLLMClient(),
    )
    client = app.test_client()

    response = client.post("/chat", json={"text": "hello"})

    assert response.status_code == 200
    assert response.get_json()["response"] == "echo:hello"


def test_context_manager_to_dict_returns_safe_copies():
    manager = ContextManager()
    manager.update_visual(VisualContext(objects_detected=[{"object": "book"}]))
    manager.spontaneous_observations.append("test")

    snapshot = manager.to_dict()
    snapshot["visual"]["objects_detected"].append({"object": "lamp"})
    snapshot["observations"].append("mutated")

    assert len(manager.visual.objects_detected) == 1
    assert manager.spontaneous_observations == ["test"]


def test_faiss_manager_index_id_snapshot_is_stable_copy():
    manager = FAISSManager(index_path=None)
    manager.id_map = [1, 2, 3]

    snapshot = manager.get_indexed_ids()
    snapshot.add(99)  # get_indexed_ids returns set

    assert manager.id_map == [1, 2, 3]


def test_tool_suite_facade_dispatches_via_registry():
    from walle.tools.robot.executor import get_robot_control_tools
    from walle.tools.knowledge import get_knowledge_tools

    robot = FakeExecutor({"drive_forward": "Driving"})
    knowledge = FakeExecutor({"consult_internet_for_facts": "facts"})

    registry = ToolRegistry()
    registry.register(SchemaExecutorToolProvider(get_robot_control_tools, robot))
    registry.register(SchemaExecutorToolProvider(get_knowledge_tools, knowledge, request_heartbeat_after=True))
    facade = ToolSuiteFacade(registry)

    result = facade.execute_tool(
        name="drive_forward",
        args={"speed": 25},
        user_message_already_sent=False,
    )

    assert result.tool_result == "Driving"
    assert robot.calls == [("drive_forward", {"speed": 25})]


def test_vision_service_exposes_public_state():
    vision = StubVisionService(ContextManager(), camera_index=-1, fps=1)

    assert vision.backend_name == "simplenamespace"
    assert vision.is_active is False
