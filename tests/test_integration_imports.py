"""Smoke tests: verify all package modules import without errors."""

import importlib


def test_core_package_imports():
    assert importlib.import_module("walle") is not None
    assert importlib.import_module("walle.orchestrator") is not None
    assert importlib.import_module("walle.chat_loop") is not None
    assert importlib.import_module("walle.startup") is not None
    assert importlib.import_module("walle.lifecycle") is not None
    assert importlib.import_module("walle.serial_manager") is not None


def test_memory_imports():
    assert importlib.import_module("walle.memory.config") is not None
    assert importlib.import_module("walle.memory.memory_system") is not None
    assert importlib.import_module("walle.memory.repository") is not None
    assert importlib.import_module("walle.memory.embeddings") is not None
    assert importlib.import_module("walle.memory.vector_index") is not None
    assert importlib.import_module("walle.memory.search_chain") is not None
    assert importlib.import_module("walle.memory.context_manager") is not None
    assert importlib.import_module("walle.memory.heartbeat") is not None


def test_tools_imports():
    assert importlib.import_module("walle.tools.registry") is not None
    assert importlib.import_module("walle.tools.communication") is not None
    assert importlib.import_module("walle.tools.knowledge") is not None
    assert importlib.import_module("walle.tools.personality") is not None
    assert importlib.import_module("walle.tools.memory_tools") is not None
    assert importlib.import_module("walle.tools.robot.catalog") is not None
    assert importlib.import_module("walle.tools.robot.executor") is not None


def test_vision_imports():
    assert importlib.import_module("walle.vision.service") is not None
    assert importlib.import_module("vision.face_recognition.common") is not None
    assert importlib.import_module("vision.face_recognition.scan_people") is not None
    assert importlib.import_module("vision.face_recognition.create_embeddings") is not None
    assert importlib.import_module("vision.face_recognition.recognize_face") is not None
    assert importlib.import_module("vision.face_recognition.track_and_turn_head") is not None
