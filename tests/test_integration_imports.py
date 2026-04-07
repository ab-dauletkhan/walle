import importlib


def test_orchestrator_support_imports_without_sys_path_hacks():
    module = importlib.import_module("orchestrator_support")
    assert module is not None


def test_memory_modules_import_as_package():
    robot_tools = importlib.import_module("memory.robot_tools")
    memory_system = importlib.import_module("memory.memory_system")
    assert robot_tools is not None
    assert memory_system is not None


def test_top_level_modules_import_with_package_style_dependencies():
    vision_service = importlib.import_module("vision_service")
    walle_main = importlib.import_module("walle_main")
    config_jetson = importlib.import_module("config_jetson")
    assert vision_service is not None
    assert walle_main is not None
    assert config_jetson is not None
