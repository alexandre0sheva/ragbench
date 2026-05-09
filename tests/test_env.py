import os

from ragbench.utils import env


def test_project_env_loads_openai_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    env._LOADED_ENV_PATHS.clear()

    assert env.has_openai_key() is True
    assert os.environ["OPENAI_API_KEY"] == "test-key"


def test_project_env_can_load_relative_to_config_path(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "configs" / "vector.yaml"
    config.parent.mkdir()
    config.write_text("run: {}\n", encoding="utf-8")
    (project / ".env").write_text("OPENAI_API_KEY=config-path-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env._LOADED_ENV_PATHS.clear()

    env.load_project_env(config)

    assert os.environ["OPENAI_API_KEY"] == "config-path-key"
