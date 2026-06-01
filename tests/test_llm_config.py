from src.utils.llm_config import build_minimax_llm


def test_build_minimax_llm_uses_configured_temperature(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
    config = {
        "minimax": {
            "api_key": "${MINIMAX_API_KEY}",
            "base_url": "https://api.minimax.chat",
            "model": "MiniMax-M2.7",
            "temperature": 0.2,
        }
    }

    llm = build_minimax_llm(config)

    assert llm.api_key == "env-key"
    assert llm.base_url == "https://api.minimax.chat"
    assert llm.model == "MiniMax-M2.7"
    assert llm.temperature == 0.2


def test_build_minimax_llm_defaults_temperature_to_low_value(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
    config = {
        "minimax": {
            "api_key": "${MINIMAX_API_KEY}",
            "base_url": "https://api.minimax.chat",
            "model": "MiniMax-M2.7",
        }
    }

    llm = build_minimax_llm(config)

    assert llm.temperature == 0.1
