from __future__ import annotations


def test_streamlit_module_imports() -> None:
    import traffic_agent.app.streamlit_app as app

    assert callable(app.main)
