from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workbench_imports_core_rag_api():
    app = load_module("workbench_app_mod", "workbench_app.py")

    assert callable(app.answer_stream)
    assert callable(app.call_oss_direct)
    assert app.D_STEPS[0][0] == "D0"


def test_standalone_capa_app_imports_shared_core():
    app = load_module("standalone_capa_app_mod", "scripts/app.py")

    assert callable(app.answer)
    assert callable(app._load_bge)
    assert "OSS stack — gpt-oss-120b + gpt-oss-20b" in app.MODEL_STACK_OPTIONS


def test_eval_and_runtime_models_match_current_oss_stack():
    eval_mod = load_module("eval_mod", "evaluation/eval.py")

    from capa_8d_expert.answer import (
        ANSWER_MODEL,
        LLM_RERANK_MODEL,
        MODEL_STACKS,
        REWRITE_MODEL,
        OSS_120B_MODEL,
        OSS_20B_MODEL,
    )

    assert eval_mod.JUDGE_MODEL == "claude-sonnet-4-6"
    assert ANSWER_MODEL == REWRITE_MODEL == OSS_120B_MODEL == "groq/openai/gpt-oss-120b"
    assert LLM_RERANK_MODEL == OSS_20B_MODEL == "groq/openai/gpt-oss-20b"
    assert set(MODEL_STACKS) == {"oss", "legacy_mixed", "cheap_oss"}
    assert MODEL_STACKS["legacy_mixed"].answer_model == "gpt-4o-mini"
    assert MODEL_STACKS["cheap_oss"].answer_model == "groq/openai/gpt-oss-20b"
    assert "4-5" not in eval_mod.JUDGE_MODEL
    assert all("gpt-4o" not in model and "haiku" not in model for model in [ANSWER_MODEL, REWRITE_MODEL, LLM_RERANK_MODEL])


def test_script_wrappers_remain_thin_launchers():
    launcher = (ROOT / "scripts" / "workbench_app.py").read_text()
    answer_wrapper = (ROOT / "scripts" / "answer.py").read_text()
    ingest_wrapper = (ROOT / "scripts" / "ingest.py").read_text()

    assert "runpy.run_path" in launcher
    assert "from capa_8d_expert.answer import main" in answer_wrapper
    assert "from capa_8d_expert.ingest import main" in ingest_wrapper


def test_stale_standalone_rag_scripts_do_not_reappear():
    assert not (ROOT / "scripts" / "answer_groq.py").exists()
    assert not (ROOT / "scripts" / "answer_original.py").exists()
