"""Pruebas unitarias para progreso de descarga, límite RAM, bloqueo Large en CPU y stepper UI."""
from __future__ import annotations

import io
import os
import tempfile

from src import sam_engine
from src.ui import compute_stage_states


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_ensure_checkpoint_reports_progress_and_is_atomic(monkeypatch):
    payload = b"sam2-checkpoint-bytes" * 100
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "sam2_test.pt")
        info = {"checkpoint": ckpt, "url": "https://example.invalid/sam2_test.pt"}
        monkeypatch.setattr(sam_engine.urllib.request, "urlopen", lambda url, timeout=60: _FakeResponse(payload))

        out = sam_engine.ensure_checkpoint(info, progress_callback=lambda done, total: calls.append((done, total)))
        assert out == ckpt
        assert os.path.exists(ckpt)
        assert not os.path.exists(ckpt + ".part")
        assert calls[0] == (0, len(payload))
        assert calls[-1] == (len(payload), len(payload))


def test_max_models_in_ram_env(monkeypatch):
    monkeypatch.delenv("GEOSAM_MAX_MODELS", raising=False)
    assert sam_engine.max_models_in_ram() == 1
    monkeypatch.setenv("GEOSAM_MAX_MODELS", "2")
    assert sam_engine.max_models_in_ram() == 2
    monkeypatch.setenv("GEOSAM_MAX_MODELS", "invalid")
    assert sam_engine.max_models_in_ram() == 1


def test_large_model_blocked_on_cpu_unless_allowed(monkeypatch):
    monkeypatch.setattr(sam_engine, "cuda_is_available", lambda: False)
    monkeypatch.delenv("GEOSAM_ALLOW_LARGE_CPU", raising=False)

    ok_tiny, _ = sam_engine.can_load_model("sam2.1_hiera_tiny")
    ok_large, reason = sam_engine.can_load_model("sam2.1_hiera_large")
    assert ok_tiny is True
    assert ok_large is False and "GEOSAM_ALLOW_LARGE_CPU=1" in reason

    monkeypatch.setenv("GEOSAM_ALLOW_LARGE_CPU", "1")
    ok_large_override, _ = sam_engine.can_load_model("sam2.1_hiera_large")
    assert ok_large_override is True


def test_stepper_state_progression():
    assert compute_stage_states(False, False, False, False, 0) == ["active", "pending", "pending", "pending", "pending"]
    assert compute_stage_states(True, True, False, False, 0) == ["done", "done", "active", "pending", "pending"]
    assert compute_stage_states(True, True, True, True, 0) == ["done", "done", "done", "active", "pending"]
    assert compute_stage_states(True, True, True, True, 1) == ["done", "done", "done", "done", "active"]
    assert compute_stage_states(True, True, False, False, 1) == ["done", "done", "active", "done", "done"]

