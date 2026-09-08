import sys
from types import ModuleType

from inpaint_core.config import AppConfig
from inpaint_core.models.artifacts import prefetch_model_assets


def test_prefetch_replaces_remote_ids_with_local_paths(monkeypatch, tmp_path):
    calls = []
    fake_hub = ModuleType("huggingface_hub")

    def snapshot_download(repo_id, cache_dir=None, allow_patterns=None):
        calls.append((repo_id, allow_patterns))
        destination = tmp_path / repo_id.replace("/", "--")
        destination.mkdir(exist_ok=True)
        if allow_patterns:
            for relative in allow_patterns:
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        return str(destination)

    fake_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    checkpoint = tmp_path / "realesrgan.pth"
    checkpoint.touch()
    config = AppConfig.from_dict({
        "models": {
            "segmentation": {"backend": "sam2", "model_id": "org/sam"},
            "flux_fill": {"backend": "flux_fill", "model_id": "org/fill"},
            "flux_generation": {"backend": "flux", "model_id": "org/base"},
            "flux_kontext": {"backend": "flux_kontext", "model_id": "org/kontext"},
            "omnipaint": {
                "backend": "omnipaint", "model_id": "org/omni",
                "base_model_id": "org/base",
            },
            "upscaler": {"backend": "realesrgan", "checkpoint": str(checkpoint)},
        },
        "artifacts": {"weights_dir": str(tmp_path / "weights")},
    })

    result = prefetch_model_assets(config)

    assert config.segmentation.model_id == result["sam2"]
    assert config.flux_fill.model_id == result["flux_fill"]
    assert config.omnipaint.model_id == result["omnipaint_adapters"]
    assert config.omnipaint.options["base_model_id"] == result["omnipaint_base"]
    assert [repo for repo, _ in calls].count("org/base") == 1
