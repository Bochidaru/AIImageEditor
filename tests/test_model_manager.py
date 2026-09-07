from inpaint_core.models import ModelManager


def test_sequential_policy_releases_previous_resource():
    manager = ModelManager("sequential")
    manager.get("sam2", object)
    manager.get("flux_fill", object)
    assert not manager.contains("sam2")
    assert manager.contains("flux_fill")


def test_resident_policy_keeps_resources():
    manager = ModelManager("resident")
    manager.get("sam2", object)
    manager.get("flux_fill", object)
    assert manager.contains("sam2")
    assert manager.contains("flux_fill")
