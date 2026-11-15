"""Authored device scripts and kernel/closure source; numeric tables stay recorded."""

from tools.golden import encode, no_checked_data, runtime_device, runtime_kernel

BASE = "tests/data/runtime/"
OUTPUTS = (BASE + "device.json", BASE + "kernel.json")


def generate():
    """Evaluate fresh programs with an isolated store-counter initial state."""
    from thin.runtime import store

    saved = dict(store._COUNTS)
    store._COUNTS.clear()
    try:
        with no_checked_data():
            documents = runtime_device.generate_doc(), runtime_kernel.generate_doc()
        return {path: encode(doc) for path, doc in zip(OUTPUTS, documents, strict=True)}
    finally:
        store._COUNTS.clear()
        store._COUNTS.update(saved)
