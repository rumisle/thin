"""New DSL programs, split by language surface rather than stored fixtures."""

from tools.golden import dsl_control, dsl_memory, dsl_values

MODULES = (dsl_values, dsl_control, dsl_memory)
OUTPUTS = tuple(path for module in MODULES for path in module.OUTPUTS)


def generate():
    """Evaluate every authored DSL plan; the numeric magic table stays recorded."""
    fresh = {}
    for module in MODULES:
        for path, raw in module.generate().items():
            if path in fresh:
                raise ValueError(f"two DSL generators produced {path}")
            fresh[path] = raw
    return fresh
