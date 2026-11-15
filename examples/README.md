# Examples

Kernel generators, imported as `examples.vadd` with the checkout on the path.

This folder has no `__init__.py` on purpose. A kernel's address hashes every
importable package beside its generator's import root; were `examples` a
package, that root would be the checkout's, and every compile of an example
would hash `tests` (with `tests/data`) and `tools` too. As a folder of its
own, an example's address is this folder's modules and the thin it runs on.
The examples import only each other, thin and numpy
(`tests/test_runtime_unit.py` holds them to it).
