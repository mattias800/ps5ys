# Execution-layer tests

These tests cover the contracts that turn decoded GPU work into executable operations: dependency
ordering, draw realization, index interpretation, and compiled-shader cache identity. The cache tests
compare emitted modules and warm-cache behavior without running a game; they are not visual or
performance oracles. Vulkan execution coverage is separate, including tests under `tests/shared/`.

When changing a cache key, exercise both a reusable identical key and a changed code-generation
input. Diagnostic selectors are code-generation inputs too, even when normal launches leave them off.
