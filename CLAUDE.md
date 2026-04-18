# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# DannyTorch — Project Guide

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DannyTorch is a from-scratch deep learning framework built on NumPy (with optional CuPy for GPU). No PyTorch/JAX dependency — all autograd, layers, optimizers, and the Transformer are hand-written.

## Running code

There is no build step or test suite. Development is done by running scripts directly:

```bash
python test.py          # main LLM training script
python test.ipynb       # notebook equivalent
```

## Architecture

### Autograd engine — `dannytorch/tensor.py`

`tensor` is the core class. Every op (`+`, `@`, `.relu()`, `.softmax()`, etc.) creates a new `tensor` with `_prev` pointing to its inputs and a `_backward` closure capturing the gradient formula. `tensor.backward()` does a Python topo-sort over `_prev` and calls each `_backward` in reverse order.

Key design constraint: **gradients accumulate into `.grad` (a plain NumPy array) on the tensor object**, not returned. `zero_grad()` must be called explicitly before each backward.

### Neural network layers — `dannytorch/nn/nn.py`

- `Parameter(tensor)` — wraps a tensor as a trainable weight. `.data` is itself a `tensor`; `.grad` is also a `tensor` (so `param.data.data` and `param.grad.data` reach the raw NumPy arrays).
- `Module` — base class. `__setattr__` auto-registers `Parameter` → `_parameters` and `Module` → `_modules`. `parameters()` yields recursively. `ModuleList` holds lists of submodules outside `_modules`.
- `Linear(nin, nout, activation='relu', init='He')` — defaults to ReLU activation. Pass `activation=None` for a raw linear projection (required for output heads and FFN down-projections).
- `LayerNorm`, `Dropout`, `Embedding`, `Sequential`, `MLP` all follow the same `forward()` / `parameters()` pattern.

### LLM — `dannytorch/llm/llm.py`

Standard decoder-only Transformer: `Embedding → Dropout → N×TransformerBlock → LayerNorm → Linear(out_head)`.

`TransformerBlock`: pre-norm is **not** used — LayerNorm is applied **after** the residual add (post-norm). Attention uses RoPE (applied to Q and K inside `MultiheadAttention`). The FFN is `Linear(d→4d, relu) → ReLU → Linear(4d→d, activation=None) → Dropout`.

`MultiheadAttention` reshapes to `(B, H, T, head_dim)`, applies causal mask via `masked_fill(-inf)`, then softmax on axis=-1.

### Positional encoding — `dannytorch/lang/lang.py`

`rope(dim, seq_len)` precomputes cos/sin caches. Applied as `x * cos + rotate_half(x) * sin` where `rotate_half` is `[-x2, x1]` (second half negated, prepended). Called with `seq_len` at forward time to slice the cache.

### Optimizers — `dannytorch/optim/optim.py`

`Adam` stores moment buffers as plain NumPy arrays indexed by parameter position (not by parameter identity). Schedulers (`CosineAnnealingLR`, `StepLR`, `ExponentialLR`) must be stepped **once per epoch**, not per batch.

## Known gotchas

- `Parameter.grad` is a `tensor`, not a bare NumPy array. Access the array via `param.grad.data`. Optimizers already do this.
- `Linear` applies activation by default (`'relu'`). Output heads and FFN down-projections need `activation=None`.
- CuPy is used automatically if installed (`try: import cupy as np`). The framework is otherwise identical on CPU.
- `last_logit()` in `test.py` manually detaches the final-token slice from the `(B, T, V)` output so `CrossEntropyLoss` receives a 1D logit tensor per sample — this is intentional, not a bug.