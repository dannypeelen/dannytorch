fix layerNorm params, and chunk, cat, view? sum & exp too
Linear should have Parameter objects
add multiple dimension handling for matmul
check no PyTorch stuff in RoPE
masked_fill?

build MHA & RoPE

Test: Build LLM & NestedLLM (include) <- dt.llm

PUBLISH!----------------------------

maybe add a .train() and .eval() for models

maybe go for GQA, Flash/Flex Attention (this might need kernels)

precision optimization (float32 for grad, float16 forward)

GPU support in Cython?

have a crack at JAX's implementation of autograd?


---------STATUS--------

Tensor: build tensorOps back passes

Optimizer/Scheduler: Do cosine_annealing

Activations:  DONE

nn: Embedding Forward, Parameter class, make robust!!

LLM: MHA, RoPE
