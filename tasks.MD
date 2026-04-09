check no PyTorch stuff in RoPE
masked_fill?

build MHA

Test: Build LLM & NestedLLM (include) <- dt.llm

PUBLISH!----------------------------

maybe go for GQA, Flash/Flex Attention (this might need kernels)

precision optimization (float32 for grad, float16 forward)

GPU support in Cython?

have a crack at JAX's implementation of autograd?


---------STATUS--------

TensorOps: DONE

Optimizer/Scheduler: DONE

Activations:  DONE

nn: bug sweep

LLM: MHA, RoPE
