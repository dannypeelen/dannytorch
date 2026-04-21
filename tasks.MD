Introduce batching

gradient clipping?

Add in warmup option for schedulers

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
