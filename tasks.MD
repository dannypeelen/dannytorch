SGD w/ momentum, ADaM
Learning rate scheduler
activation - leaky ReLU? SwiGLU
add add_module, make sure Module is good

Dropout and LayerNorm
build nn.Sequential type thing?
Attention MHA!! - maybe go for GQA, Flash/Flex Attention (this might need kernels)
look into MoE and RoPE & positional encoder
Test: Build LLM & NestedLLM (include) <- dt.llm

precision optimization (float32 for grad, float16 forward)
GPU support in Cython?
have a crack at JAX's implementation of autograd

check: squeeze, unsqueeze, axis options? flatten?
