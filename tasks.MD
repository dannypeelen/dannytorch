SGD w/ momentum, ADaM <- get and test
Learning rate scheduler
activation - leaky ReLU? SwiGLU
add add_module, make sure Module is good
build Embedding thing

give MLP activation choice -> figure out good standard? automatic Dropout? check for [that thing where you pass info ahead etc.]

Dropout and LayerNorm
build nn.Sequential type thing? then see w ModuleList, build for llm
Attention MHA!! - 
look into MoE and RoPE & positional encoder <- iirc MoE, then this can be its own dt.____
Test: Build LLM & NestedLLM (include) <- dt.llm

maybe go for GQA, Flash/Flex Attention (this might need kernels)
precision optimization (float32 for grad, float16 forward)
GPU support in Cython?
have a crack at JAX's implementation of autograd

check: squeeze, unsqueeze, axis options? flatten?
