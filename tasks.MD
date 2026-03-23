build Embedding forward, wrap Dropout & LayerNorm

give MLP activation choice -> figure out good standard? automatic Dropout? residual connection?

build nn.Sequential type thing? then see w ModuleList, build for llm
Attention MHA!!
build RoPE
Test: Build LLM & NestedLLM (include) <- dt.llm

PUBLISH!----------------------------

maybe go for GQA, Flash/Flex Attention (this might need kernels)
precision optimization (float32 for grad, float16 forward)
GPU support in Cython?
have a crack at JAX's implementation of autograd

check: squeeze, unsqueeze, axis options? flatten?


---------STATUS--------
Optimizer/Scheduler: Step & Exponentinal LR ------- Do cosine_annealing
Activations:  Done w Adam, SGD, RMSProp ------ finish silu, check viability of SwiGLU, build sigmoid
nn: Good for now ----- add in initialization customization
