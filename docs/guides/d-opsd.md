# On-policy self-distillation for diffusion language models

The [d-OPSD research project](../../research/d_opsd/README.md) implements
future-conditioned self-distillation for Nemotron Labs Diffusion using the shared
block-diffusion runtime. The student generates its own responses; a frozen copy
of the initial checkpoint sees additional response tokens and supplies
full-vocabulary reverse-KL targets at recorded denoising states.

The project README describes the block-attention adaptation, verification
filtering, supported runtime constraints, runnable configs, and tests. This
objective improves denoising predictions; reducing inference steps must be
measured separately at matched decoding budgets.
