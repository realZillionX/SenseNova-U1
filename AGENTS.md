# SenseNova-U1 Forge contract

The `realZillionX/SenseNova-U1` fork provides a checkpoint-specific Forge
framework for SenseNova-U1.5-8B-MoT.

- `training/` owns H200-only full-parameter FSDP2 SFT, model-only DCP checkpoints, rolling optimizer/RNG recovery, and
  atomic Hugging Face safetensors publication.
- `src/sensenova_u1/rl/` owns full-parameter FSDP2 GDPO/UniGDPO, replay,
  checkpointing, and online policy publication.
- `serving/third_party/{LightLLM,LightX2V}` plus `scripts/rl_engine/` own all
  production inference and behavior rollout.
- Downstream repositories own task data, prompts, verifier implementations,
  reward semantics, benchmark runners, and experiment packages.
- Do not add local serial checkpoint-inference demos, ComfyUI, GGUF/offload,
  evaluator copies, or task-specific reward code.
- Preserve exact model, package, image, LightLLM, and LightX2V identities in
  every SFT, RL, and serving run.
- SFT, RL, and serving share the repository-root Torch 2.8/CUDA 12.8 runtime;
  do not introduce a second training environment or non-H200 execution path.
- Ordinary serving declares `FORGE_SERVING_MODALITY`: TI2T uses one GPU per
  replica with input vision and no LightX2V worker; TI2TI uses two GPUs per
  replica with its image generator. An eight-GPU node therefore runs eight
  TI2T replicas or four TI2TI replicas.
- Serving has one total trajectory limit, `max_sequence_length`: all input and
  output text/image tokens share it, and server capacity equals it. Ordinary
  serving starts at 16384 and RL at 8192; both allow at most 10 generated images,
  with text continuing after the image ceiling. Derive internal decoder bounds
  from remaining context; never expose a separate text limit or larger capacity.
  Changes to serving, RL, launchers, or LightLLM revisions must pass
  `tests/test_rl_plan.py`, `tests/test_serving_contract.py`, and LightLLM's
  `unit_tests/server/test_trajectory_budget.py`.
- Delete generated media, smoke outputs, temporary traces, and superseded
  infrastructure after their evidence or reusable rule has been retained.
- Durable documentation describes current behavior and open TODOs only; it
  does not record task progress or completed migration history.
- Maintain one version of each document; do not create `*_CN.md` translation
  copies. Keep README visuals and ensure every command and claim is executable.
