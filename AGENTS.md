# SenseNova-U1.5-8B-MoT Forge contract

This repository is an independent, checkpoint-specific framework for
SenseNova-U1.5-8B-MoT.

- `training/` owns full-parameter InternEvo SFT and InternalEvo→HF conversion.
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
- Delete generated media, smoke outputs, temporary traces, and superseded
  infrastructure after their evidence or reusable rule has been retained.
- Durable documentation describes current behavior and open TODOs only; it
  does not record task progress or completed migration history.
- Keep the visual quality and bilingual layout of the README while ensuring
  every command and claim is executable against the current tree.
