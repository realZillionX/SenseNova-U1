# Runtime identities

Every run records the Forge commit, LightLLM and LightX2V gitlinks, checkpoint
identity, package lock, Python/Torch/CUDA/NCCL versions, attention backends,
launch parameters, and active policy version.

The LightLLM base commit is combined with the checksum-pinned
`serving/patches/lightllm-sensenova-policy.patch` overlay. Runtime preparation
copies the pinned submodule to an isolated runtime source directory and applies
the overlay there, leaving the Git checkout clean. Preflight refuses a runtime
tree missing its request-schedule or neutral-RL-sampling markers.

The SFT lock lives under `training/`. The RL/serving contract lives under
`docker/rl-engine/` and is validated by `scripts/rl_engine/preflight.py` before
the HTTP server starts. The image seals all three source revisions in
`/opt/sensenova-forge/manifests/rl-serving/source-commits.json`; host checkouts
are read directly from Git. Floating submodule branches and unlabelled source
trees are not valid substitutes for those recorded revisions.
