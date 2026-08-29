# --------------------------------------------------------
# SenseNovaVL — derived from InternVL (OpenGVLab, MIT).
# Copyright (c) 2023 OpenGVLab. Licensed under MIT.
# Copyright (c) SenseNovaLM contributors. Modifications licensed under Apache-2.0.
# --------------------------------------------------------

from importlib import import_module

_EXPORTS = {
    "NEOVisionConfig": "configuration_neo_vit",
    "SenseNovaVLChatConfig": "configuration_sensenovavl_chat",
    "NEOVisionModel": "modeling_neo_vit",
    "SenseNovaVLChatMoTModel": "modeling_sensenovavl_chat_mot",
    "build_pipeline_partition_mot_model": "modeling_sensenovavl_chat_mot",
}

__all__ = [
    "NEOVisionConfig",
    "NEOVisionModel",
    "SenseNovaVLChatConfig",
    "SenseNovaVLChatMoTModel",
    "build_pipeline_partition_mot_model",
]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
