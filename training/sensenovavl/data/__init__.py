from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_train_loader_with_data_type": (".build_dataloader", "build_train_loader_with_data_type"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
