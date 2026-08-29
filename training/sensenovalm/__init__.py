from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "initialize_trainer": (".initialize.initialize_trainer", "initialize_trainer"),
    "get_default_parser": (".initialize.launch", "get_default_parser"),
    "launch_from_slurm": (".initialize.launch", "launch_from_slurm"),
    "launch_from_torch": (".initialize.launch", "launch_from_torch"),
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
