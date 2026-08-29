from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Engine": (".engine", "Engine"),
    "NaiveAMPModel": (".naive_amp", "NaiveAMPModel"),
    "Trainer": (".trainer", "Trainer"),
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
