"""SFT model loading and one rolling optimizer/RNG recovery generation."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
)
from sensenovalm.data.sample_progress import SampleProgress

OPTIONS = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
RECOVERY_SCHEMA = "sensenova.u15.forge.sft.recovery.v1"


def checkpoint_progress(checkpoint: Path) -> tuple[SampleProgress, dict]:
    checkpoint = Path(checkpoint)
    if checkpoint.name.startswith('.') or not (checkpoint / 'dcp/.metadata').is_file():
        raise ValueError('SFT restart requires a complete model checkpoint')
    metadata = json.loads((checkpoint / 'checkpoint.json').read_text())
    if metadata.get('schema') != 'sensenova.u15.forge.sft.checkpoint.v3' or metadata.get('model_only') is not True:
        raise ValueError('SFT restart requires model-only DCP weights')
    progress = SampleProgress(**{k: metadata[k] for k in SampleProgress.__dataclass_fields__})
    progress.verify_checkpoint_target(metadata['checkpoint_target_samples'])
    expected = 'final' if progress.done else f'samples-{progress.consumed_samples:012d}'
    if checkpoint.name != expected:
        raise ValueError('SFT checkpoint directory does not match its sample count')
    return progress, metadata


def load_model_checkpoint(model, checkpoint: Path) -> SampleProgress:
    progress, _ = checkpoint_progress(checkpoint)
    state = {'model': get_model_state_dict(model, options=OPTIONS)}
    dcp.load(state, checkpoint_id=Path(checkpoint) / 'dcp')
    set_model_state_dict(model, state['model'], options=OPTIONS)
    return progress


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def _rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'python': random.getstate(), 'numpy': np.random.get_state()}


def save_recovery(*, root: Path, model, optimizer, progress: SampleProgress,
                  model_checkpoint: Path, seed: int, batch_samples: int) -> Path:
    """Commit a new generation before retiring the previous optimizer state."""
    saved_progress, _ = checkpoint_progress(model_checkpoint)
    if asdict(saved_progress) != asdict(progress):
        raise ValueError("recovery must reference the model from the same completed update")
    root = Path(root)
    name = f'samples-{progress.consumed_samples:012d}'
    staging, target = root / f'.{name}.staging', root / name
    rank = _rank()
    started = time.perf_counter()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if staging.exists() or target.exists():
            raise FileExistsError(f'recovery generation already exists: {name}')
        staging.mkdir()
    _barrier()
    state = {'optimizer': get_optimizer_state_dict(model, optimizer, options=OPTIONS)}
    dcp.save(state, checkpoint_id=staging / 'optimizer')
    with (staging / f'rng-rank-{rank:05d}.pt').open('wb') as stream:
        torch.save(_rng_state(), stream)
        stream.flush(); os.fsync(stream.fileno())
    _barrier()
    if rank == 0:
        metadata = {'schema': RECOVERY_SCHEMA, **asdict(progress),
                    'model_checkpoint': str(Path(model_checkpoint).resolve()),
                    'world_size': dist.get_world_size() if dist.is_initialized() else 1,
                    'seed': seed, 'batch_samples': batch_samples}
        with (staging / 'recovery.json').open('w') as stream:
            stream.write(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        os.rename(staging, target)
        old = None
        pointer = root / 'latest.json'
        if pointer.exists():
            old = json.loads(pointer.read_text())['generation']
        temporary = root / '.latest.json.tmp'
        with temporary.open('w') as stream:
            stream.write(json.dumps({'generation': name}) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, pointer)
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        if old is not None and old != name:
            if Path(old).name != old or not old.startswith('samples-'):
                raise ValueError('invalid prior recovery generation')
            shutil.rmtree(root / old)
        print(json.dumps({'component': 'sensenova_u15.sft_fsdp2', 'event': 'recovery_saved',
                          'consumed_samples': progress.consumed_samples, 'path': str(target),
                          'seconds': time.perf_counter() - started}), flush=True)
    _barrier()
    return target


def load_recovery(*, model, optimizer, recovery: Path, progress: SampleProgress,
                  seed: int, batch_samples: int) -> None:
    recovery = Path(recovery)
    metadata = json.loads((recovery / 'recovery.json').read_text())
    expected_world = dist.get_world_size() if dist.is_initialized() else 1
    if (metadata.get('schema') != RECOVERY_SCHEMA or metadata['world_size'] != expected_world
            or metadata['seed'] != seed or metadata['batch_samples'] != batch_samples
            or any(metadata[k] != getattr(progress, k) for k in ('samples_per_epoch', 'consumed_samples', 'optimizer_updates', 'last_update_samples'))):
        raise ValueError('SFT recovery state does not match the restored model/progress/topology')
    state = {'optimizer': get_optimizer_state_dict(model, optimizer, options=OPTIONS)}
    saved = dcp.FileSystemReader(recovery / 'optimizer').read_metadata()
    present = {path[2] for path in saved.planner_data.values() if len(path) > 2 and path[:2] == ('optimizer', 'state')}
    state['optimizer']['state'] = {name: value for name, value in state['optimizer']['state'].items() if name in present}
    dcp.load(state, checkpoint_id=recovery / 'optimizer')
    # Parameters that have never received a gradient legitimately have empty
    # Adam state; preserve that state instead of the loader's bootstrap step.
    for group in state['optimizer']['param_groups']:
        for name in group['params']:
            state['optimizer']['state'].setdefault(name, {})
    set_optimizer_state_dict(model, optimizer, state['optimizer'], options=OPTIONS)
    rng = torch.load(recovery / f'rng-rank-{_rank():05d}.pt', map_location='cpu', weights_only=False)
    torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state(rng['cuda'])
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
