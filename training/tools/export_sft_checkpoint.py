"""Export a committed model-only SFT DCP without a live trainer or GPUs."""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from sensenovalm.data.sample_progress import SampleProgress
from tools.publish_hf import convert


def committed_checkpoint(path: Path) -> dict:
    """Accept committed sample boundaries, never an incomplete staging directory."""
    path = Path(path)
    if path.name.startswith('.') or not path.is_dir():
        raise ValueError('SFT checkpoint must be a committed directory')
    metadata = json.loads((path / 'checkpoint.json').read_text())
    if (metadata.get('schema') != 'sensenova.u15.forge.sft.checkpoint.v3'
            or metadata.get('model_only') is not True):
        raise ValueError('expected a model-only SFT checkpoint')
    clock = SampleProgress(**{key: metadata[key] for key in SampleProgress.__dataclass_fields__})
    clock.verify_checkpoint_target(metadata['checkpoint_target_samples'])
    expected = 'final' if clock.done else f'samples-{clock.consumed_samples:012d}'
    if path.name != expected or not (path / 'dcp/.metadata').is_file():
        raise ValueError('SFT checkpoint name or DCP metadata is incomplete')
    if not isinstance(metadata.get('conversion_config'), dict):
        raise ValueError('SFT checkpoint has no sealed HF conversion configuration')
    return metadata


def latest_checkpoint(root: Path) -> Path:
    candidates = []
    for path in Path(root).iterdir():
        if path.is_dir() and (path.name == 'final' or path.name.startswith('samples-')):
            metadata = committed_checkpoint(path)
            candidates.append((metadata['consumed_samples'], path))
    if not candidates:
        raise ValueError('no committed SFT checkpoint exists')
    return max(candidates)[1]


def export_checkpoint(checkpoint: Path, target: Path, base_model: Path) -> dict:
    """Read sharded weights on CPU and atomically publish complete HF files."""
    metadata = committed_checkpoint(checkpoint)
    target, base_model = Path(target), Path(base_model)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'refusing to overwrite HF publication: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    reader = dcp.FileSystemReader(Path(checkpoint) / 'dcp')
    tensors = reader.read_metadata().state_dict_metadata
    if not tensors or any(not key.startswith('model.') or not isinstance(value, TensorStorageMetadata)
                          for key, value in tensors.items()):
        raise ValueError('DCP contains state outside the model tensor closure')
    state = {'model': {key.removeprefix('model.'): torch.empty(tuple(value.size), dtype=value.properties.dtype, device="cpu")
                       for key, value in tensors.items()}}
    dcp.load(state, storage_reader=reader)
    with tempfile.TemporaryDirectory(prefix=f'.{target.name}.export-', dir=target.parent) as folder:
        work = Path(folder); source = work / 'source'; source.mkdir(); staging = work / 'hf'
        torch.save(state.pop('model'), source / 'model_wp0_pp0.pt')
        del state
        torch.save(metadata['conversion_config'], source / 'model_config.pt')
        convert(src=str(source), tgt=str(staging), typ='neo++_mot', extras_from=str(base_model))
        from safetensors import safe_open
        index = json.loads((staging / 'model.safetensors.index.json').read_text())
        referenced = set(index['weight_map'].values())
        if not referenced or any(not (staging / name).is_file() for name in referenced):
            raise ValueError('HF conversion produced incomplete weights')
        for shard in staging.glob('*.safetensors'):
            if shard.name not in referenced:
                with safe_open(shard, framework='pt', device='cpu') as handle:
                    if list(handle.keys()):
                        raise ValueError('HF conversion produced unreferenced weights')
                shard.unlink()
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'HF publication appeared during export: {target}')
        os.rename(staging, target)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--latest-from', type=Path)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint or latest_checkpoint(args.latest_from)
    metadata = export_checkpoint(checkpoint, args.target, args.base_model)
    print(json.dumps({'checkpoint': str(checkpoint), 'target': str(args.target),
                      'consumed_samples': metadata['consumed_samples']}))


if __name__ == '__main__':
    main()
