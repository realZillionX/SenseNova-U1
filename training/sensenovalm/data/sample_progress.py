"""Global raw-sample exposure clock for packed SFT (no Torch dependency)."""

from dataclasses import dataclass
import math

CHECKPOINTS_PER_EPOCH = 10


@dataclass
class SampleProgress:
    samples_per_epoch: int
    max_samples: int
    consumed_samples: int = 0
    optimizer_updates: int = 0
    last_update_samples: int = 0

    def __post_init__(self):
        for name in ("samples_per_epoch", "max_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.samples_per_epoch < CHECKPOINTS_PER_EPOCH:
            raise ValueError(
                f"{CHECKPOINTS_PER_EPOCH} checkpoints per epoch require at least "
                f"{CHECKPOINTS_PER_EPOCH} samples"
            )
        if self.consumed_samples < 0 or self.optimizer_updates < 0:
            raise ValueError("sample progress cannot be negative")

    @property
    def targets(self):
        n = self.samples_per_epoch
        parts = CHECKPOINTS_PER_EPOCH
        return tuple(
            sorted(
                {
                    epoch * n + (part * n + parts - 1) // parts
                    for epoch in range((self.max_samples + n - 1) // n)
                    for part in range(1, parts + 1)
                    if epoch * n + (part * n + parts - 1) // parts <= self.max_samples
                }
                | {self.max_samples}
            )
        )

    @property
    def done(self):
        return self.consumed_samples >= self.max_samples

    def checkpoint_target(self, batch_samples):
        if type(batch_samples) is not int or batch_samples < 1 or self.done:
            raise ValueError("a live SFT update must consume positive raw samples")
        crossed = [n for n in self.targets if self.consumed_samples < n <= self.consumed_samples + batch_samples]
        if len(crossed) > 1:
            raise ValueError(
                "one optimizer update crosses multiple sample checkpoints; reduce the batch or increase the epoch size"
            )
        return crossed[0] if crossed else None

    def advance(self, batch_samples):
        target = self.checkpoint_target(batch_samples)
        self.consumed_samples += batch_samples
        self.optimizer_updates += 1
        self.last_update_samples = batch_samples
        return target

    def verify_checkpoint_target(self, target):
        if self.optimizer_updates < 1 or self.last_update_samples < 1:
            raise ValueError("checkpoint must record a completed sample update")
        previous = type(self)(
            self.samples_per_epoch,
            self.max_samples,
            self.consumed_samples - self.last_update_samples,
            self.optimizer_updates - 1,
        )
        if previous.checkpoint_target(self.last_update_samples) != target:
            raise ValueError("checkpoint does not match its sample boundary")

    def learning_rate_ratio(self, warmup_samples, schedule, min_ratio):
        if not 0 <= warmup_samples <= self.max_samples:
            raise ValueError("warmup_samples must be inside the sample budget")
        position = min(self.max_samples, self.consumed_samples)
        if position <= warmup_samples and warmup_samples:
            return position / warmup_samples
        if schedule == "constant":
            return 1.0
        if schedule != "cosine":
            raise ValueError("SFT supports constant or cosine sample schedules")
        fraction = (position - warmup_samples) / max(1, self.max_samples - warmup_samples)
        return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * fraction)) / 2
