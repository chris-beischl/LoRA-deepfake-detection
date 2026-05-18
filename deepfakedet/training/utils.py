# Copied from https://github.com/chris-beischl/liteViT
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, SequentialLR


def build_lr_scheduler_with_warmup(
    optimizer: Optimizer,
    warmup_epochs: int,
    t_max: int,
    scheduler: Callable[[Optimizer, int], LRScheduler],
) -> SequentialLR:
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1 / warmup_epochs, total_iters=warmup_epochs
    )
    main_scheduler = scheduler(optimizer, t_max - warmup_epochs)
    return SequentialLR(
        optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )
