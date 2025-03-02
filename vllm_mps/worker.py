import gc
from typing import Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.distributed

from torch import nn
from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger

from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)
from vllm.worker.model_runner_base import ModelRunnerBase

logger = init_logger(__name__)

class MPSWorker(LocalOrDistributedWorkerBase):
    """
    A worker class that executes (a partition of) the model on a MPS device.
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[ModelRunnerBase]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        ModelRunnerClass = Type
        self.model_runner: ModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            # **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)
