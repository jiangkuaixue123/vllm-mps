import gc
from typing import Dict, List, Optional

import torch
import torch.distributed
from torch import nn
from vllm import envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.worker_base import WorkerBase

from vllm_mps.v1.model_runner import MPSModelRunner

logger = init_logger(__name__)


class MPSWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # TODO: Support mps profile    
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace

        # if envs.VLLM_TORCH_PROFILER_DIR:
        #     torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
        #     logger.info("Profiling enabled. Traces will be saved to: %s",
        #                 torch_profiler_trace_dir)

        #     experimental_config = torch_npu.profiler._ExperimentalConfig(
        #         export_type=torch_npu.profiler.ExportType.Text,
        #         profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        #         msprof_tx=False,
        #         aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        #         l2_cache=False,
        #         op_attr=False,
        #         data_simplification=False,
        #         record_op_args=False,
        #         gc_detect_threshold=None,
        #     )

        #     self.profiler = torch_npu.profiler.profile(
        #         activities=[
        #             torch_npu.profiler.ProfilerActivity.CPU,
        #             torch_npu.profiler.ProfilerActivity.NPU,
        #         ],
        #         with_stack=True,
        #         profile_memory=True,
        #         with_modules=True,
        #         experimental_config=experimental_config,
        #         on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        #             torch_profiler_trace_dir))
        # else:
        #     self.profiler = None

    def init_device(self):
        if self.device_config.device.type == "mps":
            self.device = torch.device("mps")
            current_platform.empty_cache()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        set_random_seed(self.model_config.seed)

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = MPSModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        logger.warning("jcz determing available memory")
        kv_caches: Dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, FullAttentionSpec):
                dtype = layer_spec.dtype

                # Use an empty tensor instead of `None`` to force Dynamo to pass
                # it by reference, rather by specializing on the value ``None``.
                mps_k_cache = torch.tensor([], dtype=dtype, device=self.device)
                mps_v_cache = torch.tensor([], dtype=dtype, device=self.device)

                kv_caches[layer_name] = (mps_k_cache, mps_v_cache)
            else:
                raise NotImplementedError

        runner_kv_caches: List[torch.Tensor] = []
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            runner_kv_caches)

        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        total_mps_memory = torch.mps.recommended_max_memory()
        logger.warning(f"Avilable memory: {torch.mps.current_allocated_memory()}")
        # current_platform.empty_cache()
        usable_memory_size = total_mps_memory * self.cache_config.gpu_memory_utilization - \
            torch.mps.current_allocated_memory()
        npu_kv_cache_bytes = max(usable_memory_size, 0)
        logger.warning(
            f"Available memory: {usable_memory_size}, total memory: {total_mps_memory}"
        )
        return int(npu_kv_cache_bytes)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.rank == 0 else None

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        # TODO: Currently do nothing here.
        # if not self.model_config.enforce_eager:
        #     self.model_runner.capture_model()

        # # Reset the seed to ensure that the random state is not affected by
        # # the model initialization and profiling.
        # set_random_seed(self.model_config.seed)
        pass

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return self.model_runner.get_kv_cache_spec()

    def initialize_cache(self, kv_cache_configs: List[KVCacheConfig]) -> None:
        """Allocate MPS KV cache with the specified kv_cache_config."""
        kv_cache_config = kv_cache_configs[self.rank]
        logger.warning(f"jcz initialize_cache kv_cache_configs:{kv_cache_configs}")
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    # TODO: Support mps profile
    # def profile(self, is_start: bool = True):
    #     if self.profiler is None:
    #         raise RuntimeError("Profiler is not enabled.")
    #     if is_start:
    #         self.profiler.start()
    #     else:
    #         self.profiler.stop()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)


def init_worker_distributed_environment(
        parallel_config: ParallelConfig,
        rank: int,
        distributed_init_method: Optional[str] = None,
        local_rank: int = -1,
        backend: str = "gloo") -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)