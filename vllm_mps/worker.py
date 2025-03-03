import gc
from typing import Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.distributed

from torch import nn
from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, VllmConfig)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)
from vllm.sequence import ExecuteModelRequest
from vllm.platforms import current_platform
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.worker.model_runner_base import ModelRunnerBase

from vllm_mps.model_runner import MPSModelRunner

logger = init_logger(__name__)

class MPSCacheEngine:
    """Manages the KV cache for MPS backend.

    This class is responsible for initializing and managing MPS KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.
    """
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 device_config: DeviceConfig) -> None:
        assert device_config.device_type == "cpu"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for CPU backend, because we want to reuse KV cache management
        # in the scheduler.
        self.num_cpu_blocks = cache_config.num_gpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        # Initialize the cache.
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        return dtype_size * total


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

        ModelRunnerClass: Type[ModelRunnerBase] = MPSModelRunner
        self.model_runner: ModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            # **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)
        logger.info("jcz MPSWorker init 3")
        self.cache_engine: MPSCacheEngine
        self.cpu_cache: Optional[List[List[torch.Tensor]]] = None

    def init_device(self) -> None:
        if self.device_config.device.type == "mps":
            current_platform.set_device(self.device)

            current_platform.empty_cache()
            # self.init_npu_memory = current_platform.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        set_random_seed(self.model_config.seed)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """
        Initialize the KV cache with the given size in blocks.
        """
        assert (num_cpu_blocks == 0
                ), f"{type(self)} does not support swappable cache"
        
        num_cpu_blocks = num_gpu_blocks
        logger.info("jcz initialize_cache num_gpu_blocks:%d num_cpu_blocks:%d",
                    num_gpu_blocks, num_cpu_blocks)
        
        self._validate_num_cpu_blocks(num_cpu_blocks)
        self.cache_config.num_gpu_blocks = num_cpu_blocks
        self.cache_config.num_cpu_blocks = 0

        # Initialize the cache.
        self._init_cache_engine()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def load_model(self) -> None:
        """Load model onto target device."""
        self.model_runner.load_model()

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # For MPS device, the block number will be calculated based on the
        # cpu_kvcache_space.
        cache_block_size = self.get_cache_block_size_bytes()
        num_cpu_blocks = int(self.cache_config.cpu_kvcache_space_bytes //
                             cache_block_size)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        # Note: To reuse the cache management procedure,
        # use cpu cache as 'gpu cache'.
        num_gpu_blocks = num_cpu_blocks
        num_cpu_blocks = 0
        return num_gpu_blocks, num_cpu_blocks

    def get_cache_block_size_bytes(self) -> int:
        return MPSCacheEngine.get_cache_block_size(
            self.cache_config.block_size, self.cache_config.cache_dtype,
            self.model_config, self.parallel_config)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # raise NotImplementedError(
        #     "LoRA is not implemented for MPS backend currently.")
        pass

    def remove_lora(self, lora_id: int) -> bool:
        # raise NotImplementedError(
        #     "LoRA is not implemented for MPS backend currently.")
        pass

    def pin_lora(self, lora_id: int) -> bool:
        # raise NotImplementedError(
        #     "LoRA is not implemented for MPS backend currently.")
        pass

    def list_loras(self) -> Set[int]:
        # raise NotImplementedError(
        #     "LoRA is not implemented for MPS backend currently.")
        pass

    @property
    def do_metadata_broadcast(self) -> bool:
        logger.info("jcz do_metadata_broadcast")
        return False
    
    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        """
        Gets the list of kv caches to pass to the worker's model runner. Each
        element in the list is a kv cache corresponding to a particular virtual
        engine (PP stream). Used by the default `execute_model`. If the worker's
        model runner does not follow the ModelRunnerBase interface, then inherit
        from WorkerBase instead.
        """
        return self.cpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        """
        Prepare the inputs to WorkerBase.execute_worker from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        assert execute_model_req is not None
        virtual_engine: int = execute_model_req.virtual_engine
        num_seq_groups: int = len(execute_model_req.seq_group_metadata_list)
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)
        assert len(execute_model_req.blocks_to_swap_in) == 0
        assert len(execute_model_req.blocks_to_swap_out) == 0
        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
        )
        
    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        """
        Process an execution request.
        """
        raise NotImplementedError
    
    def _validate_num_cpu_blocks(self, num_cpu_blocks: int) -> None:
        """Raise errors if the num_cpu_blocks is invalid.
        """
        if num_cpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `VLLM_CPU_KVCACHE_SPACE` when "
                             "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_cpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`VLLM_CPU_KVCACHE_SPACE` or decreasing `max_model_len` when "
                "initializing the engine.")
