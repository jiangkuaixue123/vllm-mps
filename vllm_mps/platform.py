from typing import TYPE_CHECKING, Optional, Tuple

import torch

from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import Platform, PlatformEnum

logger = init_logger(__name__)

class MPSPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "mps"
    device_type: str = "mps"
    simple_compile_backend: str = "inductor"
    ray_device_key: str = "MPS"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "MPS"

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        logger.info("jcz get_device_capability")
        return None
    
    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        if use_mla:
            raise NotImplementedError
        return "vllm_mps.attention.MPSAttentionBackend"
    
    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Checks whether pin memory is available on the current platform."""
        logger.info("Currently MPS not support pin_memory")
        return False
    
    @classmethod
    def inference_mode(cls):
        return torch.inference_mode()
    
    @classmethod
    def empty_cache(self):
        return torch.mps.empty_cache()
    
    @classmethod
    def synchronize(cls):
        torch.mps.synchronize()
    
    @classmethod
    def set_device(cls, device: torch.device):
        torch.set_device(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_mps.communicator.MPSCommunicator"
    
    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        import vllm.envs as envs
        from vllm.utils import GiB_bytes
        logger.info("jcz check_and_update_config")
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = "vllm_mps.v1.worker.MPSWorker"
            else:
                parallel_config.worker_cls = "vllm_mps.worker.MPSWorker"
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            # TODO: Set block_size to 128 will lead unexpected accuracy issue in mla case.  Please set block_size to 128 back once the problem is fixed.
            cache_config.block_size = 16
        
        # Currently MPS use VLLM_CPU_KVCACHE_SPACE to represent the kv cache space.
        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
        logger.info(f"jcz kv_cache_space:{kv_cache_space}")
        if kv_cache_space >= 0:
            if kv_cache_space == 0:
                # cache_config.cpu_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
                cache_config.cpu_kvcache_space_bytes = torch.mps.recommended_max_memory()
                logger.warning(
                    f"Environment variable VLLM_CPU_KVCACHE_SPACE (GB) "
                    f"for MPS backend is not set, using torch.mps.recommended_cache_size:"
                    f"{cache_config.cpu_kvcache_space_bytes} by default.")
            else:
                cache_config.cpu_kvcache_space_bytes = kv_cache_space * GiB_bytes  # type: ignore # noqa
        else:
            raise RuntimeError(
                "Invalid environment variable VLLM_CPU_KVCACHE_SPACE"
                f" {kv_cache_space}, expect a positive integer value.")
        
        if (parallel_config.distributed_executor_backend is not None
                and parallel_config.distributed_executor_backend != "uni"):
            logger.warning(("%s is not supported on MPS, fallback to uni "
                            "distributed executor backend."),
                           parallel_config.distributed_executor_backend)
            parallel_config.distributed_executor_backend = "uni"
        
        compilation_config = vllm_config.compilation_config
        if compilation_config.level != CompilationLevel.NO_COMPILATION:
            logger.warning(
                "Compilation level %s is not supported on MPS now, forcing compilation level to NO_COMPILATION",
                compilation_config.level)
            compilation_config.level = CompilationLevel.NO_COMPILATION
    
    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True