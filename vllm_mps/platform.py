from typing import TYPE_CHECKING, Optional, Tuple

import torch

from vllm.config import VllmConfig
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
    def get_device_capability(cls, device_id: int = 0):
        logger.info("jcz get_device_capability")
        return None
    
    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        # if use_mla:
            # return "vllm_ascend.attention.AscendMLAAttentionBackend"
        # return "vllm_ascend.attention.AscendAttentionBackend"
        logger.info("jcz get_attn_backend_cls")
        return ""
    
    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Checks whether pin memory is available on the current platform."""
        logger.info("jcz is_pin_memory_available")
        return False
    
    @classmethod
    def inference_mode(cls):
        logger.info("jcz is_pin_memory_available")
        return torch.inference_mode()
    
    @classmethod
    def empty_cache(self):
        logger.info("jcz is_pin_memory_available")
        return torch.mps.empty_cache()
    
    @classmethod
    def synchronize(cls):
        logger.info("jcz is_pin_memory_available")
        torch.mps.synchronize()
    
    @classmethod
    def set_device(cls, device: torch.device):
        logger.info(f"jcz set_device:{device}")
        torch.set_device("mps")

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_mps.communicator.MPSCommunicator"
    
    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        logger.info("jcz check_and_update_config")
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_mps.worker.MPSWorker"
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            # TODO: Set block_size to 128 will lead unexpected accuracy issue in mla case.  Please set block_size to 128 back once the problem is fixed.
            cache_config.block_size = 16
    
    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True