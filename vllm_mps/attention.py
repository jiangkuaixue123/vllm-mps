from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.logger import init_logger

logger = init_logger(__name__)

@dataclass
class MPSMetadata(AttentionMetadata):
    @property
    def prefill_metadata(self) -> Optional["MPSMetadata"]:
        pass
    
    @property
    def decode_metadata(self) -> Optional["MPSMetadata"]:
        pass

class MPSMetadataBuilder(CommonMetadataBuilder[MPSMetadata]):
    pass

class MPSAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "MPS backend does not support block-sparse attention.")
        if logits_soft_cap is not None:
            logger.warning_once("MPS backend does not support logits soft cap. "
                                "Outputs may be slightly off.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)
        
        # supported_head_sizes = PagedAttention.get_supported_head_sizes()
        # if head_size not in supported_head_sizes:
        #     raise ValueError(
        #         f"Head size {head_size} is not supported by PagedAttention. "
        #         f"Supported head sizes are: {supported_head_sizes}.")
        # if kv_cache_dtype != "auto":
        #     raise NotImplementedError(
        #         "Torch SDPA backend does not support FP8 KV cache. "
        #         "Please use xFormers backend instead.")
        self.attn_type = attn_type

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MPSMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_type = self.attn_type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")
        
        return None


class MPSAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "MPS"

    @staticmethod
    def get_impl_cls() -> Type["MPSAttentionBackendImpl"]:
        return MPSAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["MPSMetadata"]:
        return MPSMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_builder_cls() -> Type["MPSMetadataBuilder"]:
        return MPSMetadataBuilder

    @classmethod
    def make_metadata_builder(cls, *args, **kwargs) -> "MPSMetadataBuilder":
        return cls.get_builder_cls()(*args, **kwargs)