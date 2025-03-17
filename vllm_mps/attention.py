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
class MPSAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.


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
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
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
        attn_metadata: MPSAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        logger.warning(f"jcz layer:{layer}")
        logger.warning(f"jcz 1 query:{query.shape}, key:{key.shape}, value:{value.shape}, kv_cache:{kv_cache.shape}")
        num_tokens = query.shape[0]
        if output is None:
            logger.warning("jcz 2 output is None")
            output = torch.empty(num_tokens,
                                self.num_heads,
                                self.head_size,
                                dtype=query.dtype,
                                device=query.device)
        
        if attn_metadata is None:
            logger.warning("MPS backend does not support attn_metadata is None")
            return output.view(num_tokens, self.hidden_size)

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")
        # View q k v to BSH.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        # TODO: Remove this contiguous in the future.
        value = value.contiguous()
        logger.warning(f"jcz 2 query:{query.shape}, key:{key.shape}, value:{value.shape}, kv_cache:{kv_cache.shape}")

        if hasattr(layer, 'quant_method'):
            # TODO: Add attr (num_prefills, prefill_metadata, decode_metadata) to AscendMetadata
            pass
        else:
            if kv_cache.numel() > 0:
                key_cache, value_cache = kv_cache[0], kv_cache[1]
                num_blocks, block_size, _ = key_cache.shape
                key_cache = key_cache.view(num_blocks, block_size,
                                           self.num_kv_heads, self.head_size)
                value_cache = value_cache.view(num_blocks, block_size,
                                               self.num_kv_heads,
                                               self.head_size)
                slots = attn_metadata.slot_mapping
                logger.warning(f"jcz 3 slot:{slots}")
                # torch_npu._npu_reshape_and_cache(key=key,
                #                                  value=value,
                #                                  key_cache=key_cache,
                #                                  value_cache=value_cache,
                #                                  slot_indices=slots)
            # # use paged attention
            # torch_npu._npu_paged_attention_splitfuse(
            #     query=query,
            #     key_cache=key_cache,
            #     value_cache=value_cache,
            #     mask=attn_metadata.attn_mask,
            #     block_table=attn_metadata.block_tables,
            #     seq_len=attn_metadata.seq_lens,
            #     context_lens=attn_metadata.context_lens,
            #     num_kv_heads=self.num_kv_heads,
            #     num_heads=self.num_heads,
            #     scale_value=self.scale,
            #     out=output)
        return output.view(num_tokens, self.hidden_size)


class MPSAttentionBackend(AttentionBackend):

    # accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "MPS_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> Type["MPSAttentionBackendImpl"]:
        return MPSAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["MPSAttentionMetadata"]:
        return MPSAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)