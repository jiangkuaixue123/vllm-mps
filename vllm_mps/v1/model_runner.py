import gc
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.distributed
import numpy as np
from torch import nn
from vllm import envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.forward_context import set_forward_context
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LayerBlockType, cdiv, is_pin_memory_available)
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.model_executor.model_loader import get_model

from vllm_mps.attention import MPSAttentionBackend, MPSAttentionMetadata

logger = init_logger(__name__)

class MPSModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        # self.pin_memory = False
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype

        self.kv_caches: List[torch.Tensor] = []

        self.model: nn.Module

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

        # Multi-modal data support
        # self.input_registry = INPUT_REGISTRY
        # self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope

        self.is_multimodal_model = model_config.is_multimodal_model
        self.block_size = cache_config.block_size
        self.max_num_reqs = scheduler_config.max_num_seqs
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )
        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}

        # Persistent buffers for CUDA graphs.
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device)

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.input_ids_np = self.input_ids_cpu.numpy()
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config).to(self.device)
        
    def get_model(self) -> nn.Module:
        return self.model

    def initialize_kv_cache(self, kv_cache_configs: List[KVCacheConfig]) -> None:
        if len(kv_cache_configs.groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        
        kv_caches: Dict[str, torch.Tensor] = {}
        
        for layer_name, layer_spec in kv_cache_configs.kv_cache_spec.items():
            tensor_config = kv_cache_configs.tensors[layer_name]
            assert tensor_config.size % layer_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // layer_spec.page_size_bytes
            if isinstance(layer_spec, FullAttentionSpec):
                kv_cache_shape = MPSAttentionBackend.get_kv_cache_shape(
                    num_blocks, layer_spec.block_size, layer_spec.num_kv_heads,
                    layer_spec.head_size)
                dtype = layer_spec.dtype
                kv_caches[layer_name] = torch.zeros(kv_cache_shape,
                                                    dtype=dtype,
                                                    device=self.device)
            else:
                raise NotImplementedError
    
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)
        
    def get_kv_cache_spec(self) -> KVCacheSpec:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: KVCacheSpec = {}
        for layer_name, attn_module in forward_ctx.items():
            logger.warning("layer_name: %s", layer_name)
            logger.warning("attn_module: %s", attn_module)
            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention, MLA.
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                logger.info("AttentionType.DECODER")
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec
    
    def profile_run(self) -> None:
        # TODO: Need implement
        pass

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: List[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            start_index = (len(req_state.block_ids) -
                           len(req_data.new_block_ids))
            self.input_batch.block_table.append_row(req_index, start_index,
                                                    req_data.new_block_ids)
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Tuple[MPSAttentionMetadata, torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        max_num_scheduled_tokens = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens[i] = num_tokens
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        max_seq_len = self.seq_lens_np[:num_reqs].max()

        # Copy the tensors to the GPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens],
                non_blocking=True)
        query_start_loc = self.query_start_loc_cpu[:num_reqs + 1].to(
            self.device, non_blocking=True)
        seq_lens = self.seq_lens_cpu[:num_reqs].to(self.device,
                                                   non_blocking=True)
        slot_mapping = self.slot_mapping_cpu[:total_num_scheduled_tokens].to(
            self.device, non_blocking=True).long()

        # Prepare for cascade attention if needed.
        # Currently not support cascade attention.
        common_prefix_len = 0
        # common_prefix_len = self._compute_cascade_attn_prefix_len(
        #     num_scheduled_tokens,
        #     scheduler_output.num_common_prefix_blocks,
        # )
        use_cascade = common_prefix_len > 0
        if use_cascade:
            # TODO: Optimize.
            cu_prefix_query_lens = torch.tensor(
                [0, total_num_scheduled_tokens],
                dtype=torch.int32,
                device=self.device)
            prefix_kv_lens = torch.tensor([common_prefix_len],
                                          dtype=torch.int32,
                                          device=self.device)
            suffix_kv_lens = (self.seq_lens_np[:num_reqs] - common_prefix_len)
            suffix_kv_lens = torch.from_numpy(suffix_kv_lens).to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None

        attn_metadata = MPSAttentionMetadata(
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=(
                self.input_batch.block_table.get_device_tensor()[:num_reqs]),
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
        )

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if use_spec_decode:
            logits_indices = self._calc_spec_decode_metadata(
                scheduler_output, cu_num_tokens)
        else:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return attn_metadata, logits_indices

    @current_platform.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, torch.Tensor]:
        logger.warning(f"jcz execute_model scheduler_output: {scheduler_output}")
        self._update_states(scheduler_output)
        logger.warning(f"jcz end self._update_states")
        
        if self.is_multimodal_model:
            # Currently not supported mm.
            # self._execute_encoder(scheduler_output)
            # encoder_outputs = self._gather_encoder_outputs(scheduler_output)
            raise ValueError("Not support multimodal model.")
        else:
            encoder_outputs = []
        
        # Prepare the decoder inputs.
        attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # Currenty not support graph
        # if (self.use_cuda_graph
        #         and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
        #     # Use piecewise CUDA graphs.
        #     # Add padding to the batch size.
        #     num_input_tokens = self.vllm_config.pad_for_cudagraph(
        #         num_scheduled_tokens)
        # else:
        #     # Eager mode.
        #     num_input_tokens = num_scheduled_tokens
        num_input_tokens = num_scheduled_tokens
        attn_metadata.num_input_tokens = num_input_tokens

        if self.is_multimodal_model:
            # # NOTE(woosuk): To unify token ids and soft tokens (vision
            # # embeddings), we always use embeddings (rather than token ids)
            # # as input to the multimodal model, even when the input is text.
            # input_ids = self.input_ids[:num_scheduled_tokens]
            # if encoder_outputs:
            #     inputs_embeds = self.model.get_input_embeddings(
            #         input_ids, encoder_outputs)
            # else:
            #     inputs_embeds = self.model.get_input_embeddings(input_ids)
            # # TODO(woosuk): Avoid the copy. Optimize.
            # self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            # inputs_embeds = self.inputs_embeds[:num_input_tokens]
            # input_ids = None
            raise ValueError("Not support multimodal model.")
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(
                    v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_input_tokens]
                for k, v in self.intermediate_tensors.items()
            })
        
        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        logger.warning(f"jcz begin hidden_states input_ids: {input_ids} "
                       f"positions: {positions}"
                       f"inputs_embeds: {inputs_embeds}")
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=self.kv_caches,
                attn_metadata=None,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        
        logger.warning("jcz end hidden_states")
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            return hidden_states

        hidden_states = hidden_states[:num_scheduled_tokens]
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.get_sampling_metadata(
            scheduler_output.scheduled_spec_decode_tokens)
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states,
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_mask = sampled_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            # TODO(woosuk): Optimize this.
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in sampled_token_ids[valid_mask].split(gen_lens)
            ]

        if not self.use_spec_decode:
            spec_token_ids = None
        else:
            spec_token_ids = self.generate_draft_token_ids(
                valid_sampled_token_ids)

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
        )
        return model_runner_output