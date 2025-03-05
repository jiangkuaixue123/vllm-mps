from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Type, TypeVar, Union)

import torch
import torch.distributed
import torch.nn as nn

from vllm.logger import init_logger

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceGroupMetadata, SequenceOutput)
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

TModelInputForMPS = TypeVar('TModelInputForMPS', bound="ModelInputForMPS")

@dataclass(frozen=True)
class ModelInputForMPS(ModelRunnerInputBase):
    pass

@dataclass(frozen=True)
class ModelInputForMPSWithSamplingMetadata(ModelInputForMPS):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None
    is_prompt: Optional[bool] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        # tensor_dict = {
        #     "input_tokens": self.input_tokens,
        #     "input_positions": self.input_positions,
        #     "token_type_ids": self.token_type_ids,
        #     "multi_modal_kwargs": self.multi_modal_kwargs,
        # }
        # _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        # _add_sampling_metadata_broadcastable_dict(tensor_dict,
        #                                           self.sampling_metadata)
        # return tensor_dict
        return None

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForMPSWithSamplingMetadata":
        # tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        # if attn_backend is not None:
        #     tensor_dict = _init_attn_metadata_from_tensor_dict(
        #         attn_backend, tensor_dict)
        # return cls(**tensor_dict)
        None


class MPSModelRunner(ModelRunnerBase[ModelInputForMPS]):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ):
        ModelRunnerBase.__init__(self, vllm_config=vllm_config)

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForMPSWithSamplingMetadata:
        model_input = \
            ModelInputForMPSWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            )
        return model_input

    def load_model(self) -> None:
        # with torch.device
        self.device = torch.device("mps")
        self.model = get_model(vllm_config=self.vllm_config).to(self.device)

        # self.model_memory_usage = m.consumed_memory
        # logger.info("Loading model weights took %.4f GB",
        #             self.model_memory_usage / float(2**30))
    
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForMPSWithSamplingMetadata:
        """
        Prepare the inputs to ModelRunnerBase.execute_model from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        return None

    def get_model(self) -> nn.Module:
        return self.model