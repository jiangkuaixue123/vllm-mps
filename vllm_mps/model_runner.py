from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Type, TypeVar, Union)

import torch
import torch.distributed
import torch.nn as nn
import torch_npu

from vllm.logger import init_logger

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceGroupMetadata, SequenceOutput)
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,)

logger = init_logger(__name__)

TModelInputForMPS = TypeVar('TModelInputForMPS', bound="ModelInputForMPS")

@dataclass(frozen=True)
class ModelInputForMPS(ModelRunnerInputBase):
    pass


class MPSModelRunner(ModelRunnerBase[ModelInputForMPS]):

    def __init__(
        self,
        vllm_config: VllmConfig
    ):
        ModelRunnerInputBase.__init__(self, vllm_config=vllm_config)
        

    def load_model(self) -> None:
        self.device = self.device_config.device