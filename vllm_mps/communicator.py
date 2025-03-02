from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase


class MPSCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        # init device according to local rank
        self.device = torch.device("mps")