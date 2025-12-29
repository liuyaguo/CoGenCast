import os
import torch
from models import My_Model

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'My_Model': My_Model,
        }
        self.device =args.device
        model = self._build_model().to(self.device)
        # 多卡时封装 DDP
        if getattr(args, "world_size", 1) > 1 and dist.is_initialized():
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        self.model = model

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
      return self.args.device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
