import torch
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

class Scaler(object):
    def __init__(self, model):
        if model.fsdp:
            self.scaler = ShardedGradScaler()
        else:
            return torch.cuda.amp.GradScaler()
 