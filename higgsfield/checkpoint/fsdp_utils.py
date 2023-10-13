from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)

from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    StateDictType,
)

fullstate_save_policy = FullStateDictConfig(
    offload_to_cpu=True,
    rank0_only=True,
)

def fsdp_model_state_dict_rank0(model):
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()
        
    return cpu_state