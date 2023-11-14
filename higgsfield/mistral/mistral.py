import os
import functools
from pathlib import Path

import torch
import torch.distributed as dist

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
)

from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from transformers import (
    MistralForCausalLM,
    MistralConfig,
    default_data_collator,
)
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from optimum.bettertransformer import BetterTransformer

from higgsfield.checkpoint.fsdp_checkpoint import (
    save_distributed_model_rank0,
    fsdp_model_state_dict_rank0,
)

from higgsfield.mistral.mistral_utils import (
    load_mistral_from_checkpoint,
    load_mistral_from_config,
)

class Mistral(FSDP):
    def __init__(
        self,
        model_name,
        checkpoint_path=None,
        zero_stage=3,
        fast_attn=False,
        precision="bf16",
        cpu_init_rank0=False,
        cpu_offload=False,
        num_embeddings=None,
        cache_dir=None,
    ):
        
        rank = dist.get_rank()
        
        
        model = MistralForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
            
        if num_embeddings:
            model.resize_token_embeddings(num_embeddings)

            
        if fast_attn:
            #raise NotImplementedError("Fast attention is not supported yet")
            model = BetterTransformer.transform(model)
        
        fpSixteen = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

        bfSixteen_mixed = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        
        pure_bf16 = False
        if precision == "fp16":
            mixed_precision_policy = fpSixteen
            
        elif precision == "bf16":
            mixed_precision_policy = None
            pure_bf16 = True
            
        elif precision == "bf16_mixed":
            mixed_precision_policy = bfSixteen_mixed

        else:
            mixed_precision_policy = None
            
        if pure_bf16:
            model.to(torch.bfloat16) 

        wrapping_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                MistralDecoderLayer,
            }
        )
        
        if zero_stage == 0:
            sharding_strategy = ShardingStrategy.NO_SHARD
        
        elif zero_stage == 1:
            raise NotImplementedError("stage 1 is not supported. Only 0 2 3")
            
        elif zero_stage == 2:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            
        elif zero_stage == 3:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            raise NotImplementedError("stage can be only 0 2 3")

        if cpu_init_rank0 and rank != 0:
            param_init_fn = lambda module: module.to_empty(
                device=torch.device('cuda'),
                recurse=False,
            )
        else:
            param_init_fn = None
            
        if cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        super().__init__(
            model,
            auto_wrap_policy=wrapping_policy,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=cpu_init_rank0,
            param_init_fn=param_init_fn,
        )
        
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        check_fn = lambda submodule: isinstance(submodule, MistralDecoderLayer)

        apply_activation_checkpointing(
            self,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )
        
        fsdp = True
        self.precision = precision
        self.fsdp = fsdp
        self.model_name = model_name
        self.num_embeddings = num_embeddings
    
    def __call__(self, batch):
        local_rank = int(os.environ["LOCAL_RANK"])
        
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
            
        if self.precision == "fp16":
            with torch.cuda.amp.autocast(): 
                loss = super().__call__(**batch).loss
        else:
            loss = super().__call__(**batch).loss
            
        return loss
    
    def save_model(self, save_path):
        '''
            Save model's weight to master node 
                ~/.cache/higgsfield/{save_path}
        '''
        if "/" == save_path[0]:
            save_path = save_path[1:]
            
        head, tail = os.path.split(save_path)
        
        path = Path.home() / ".cache/higgsfield" / head
        path.mkdir(exist_ok=True, parents=True)
    
        save_distributed_model_rank0(path / tail, self)
        
    def save_huggingface_model(self, save_path):
        '''
            Save model's weight in huggingface format to master node 
                ~/.cache/higgsfield/{save_path}
        '''
        if "/" == save_path[0]:
            save_path = save_path[1:]
            
        head, tail = os.path.split(save_path)
        
        path = Path.home() / ".cache/higgsfield" / head
        path.mkdir(exist_ok=True, parents=True)
        cpu_state = fsdp_model_state_dict_rank0(self)
        
        if dist.get_rank() == 0:
           model = load_mistral_from_config(self.model_name, num_embeddings=self.num_embeddings)
           model.load_state_dict(cpu_state)
           model.save_pretrained(path / tail)
        
    def push_to_hub(self, repo_id, token):
        cpu_state = fsdp_model_state_dict_rank0(self)
        
        if dist.get_rank() == 0:
           model = load_mistral_from_config(self.model_name, num_embeddings=self.num_embeddings)
           model.load_state_dict(cpu_state) 
           model.push_to_hub(repo_id, token=token)