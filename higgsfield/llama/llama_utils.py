import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
)
from higgsfield.checkpoint import fsdp_model_state_dict_rank0

def load_llama_from_config(model_name):
    config = LlamaConfig.from_pretrained(model_name)
    model = LlamaForCausalLM(config)
    return model

def load_llama_from_checkpoint(model_name, checkpoint_path):
    model = load_llama_from_config(model_name)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    return model