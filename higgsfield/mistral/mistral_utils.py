import torch
from transformers import (
    MistralConfig,
    MistralForCausalLM,
)
from higgsfield.checkpoint import fsdp_model_state_dict_rank0

def load_mistral_from_config(model_name, num_embeddings=None):
    config = MistralConfig.from_pretrained(model_name)
    model = MistralForCausalLM(config)
    
    if num_embeddings:
        model.resize_token_embeddings(num_embeddings)
        
    return model

def load_mistral_from_checkpoint(model_name, checkpoint_path, num_embeddings=None):
    model = load_mistral_from_config(model_name, num_embeddings=num_embeddings)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    return model