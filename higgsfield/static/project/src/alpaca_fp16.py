import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from higgsfield.llama import Llama
from higgsfield.loaders import LlamaLoader 
from higgsfield.checkpoint import Checkpoint
from higgsfield.training import  clip_grad_norm, Scaler
from higgsfield.experiment import experiment, param

from src.dataset import AlpacaDataset

@experiment("alpaca_fp16")
@param("size", options=["7b", "13b", "70b"])
@param("num_epochs", default=1, description="Number of epochs")
def train(params):
    
    if params.size == "7b":
        model_name = "meta-llama/Llama-2-7b-hf"
    elif params.size == "13b":
        model_name = "meta-llama/Llama-2-13b-hf"
    elif params.size == "70b":
        model_name = "meta-llama/Llama-2-70b-hf"
    
    model = Llama(
        model_name=model_name,
        zero_stage=3,
        cpu_init_rank0=True,
        fast_attn=False,
        precision="fp16",
        cpu_offload=False,
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.0,
    )
    
    lr_scheduler = StepLR(
        optimizer,
        step_size=1,
        gamma=0.85,
    )
    
    scaler = Scaler(model)
    
    # ~/.cache/{project-name}/experiments/{experiment_name}/{run_name}/
    checkpoint = Checkpoint(
        model,
        optimizer,
        lr_scheduler,
        scaler,
    )
    
    dataset_name = "tatsu-lab/alpaca"
    dataset = AlpacaDataset(dataset_name, split="train")
    
    train_loader = LlamaLoader(
        dataset,
        max_sequence_length=2048,
        batch_size_per_gpu=1,
    )
    
    for epoch in range(params.num_epochs):
        for i, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            loss = model(batch)
            
            if params.rank == 0:
                print("Loss: ", loss)
            
            scaler.scale(loss).backward()    
            
            clip_grad_norm(1.0, model, optimizer, scaler)
            scaler.step(optimizer)
            scaler.update()

            if i % 30 == 0 or i == len(train_loader) - 1:
                checkpoint.save(epoch, i)
            
        lr_scheduler.step()
        
    model.save_huggingface_model("my-alpaca")
