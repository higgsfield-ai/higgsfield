## Tutorial

This section runs through the API for common tasks in Large Language Models training. 

### Working with distributed model
Higgsfield provides simple primitives to work with distributed models.
```python
from higgsfield.llama import Llama70b
from higgsfield.loaders import LlamaLoader

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from datasets import load_dataset
```

`Llama70b` is ready to use sharded class of Llama 70b model. You can control sharding strategy with arguments.

```python
model = Llama70b(
    zero_stage=3,
    fast_attn=False,
    precision="bf16",
)
```
- `zero_stage` argument controls what sharding strategy to use. `zero_stage=3` is set to fully shard the model parameters, gradients and optimizer states. This makes the training of some very large models feasible and helps to fit larger models or batch sizes for our training job. This would come with the cost of increased communication volume. `zero_stage=2` shards only optimizer states and gradients reducing the communication overhead. For more information check [Deepspeed](https://arxiv.org/pdf/1910.02054.pdf)'s and [FSDP](https://arxiv.org/pdf/2304.11277.pdf) papers.

- `precision` argument supports flexible mixed precision training allowing for types such as bf16 or fp16. Former well-suited for deep learning tasks where numerical stability and convergence are essential. But currently bfloat16 is only available on Ampere GPUs, so you need to confirm native support before you use it.

- `fast_attn` leverages classical techniques (tiling, recomputation) to significantly speed up attention computation and reduce memory usage from quadratic to linear in sequence length.

### Preparing Data

```python
class AlpacaDataset:
    def __init__(self, dataset_name, split="train"):
        self.dataset = load_dataset(dataset_name, split=split)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        instruction = item["instruction"]
        
        if "input" in item.keys():
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:"
            )
        else:
            input = item["input"]
            
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            )
            
        completion = item["output"]
        
        return {
            "prompt": prompt,
            "completion": completion,
        }
```

```python
dataset = AplacaDataset("tatsu-lab/alpaca", split="train")

train_loader = LlamaLoader(
    alpaca,
    max_sequence_length=2048,
    batch_size=64*6,
)
```

### Optimizing the Model Parameters
Higgsfield's distributed model works with standard PyTorch training flow. 
Creating optimizer and learning scheduler.
```python
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
```

In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.

```python
for epoch in range(3):
    for i, batch in enumerate(train_loader):
        
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

    lr_scheduler.step()
```

### Saving Model
Saving pytorch model. 
```python
model.save("alpaca-70b/model.pt")
```

Saving in hugginface format or push it to the hub
```python
model.save_huggingface_model("alpaca-hf-70b")
```

Or push it the hub
```python
model.push_to_hub("alpaca-70b")
```

## Training stabilization techniques 
It's easy to use and customize different training techniques because we follow standard PyTorch workflow.

### Gradient accumulation

```python

grad_accumulation_steps = 16

for epoch in range(3):
    for i, batch in enumerate(train_loader):
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (i + 1) % grad_accumulation_steps == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()        
```
### Gradient clipping 

```python
from higgsfield.training import clip_grad_norm

max_grad_norm = 1.0

for epoch in range(3):
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm:
            clip_grad_norm(model, optimizer, max_grad_norm)

        optimizer.step()
```

### FP16 gradient scaling
```python
from higgsfield.training import Scaler, clip_grad_norm

scaler = Scaler(model)

for epoch in range(3):
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        scaler.scale(loss).backward()

        if max_grad_norm:
            clip_grad_norm(max_grad_norm, model, optimizer, scaler)

        scaler.step(optimizer)
        scaler.update()
```

## Monitoring

### Wandb support
You can use Wandb logic inside the project, the only exception and requirement would be to place it under the if condition `if params.rank == 0:`.

```python
import wandb

@experiment("alpaca")
def train(params):
    ...

    if params.rank == 0:
        wandb.init(
            project="My Llama2",
        )


    for epoch in range(1):
        for i, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            loss = model(batch)
                
            loss.backward()
            optimizer.step()
                
            if params.rank == 0:
                wandb.log({
                    "train/loss": loss.item(),
                })
```