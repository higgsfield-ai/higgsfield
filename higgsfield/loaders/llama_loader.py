import torch.distributed as dist

from torch.utils.data import (
    DistributedSampler, 
    DataLoader
)

from transformers import (
    LlamaTokenizer, 
    default_data_collator
)

from higgsfield.dataset import TorchCompletionDataset

class HiggsfieldSampler(DistributedSampler):
    def __init__(
        self, 
        dataset, 
        shuffle=True,
        seed=0, 
        drop_last=False
    ):
        rank=dist.get_rank()
        num_replicas=dist.get_world_size()
        
        super(HiggsfieldSampler, self).__init__(
            dataset=dataset, 
            num_replicas=num_replicas,
            rank=rank, 
            shuffle=shuffle,
            seed=seed, 
            drop_last=drop_last,
        )

class LlamaLoader(DataLoader):
    def __init__(
        self,
        dataset, 
        tokenizer=LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf"),
        max_sequence_length=2048,
        batch_size_per_gpu=1,
        shuffle=True, 
        seed=0,
        num_workers=0, 
        pin_memory=False, 
        drop_last=False,
        timeout=0, 
        worker_init_fn=None,
        multiprocessing_context=None, 
        *, 
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device=""
    ):
    
        dataset = TorchCompletionDataset(
            dataset,
            tokenizer,
            max_sequence_length,
        )
        
        sampler = HiggsfieldSampler(dataset, shuffle=shuffle, seed=seed,)
        
        super(LlamaLoader, self).__init__(
            dataset, 
            batch_size=batch_size_per_gpu,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory, 
            drop_last=drop_last,
            timeout=timeout, 
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context, 
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device 
        )
        