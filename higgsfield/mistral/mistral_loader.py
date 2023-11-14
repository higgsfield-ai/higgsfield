import torch.distributed as dist

from torch.utils.data import (
    DistributedSampler, 
    DataLoader
)

from transformers import (
    AutoTokenizer, 
    default_data_collator
)

from higgsfield.dataset import TorchCompletionDataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|unk|>"

def get_tokenizer(model_name, max_length, cache_dir=None):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        pad_token=DEFAULT_PAD_TOKEN,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    
    return tokenizer

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

class MistralLoader(DataLoader):
    def __init__(
        self,
        dataset, 
        tokenizer=None,
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
        
        if not tokenizer:
            tokenizer = get_tokenizer("mistralai/Mistral-7B-v0.1", max_sequence_length)
    
        dataset = TorchCompletionDataset(
            dataset,
            tokenizer,
            max_sequence_length,
        )
        
        sampler = HiggsfieldSampler(dataset, shuffle=shuffle, seed=seed,)
        
        super(MistralLoader, self).__init__(
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
 