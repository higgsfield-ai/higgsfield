import copy
import torch
from torch.utils.data import Dataset

class CompletionDataset(Dataset):
    '''
    def __getitem__(self, idx):
        ...
        return {
            "prompt": prompt,
            "completion": completion,
        }
    '''
    pass

class LMDataset(Dataset):
    '''
    def __getitem__(self, idx):
        ...
        return "whatever sequence you want to return as a string"
    '''
    pass

class TorchMultiTurnDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_sequence_length):
        self.dataset             = dataset
        self.tokenizer           = tokenizer
        self.max_sequence_length = max_sequence_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        items = self.dataset[idx]
        
        IGNORE_INDEX = -100
        
        multi_labels  = []
        multi_example = []
        for item in items:
            prompt     = item["prompt"]
            completion = item["completion"]
            
            example = prompt + completion
            prompt = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            example = self.tokenizer.encode(example)
            example.append(self.tokenizer.eos_token_id)
            example = torch.tensor(
                example, dtype=torch.int64
            )
                
            labels = copy.deepcopy(example)
            labels[: len(prompt)] = -1
            
            multi_example.append(example)
            multi_labels.append(labels)
            
        example = torch.cat(multi_example)
        labels  = torch.cat(multi_labels)
        
        padding = self.max_sequence_length - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            labels  = torch.cat((labels,  torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_sequence_length]
            labels  = labels[:  self.max_sequence_length]
        
        label_mask = labels.ge(0)
        labels[~label_mask] = IGNORE_INDEX
        
        example_mask = example.ge(0)
        example[~example_mask] = 0
        example_mask = example_mask.float()
        
        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

class TorchCompletionDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_sequence_length):
        self.dataset             = dataset
        self.tokenizer           = tokenizer
        self.max_sequence_length = max_sequence_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        prompt = item["prompt"]
        completion = item["completion"]
        
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        
        example = prompt + completion
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_sequence_length - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_sequence_length]
            
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        label_mask = labels.ge(0)
        labels[~label_mask] = IGNORE_INDEX
        
        example_mask = example.ge(0)
        example[~example_mask] = 0
        example_mask = example_mask.float()
        
        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }

class TorchLMDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_sequence_length):
        self.dataset   = dataset
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
       
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]
        
        IGNORE_INDEX = -100
        
        example = self.tokenizer.encode(x, add_special_tokens=False)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        
        padding = self.max_sequence_length - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_sequence_length]
       
        labels = copy.deepcopy(example)
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()       
        
        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
        
        