from datasets import load_dataset 

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