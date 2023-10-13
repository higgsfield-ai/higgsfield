import torch
import gc

def get_tensors_from_gc(gpu_only=True):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue
            
            if tensor.is_cuda or not gpu_only:
                yield tensor
        except Exception:  # nosec B112 pylint: disable=broad-exception-caught
            continue

def empty_cache():
    cnt = 0
    for obj in get_tensors_from_gc():
        obj.detach()
        obj.grad = None
        obj.untyped_storage().resize_(0)
    cnt += 1
    gc.collect()
    torch.cuda.empty_cache()