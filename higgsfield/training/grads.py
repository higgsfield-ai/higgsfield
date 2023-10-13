def clip_grad_norm(max_grad_norm, model, optimizer, scaler=None):
    model = model
    
    if scaler:
        scaler.unscale_(optimizer)
        
    if hasattr(optimizer, 'clip_grad_norm'):
        optimizer.clip_grad_norm(max_grad_norm)
        
    elif hasattr(model.model, 'clip_grad_norm_'):
        model.clip_grad_norm_(max_grad_norm)      