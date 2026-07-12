from .evaluator_unified import unified_evaluate

def Dual_evaluator(model, criterion, postprocessors,loader_val, device,args,thr=0.3):
    
    return unified_evaluate(model, criterion, postprocessors,loader_val, device,args,thr=0.3)