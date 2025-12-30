import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name, device_map="auto"):
    """
    Loads the LLM and tokenizer with standard settings for inference.
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token is set for models that lack it (like Llama 2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer
