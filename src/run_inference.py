import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.stimulator import NeuroStimulator

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1" # or any supported model
LAYER_ID = 15 # Middle layers often hold "style" or "truthfulness" concepts

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    stimulator = NeuroStimulator(model, tokenizer, layer_id=LAYER_ID)

    # 1. Define the behavior we want to induce (The "Neurostimulation")
    # Example: Making the model obsessed with Shakespeare
    print("Calculating steering vector (Normal -> Shakespearean)...")
    stimulator.calculate_steering_vector(
        positive_text="Thou art a summer's day.", 
        negative_text="It is a sunny day."
    )

    prompt = "Tell me about learning to code."

    # 2. Run Normal Inference
    print("\n--- Normal Output ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(generated[0], skip_special_tokens=True))

    # 3. Run Stimulated Inference
    print("\n--- Neurostimulated Output (Strength 2.0) ---")
    with stimulator.stimulate(strength=2.0):
        generated_stim = model.generate(**inputs, max_new_tokens=50)
        print(tokenizer.decode(generated_stim[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
