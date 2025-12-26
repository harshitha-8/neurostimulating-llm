# Neurostimulating LLMs: Inference-Time Fine-Tuning

This repository implements **Inference-Time Activation Steering** (often metaphorically called "Neurostimulation"). 

Instead of fine-tuning the model's weights (which is slow and expensive), we calculate a "steering vector" representing a specific concept (e.g., honesty, creativity, conciseness) and inject it into the model's activations during the forward pass. This "stimulates" the model to behave in a specific way without permanent training.

## Features
- **Zero-Weight Updates**: Modify behavior instantly at runtime.
- **Layer-Specific Stimulation**: Target specific layers to influence style vs. reasoning.
- **Dynamic Control**: Adjust the "strength" of the stimulation on the fly.

## Structure
- `src/stimulator.py`: Contains the hook logic to inject vectors.
- `scripts/run_inference.py`: Demo showing how to steer a model's personality.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the demo: `python -m scripts.run_inference`
