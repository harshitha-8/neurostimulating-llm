import torch
from contextlib import contextmanager

class NeuroStimulator:
    def __init__(self, model, tokenizer, layer_id):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_id = layer_id
        self.steering_vector = None
        self.hook_handle = None

    def calculate_steering_vector(self, positive_text, negative_text):
        """
        Creates the 'stimulus' by finding the difference in activation 
        between a positive example and a negative example.
        """
        print(f"Extracting vectors from layer {self.layer_id}...")
        
        # Helper to get activation
        def get_activation(text):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            # Access hidden state of the specific layer. 
            # Shape: [batch, seq_len, hidden_dim]. We take the last token.
            return outputs.hidden_states[self.layer_id][0, -1, :]

        pos_act = get_activation(positive_text)
        neg_act = get_activation(negative_text)
        
        # The vector is the direction from negative to positive
        self.steering_vector = pos_act - neg_act
        return self.steering_vector

    def _steering_hook(self, module, input, output):
        """
        The hook function that adds the steering vector to the layer's output.
        """
        if self.steering_vector is not None:
            # Add vector to all tokens in the sequence (broadcasting)
            # Strength (coeff) can be adjusted here or passed dynamically
            coeff = getattr(self, 'current_strength', 1.0)
            return output + (self.steering_vector * coeff)

    @contextmanager
    def stimulate(self, strength=1.0):
        """
        Context manager to apply the stimulation only during this block.
        """
        self.current_strength = strength
        
        # Register the hook on the specified layer
        layer = self.model.model.layers[self.layer_id]
        self.hook_handle = layer.register_forward_hook(self._steering_hook)
        
        try:
            yield
        finally:
            # Remove hook after generation to return model to normal
            if self.hook_handle:
                self.hook_handle.remove()
