import torch
from transformers import Trainer

class BookingCustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss calculation hook.
        """
        # 1. Forward pass to get standard outputs
        outputs = model(**inputs)
        
        # 2. Extract logits and target labels
        logits = outputs.get("logits")  # Shape: [batch_size, sequence_length, vocab_size]
        labels = inputs.get("labels")  # Shape: [batch_size, sequence_length]
        
        # 3. Define your custom loss logic
        # Shift tokens for Causal Language Modeling (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Standard Cross-Entropy base
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # --- Custom Logic Example ---
        # Reshape losses back to [batch_size, sequence_length - 1]
        token_losses = token_losses.view(shift_labels.size())
        
        # Let's say token ID 10432 represents standard financial terms (e.g., "$", "refund").
        # We find where those labels are and triple their penalty weight.
        critical_token_mask = (shift_labels == 10432).float()
        custom_weights = 1.0 + (critical_token_mask * 2.0) 
        
        # Apply the weights and take the mean
        weighted_loss = (token_losses * custom_weights).sum() / (shift_labels != -100).sum()
        # ----------------------------

        return (weighted_loss, outputs) if return_outputs else weighted_loss