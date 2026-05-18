import evaluate
import numpy as np

# Load evaluation modules
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

def compute_booking_metrics(eval_preds, tokenizer):
    """
    Computes custom NLP metrics during validation steps.
    """
    predictions, labels = eval_preds
    
    # Handle cases where predictions are returned as tuples (logits, past_key_values)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    # Get the highest probability token ID at each sequence position
    pred_ids = np.argmax(predictions, axis=-1)
    
    # Replace cross-entropy padding token markers (-100) with the tokenizer's actual pad token ID
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    
    # Decode token sequences back into clean, readable text
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean whitespace strings for ROUGE evaluation formatting
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels] # BLEU requires nested lists
    
    # 1. Calculate standard lexical overlap scores
    rouge_results = rouge_metric.compute(
        predictions=decoded_preds, 
        references=[lbl[0] for lbl in decoded_labels], 
        use_stemmer=True
    )
    
    bleu_results = bleu_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels
    )
    
    # 2. Custom Rule-Based Domain Metric
    # Let's track how often the model accurately states 'refund' when the ground truth demands a refund.
    correct_claims = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        if ("refund" in label[0].lower()) == ("refund" in pred.lower()):
            correct_claims += 1
    claim_accuracy = correct_claims / len(decoded_preds)

    # 3. Consolidate dictionary to return to training logs
    return {
        "validation_rouge1": rouge_results["rouge1"],
        "validation_rougeL": rouge_results["rougeL"],
        "validation_bleu": bleu_results["bleu"],
        "validation_claim_intent_accuracy": claim_accuracy
    }