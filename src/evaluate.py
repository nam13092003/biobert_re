import argparse
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import sys
import json
import time
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from accelerate import Accelerator
from sklearn.metrics import classification_report

# Add project root to sys.path to enable absolute imports of src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import N2C2Dataset, INV_LABEL_MAP
from src.logger import setup_logger

def calculate_re_metrics(all_preds, all_labels):
    """Computes Precision, Recall, and F1-score for positive relation extraction classes (IDs 1-8)."""
    # Labeled classes: 1 to 8 (excluding 0 which is 'false')
    tp = 0
    fp = 0
    fn = 0
    
    for pred, label in zip(all_preds, all_labels):
        if label > 0: # Actual positive relation
            if pred == label:
                tp += 1
            else:
                fn += 1
        else: # Actual negative (false)
            if pred > 0:
                fp += 1
                
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BioBERT on n2c2 2010.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/eval.yaml", 
        help="Path to the evaluation configuration YAML file."
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=2, 
        help="Number of GPUs/processes for evaluation."
    )
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Setup Logger
    log_file = os.path.join(config.get("log_dir", "logs"), "evaluate", "eval.log")
    logger = setup_logger("Evaluation", log_file)
    
    # Initialize Accelerator for proper device management (GPU/Multi-GPU)
    accelerator = Accelerator()
    is_main = accelerator.is_local_main_process
    model_name_or_path = config.get("model_name_or_path", "dmis-lab/biobert-base-cased-v1.2")
    
    # ---------------------------------------------------------------
    # Download guard: main process downloads first, then all processes
    # load from the now-populated cache.
    # ---------------------------------------------------------------
    num_labels = 9
    if is_main:
        logger.info("Main process downloading model and tokenizer to cache...")
        AutoTokenizer.from_pretrained(model_name_or_path)
        AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        logger.info("Model and tokenizer download complete.")
        
    # Synchronize all processes before loading tokenizer and model
    accelerator.wait_for_everyone()
    
    # All processes load from the populated cache (local_files_only=True completely prevents network/filelock deadlocks)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    
    # Load Test Dataset
    max_seq_length = config.get("max_seq_length", 128)
    test_file = config.get("test_file")
    
    logger.info(f"Loading test dataset from {test_file}...")
    test_dataset = N2C2Dataset(
        file_path=test_file,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_train=False
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.get("batch_size", 16), 
        shuffle=False
    )
    
    # Load Model – all processes load from the populated cache
    logger.info("Initializing model architecture...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        local_files_only=True
    )
    
    # Prepare Dataloader and Model with accelerator
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    
    # Load saved state
    checkpoint_path = config.get("checkpoint_path", "checkpoints/latest")
    logger.info(f"Loading checkpoint weights from {checkpoint_path}...")
    accelerator.load_state(checkpoint_path)
    
    # Evaluation loop
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info("Running inference on test dataset...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in test_dataloader:
            outputs = model(**batch)
            logits = outputs.logits
            
            # Gather predictions and labels across distributed GPUs using gather_for_metrics to handle padding correctly
            preds = accelerator.gather_for_metrics(logits.argmax(dim=-1))
            labels = accelerator.gather_for_metrics(batch["labels"])
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    inference_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_re_metrics(all_preds, all_labels)
    
    # Generate detailed classification report
    target_names = [INV_LABEL_MAP[i] for i in range(num_labels)]
    cls_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=target_names, 
        digits=4,
        output_dict=True
    )
    
    logger.info("***** Evaluation Results *****")
    logger.info(f"  Test dataset size = {len(test_dataset)}")
    logger.info(f"  Inference completed in {inference_time:.2f} seconds")
    logger.info(f"  Micro-average F1 (excluding 'false'): {metrics['f1_score']:.2%}")
    logger.info(f"  Precision: {metrics['precision']:.2%} | Recall: {metrics['recall']:.2%}")
    logger.info(f"  TP: {metrics['true_positives']} | FP: {metrics['false_positives']} | FN: {metrics['false_negatives']}")
    
    # Save results to json file
    if accelerator.is_local_main_process:
        results_file = os.path.join(config.get("output_dir", "outputs"), "results.json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        results_data = {
            "checkpoint_used": checkpoint_path,
            "dataset_size": len(test_dataset),
            "inference_time_sec": inference_time,
            "metrics": metrics,
            "classification_report": cls_report,
            "timestamp": time.time()
        }
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=4)
        logger.info(f"Saved evaluation metrics to {results_file}")

if __name__ == "__main__":
    main()
