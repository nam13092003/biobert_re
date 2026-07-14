import argparse
import os
import sys
import yaml
from transformers import AutoTokenizer

# Add project root to sys.path to enable absolute imports of src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed
from src.dataset import N2C2Dataset
from src.model import get_model
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train BioBERT on n2c2 2010 Relation Extraction.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train.yaml", 
        help="Path to the training configuration YAML file."
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        action="store_true", 
        help="Whether to resume training from the latest checkpoint."
    )
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Save the path inside config for trainer checkpoint saving
    config["config_file_path"] = args.config
    
    # Set random seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Initialize Accelerator to check process status
    from accelerate import Accelerator
    accelerator = Accelerator()
    is_main = accelerator.is_local_main_process
    
    # Download model config/weights first on main process
    from transformers import AutoModelForSequenceClassification
    model_name_or_path = config.get("model_name_or_path", "dmis-lab/biobert-base-cased-v1.2")
    
    if is_main:
        print("Main process downloading model and tokenizer...")
        AutoTokenizer.from_pretrained(model_name_or_path)
        AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        print("Model and tokenizer download complete.")
    
    # Synchronize all processes before loading tokenizer and model
    accelerator.wait_for_everyone()
    
    # Non-main processes read locally/offline from the populated cache
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=not is_main)
    
    # Load Datasets
    max_seq_length = config.get("max_seq_length", 128)
    
    print("Loading training dataset...")
    train_dataset = N2C2Dataset(
        file_path=config.get("train_file"),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_train=True
    )
    
    print("Loading validation dataset...")
    dev_dataset = N2C2Dataset(
        file_path=config.get("dev_file"),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_train=False
    )
    
    # Load Model (non-main processes read offline from the cache)
    print("Initializing BioBERT model...")
    num_labels = 9  # 8 relation classes + 1 false class
    model = get_model(model_name_or_path, num_labels=num_labels, local_files_only=not is_main)
    
    # Check if we should automatically resume
    resume_flag = args.resume_from_checkpoint
    if not resume_flag:
        latest_ckpt = os.path.join(config.get("checkpoint_dir", "checkpoints"), "latest")
        if os.path.exists(latest_ckpt) and os.path.exists(os.path.join(latest_ckpt, "trainer_state.json")):
            resume_flag = True
            print("Auto-detected existing checkpoint. Enabling resume.")
            
    # Initialize Trainer and Start Training
    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset
    )
    
    trainer.train(resume_from_checkpoint=resume_flag)

if __name__ == "__main__":
    main()
