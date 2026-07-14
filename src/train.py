import argparse
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import sys
import yaml
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# Add project root to sys.path to enable absolute imports of src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed
from src.dataset import N2C2Dataset
from src.trainer import Trainer


def download_model_on_main(model_name_or_path: str):
    """Pre-download model and tokenizer so the cache is populated.
    
    Called only on the main process before wait_for_everyone().
    """
    print("Main process: downloading model and tokenizer to cache...")
    AutoTokenizer.from_pretrained(model_name_or_path)
    AutoConfig.from_pretrained(model_name_or_path, num_labels=9)
    AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=9)
    print("Main process: download complete.")


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
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=2, 
        help="Number of GPUs/processes for training."
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
    
    # ---------------------------------------------------------------
    # Create the Trainer first – it owns the single Accelerator instance.
    # We pass model=None initially; model will be loaded after the
    # download guard below, then assigned to trainer.model.
    # ---------------------------------------------------------------
    trainer = Trainer(
        config=config,
        model=None,           # will be set after download guard
        train_dataset=None,   # will be set below
        dev_dataset=None      # will be set below
    )
    
    accelerator = trainer.accelerator
    is_main = accelerator.is_local_main_process
    model_name_or_path = config.get("model_name_or_path", "dmis-lab/biobert-base-cased-v1.2")
    
    # ---------------------------------------------------------------
    # Download guard: main process downloads first, then all processes
    # load from the now-populated cache.
    # ---------------------------------------------------------------
    if is_main:
        download_model_on_main(model_name_or_path)
    
    accelerator.wait_for_everyone()
    
    # All processes load from cache (local_files_only=True completely prevents network/filelock deadlocks)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    
    # Load Datasets
    max_seq_length = config.get("max_seq_length", 128)
    
    if is_main:
        print("Loading training dataset...")
    train_dataset = N2C2Dataset(
        file_path=config.get("train_file"),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_train=True
    )
    
    if is_main:
        print("Loading validation dataset...")
    dev_dataset = N2C2Dataset(
        file_path=config.get("dev_file"),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_train=False
    )
    
    # Load Model – all processes load from the populated cache
    if is_main:
        print("Initializing BioBERT model...")
    num_labels = 9  # 8 relation classes + 1 false class
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        local_files_only=True
    )
    
    # Assign the loaded objects to the trainer
    trainer.model = model
    trainer.train_dataset = train_dataset
    trainer.dev_dataset = dev_dataset
    
    # Check if we should automatically resume
    resume_flag = args.resume_from_checkpoint
    if not resume_flag:
        latest_ckpt = os.path.join(config.get("checkpoint_dir", "checkpoints"), "latest")
        if os.path.exists(latest_ckpt) and os.path.exists(os.path.join(latest_ckpt, "trainer_state.json")):
            resume_flag = True
            if is_main:
                print("Auto-detected existing checkpoint. Enabling resume.")
            
    # Start Training
    trainer.train(resume_from_checkpoint=resume_flag)

if __name__ == "__main__":
    main()
