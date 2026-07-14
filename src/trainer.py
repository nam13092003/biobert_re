import os
import json
import shutil
import time
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm

from src.utils import get_gpu_memory_summary
from src.logger import setup_logger

class Trainer:
    def __init__(self, config: dict, model, train_dataset, dev_dataset):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        
        # Initialize Accelerator (single instance for the entire process)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            mixed_precision=config.get("mixed_precision", "fp16")
        )
        
        # Logger setup - only output log files from main process
        self.log_file = os.path.join(config.get("log_dir", "logs"), "train", "train.log")
        self.logger = setup_logger("Trainer", self.log_file)
        
        self.seed = config.get("seed", 42)
        self.epochs = config.get("epochs", 5)
        self.batch_size = config.get("batch_size", 8)
        self.lr = float(config.get("learning_rate", 2e-5))
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.save_steps = config.get("save_steps", 500)
        self.eval_steps = config.get("eval_steps", 500)
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.output_dir = config.get("output_dir", "outputs")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, resume_from_checkpoint: bool = False):
        # Create DataLoaders
        train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        dev_dataloader = DataLoader(
            self.dev_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # Setup Optimizer and Scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        num_training_steps = len(train_dataloader) * self.epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
        
        # Prepare for distributed training
        self.model, optimizer, train_dataloader, dev_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, dev_dataloader, lr_scheduler
        )
        
        start_epoch = 0
        global_step = 0
        
        # Resume from checkpoint
        if resume_from_checkpoint:
            latest_checkpoint_path = os.path.join(self.checkpoint_dir, "latest")
            if os.path.exists(latest_checkpoint_path):
                self.logger.info(f"Resuming training from checkpoint: {latest_checkpoint_path}")
                self.accelerator.load_state(latest_checkpoint_path)
                
                # Retrieve custom training state (epoch & step)
                state_file = os.path.join(latest_checkpoint_path, "trainer_state.json")
                if os.path.exists(state_file):
                    with open(state_file, "r") as f:
                        state_data = json.load(f)
                        start_epoch = state_data.get("epoch", 0)
                        global_step = state_data.get("step", 0)
                self.logger.info(f"Resumed at epoch {start_epoch}, step {global_step}")
            else:
                self.logger.warning("Resume requested but checkpoints/latest does not exist. Starting from scratch.")

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num Epochs = {self.epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.batch_size}")
        self.logger.info(f"  Total optimization steps = {num_training_steps}")
        
        self.model.train()
        
        for epoch in range(start_epoch, self.epochs):
            self.logger.info(f"--- Starting Epoch {epoch+1}/{self.epochs} ---")
            epoch_loss = 0
            
            # Wrap train_dataloader with tqdm on main process only
            progress_bar = tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch+1}", 
                disable=not self.accelerator.is_local_main_process,
                mininterval=2.0
            )
            
            for step, batch in enumerate(progress_bar):
                # Check if we should skip steps when resuming
                if resume_from_checkpoint and global_step > (epoch * len(train_dataloader) + step):
                    continue
                    
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Step update
                global_step += 1
                loss_val = loss.item()
                epoch_loss += loss_val
                
                progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
                
                # Log step information to file
                if self.accelerator.is_local_main_process and global_step % 50 == 0:
                    gpu_mem = get_gpu_memory_summary()
                    current_lr = lr_scheduler.get_last_lr()[0]
                    self.logger.info(
                        f"Epoch {epoch+1} | Step {global_step} | Loss: {loss_val:.4f} | LR: {current_lr:.2e} | {gpu_mem}"
                    )
                
                # Save checkpoints
                if global_step % self.save_steps == 0:
                    self._save_checkpoint(epoch, global_step)
                    
                # Evaluate periodically
                if global_step % self.eval_steps == 0:
                    self.evaluate(dev_dataloader, global_step)
                    self.model.train() # Make sure to switch back to training mode
            
            avg_loss = epoch_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
            
        # Final save
        self._save_checkpoint(self.epochs - 1, global_step)
        self.logger.info("Training completed successfully.")

    def _save_checkpoint(self, epoch: int, step: int):
        # save_state MUST be called on ALL processes – it handles
        # internal synchronisation and per-rank saving automatically.
        checkpoint_name = f"checkpoint-{step}"
        ckpt_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        latest_path = os.path.join(self.checkpoint_dir, "latest")

        self.accelerator.wait_for_everyone()

        self.logger.info(f"Saving checkpoint state to {ckpt_path}...")
        self.accelerator.save_state(ckpt_path)

        # Only the main process writes extra metadata / copies files
        if self.accelerator.is_local_main_process:
            state_data = {"epoch": epoch, "step": step}
            for path in [ckpt_path, latest_path]:
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "trainer_state.json"), "w") as f:
                    json.dump(state_data, f)
                # Copy train.yaml config file to the checkpoint folder
                shutil.copy(self.config["config_file_path"], os.path.join(path, "config.yaml"))
            
            # Recreate checkpoints/latest to point to the newest save
            self.logger.info("Updating latest checkpoint pointer...")
            for item in os.listdir(ckpt_path):
                s = os.path.join(ckpt_path, item)
                d = os.path.join(latest_path, item)
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)

    def evaluate(self, dev_dataloader, step: int):
        self.logger.info("***** Running validation evaluation *****")
        self.model.eval()
        
        total_eval_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dev_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                # Gather loss and predictions across processes using gather_for_metrics to handle padding correctly
                losses = self.accelerator.gather_for_metrics(loss.repeat(batch["input_ids"].size(0)))
                preds = self.accelerator.gather_for_metrics(logits.argmax(dim=-1))
                labels = self.accelerator.gather_for_metrics(batch["labels"])
                
                total_eval_loss += losses.sum().item()
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)
                
        avg_eval_loss = total_eval_loss / total_predictions
        accuracy = correct_predictions / total_predictions
        
        self.logger.info(
            f"Validation at Step {step} | Loss: {avg_eval_loss:.4f} | Accuracy: {accuracy:.2%}"
        )
        
        # Save metrics to json file
        if self.accelerator.is_local_main_process:
            metrics_file = os.path.join(self.config.get("log_dir", "logs"), "train", "metrics.json")
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            
            metrics = {}
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            metrics[str(step)] = {
                "val_loss": avg_eval_loss,
                "val_accuracy": accuracy,
                "timestamp": time.time()
            }
            
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
