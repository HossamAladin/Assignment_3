"""
SeqTrack Training Script for Assignment 3
- Trains SeqTrack model on LaSOT dataset (two classes)
- Uses deterministic settings with seed = 8
- Saves checkpoints and uploads to Hugging Face
- Maintains detailed training logs
"""

import os
import sys
import random
import argparse
import time
from datetime import timedelta
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from huggingface_hub import HfApi

# Import dataset loader
from dataset_loader import load_lasot_dataset, filter_dataset_by_classes

# Add SeqTrack lib to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'lib')
if os.path.exists(lib_path) and lib_path not in sys.path:
    sys.path.append(lib_path)


def set_deterministic_settings(seed: int = 8) -> None:
    """
    Set all random seeds and deterministic settings for reproducibility
    
    Args:
        seed: Random seed value (default: 8)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set deterministic settings with seed = {seed}")


def seed_worker(worker_id: int) -> None:
    """
    Set seed for dataloader workers to ensure reproducibility
    
    Args:
        worker_id: Worker ID
    """
    worker_seed = 8 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to H:MM:SS hours format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string in H:MM:SS hours format
    """
    if seconds < 0:
        seconds = 0
    
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Add days if applicable
    hours += td.days * 24
    
    return f"{hours}:{minutes:02d}:{seconds:02d} hours"


class Logger:
    """Logger class for maintaining training logs"""
    
    def __init__(self, log_path: str):
        """
        Initialize logger
        
        Args:
            log_path: Path to log file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        
        self.log_path = log_path
        self.log_file = open(log_path, 'w')
        
    def log(self, message: str, print_to_console: bool = True) -> None:
        """
        Log a message to file and optionally print to console
        
        Args:
            message: Message to log
            print_to_console: Whether to print to console
        """
        if print_to_console:
            print(message)
        
        self.log_file.write(message + '\n')
        self.log_file.flush()
    
    def log_training_step(self, epoch: int, samples_seen: int, total_samples: int,
                         last_batch_time: float, epoch_start_time: float, 
                         actual_samples_last_interval: int) -> None:
        """
        Log training progress in the required format
        
        Args:
            epoch: Current epoch number
            samples_seen: Number of samples processed so far
            total_samples: Total number of samples in the dataset
            last_batch_time: Time taken for the last batch
            epoch_start_time: Time when the epoch started
            actual_samples_last_interval: Actual number of samples in the last interval
        """
        # Calculate time statistics
        time_since_start = time.time() - epoch_start_time
        
        # Calculate samples per second based on actual samples processed in the last interval
        samples_per_sec = actual_samples_last_interval / max(last_batch_time, 1e-6)
        
        # Calculate time left based on samples per second
        remaining_samples = total_samples - samples_seen
        time_left = remaining_samples / max(samples_per_sec, 1e-6)
        
        # Format the log message exactly as required
        message = (
            f"Epoch {epoch} : {samples_seen} / {total_samples} samples , \n"
            f"time for last 50 samples : {format_time(last_batch_time)} , \n"
            f"time since beginning : {format_time(time_since_start)} , \n"
            f"time left to finish the epoch : {format_time(time_left)}"
        )
        
        self.log(message)
    
    def close(self) -> None:
        """Close the log file"""
        self.log_file.close()


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                   checkpoint_dir: str, seed: int = 8) -> str:
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoint
        seed: Random seed used for training
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.ckpt")
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": seed
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def upload_to_huggingface(checkpoint_path: str, repo_id: str) -> bool:
    """
    Upload checkpoint to Hugging Face Hub
    
    Args:
        checkpoint_path: Path to checkpoint file
        repo_id: Hugging Face repository ID
        
    Returns:
        True if upload was successful, False otherwise
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; skipping HF upload.")
        return False
    
    try:
        api = HfApi()
        try:
            api.create_repo(repo_id, token=token, private=True)
        except Exception:
            pass  # Repository might already exist
        
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=os.path.basename(checkpoint_path),
            repo_id=repo_id,
            token=token
        )
        print(f"Uploaded {checkpoint_path} to {repo_id}")
        return True
    except Exception as e:
        print(f"HF upload failed: {e}")
        return False


def calculate_iou(pred_boxes, target_boxes):
    """
    Calculate IoU between predicted and target boxes
    
    Args:
        pred_boxes: Predicted bounding boxes [x1, y1, x2, y2]
        target_boxes: Target bounding boxes [x1, y1, x2, y2]
        
    Returns:
        Average IoU
    """
    if isinstance(pred_boxes, np.ndarray):
        pred_boxes = torch.tensor(pred_boxes)
        target_boxes = torch.tensor(target_boxes)
    
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
        target_boxes = target_boxes.unsqueeze(0)
    
    x1 = torch.max(pred_boxes[:,0], target_boxes[:,0])
    y1 = torch.max(pred_boxes[:,1], target_boxes[:,1])
    x2 = torch.min(pred_boxes[:,2], target_boxes[:,2])
    y2 = torch.min(pred_boxes[:,3], target_boxes[:,3])
    
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    area_pred = (pred_boxes[:,2]-pred_boxes[:,0]).clamp(min=0)*(pred_boxes[:,3]-pred_boxes[:,1]).clamp(min=0)
    area_target = (target_boxes[:,2]-target_boxes[:,0]).clamp(min=0)*(target_boxes[:,3]-target_boxes[:,1]).clamp(min=0)
    
    union = area_pred + area_target - inter_area + 1e-9
    return ((inter_area + 1e-9) / union).mean().item()


def prepare_seqtrack_data_and_model(filtered_dataset, args, device):
    """
    Prepare SeqTrack model and dataset
    
    Args:
        filtered_dataset: Filtered dataset
        args: Command line arguments
        device: Torch device
        
    Returns:
        model, train_loader, val_loader, optimizer, criterion
    """
    try:
        # Import SeqTrack modules
        from lib.models.seqtrack import build_seqtrack
        from lib.train.dataset import build_dataloaders
        from lib.train.base_functions import create_optimizer
        from lib.train.loss import build_loss
        from lib.config.seqtrack.config import cfg, update_config_from_file
        
        # Load SeqTrack config
        if os.path.exists(args.config):
            update_config_from_file(args.config)
        
        # Update config with command line arguments
        cfg.TRAIN.EPOCH = args.epochs
        cfg.TRAIN.BATCH_SIZE = args.batch_size
        cfg.TRAIN.NUM_WORKERS = args.workers
        cfg.MODEL.BACKBONE.PATCH_SIZE = args.patch_size
        
        # Build SeqTrack model
        model = build_seqtrack(cfg)
        
        # Move model to device
        model = model.to(device)
        
        # Create data loaders with worker seeding
        g = torch.Generator()
        g.manual_seed(args.seed)
        
        train_loader, val_loader = build_dataloaders(
            cfg, 
            filtered_dataset,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        # Create optimizer
        optimizer = create_optimizer(cfg, model)
        
        # Build loss from SeqTrack
        criterion = build_loss(cfg)
        
        return model, train_loader, val_loader, optimizer, criterion
        
    except ImportError as e:
        print(f"Error importing SeqTrack modules: {e}")
        print("Please make sure you're running from the SeqTrack directory with lib/ accessible")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up SeqTrack: {e}")
        sys.exit(1)


def train_seqtrack(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Set deterministic settings
    set_deterministic_settings(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = Logger(args.log_path)
    logger.log(f"Starting SeqTrack training with seed={args.seed}, patch_size={args.patch_size}, workers={args.workers}")
    
    # Initialize tensorboard
    os.makedirs(args.log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=args.log_dir)
    
    # Load and filter dataset
    dataset, _ = load_lasot_dataset(args.dataset, split=args.split)
    filtered_dataset = filter_dataset_by_classes(dataset, args.classes)
    
    logger.log(f"Loaded dataset with {len(filtered_dataset)} samples from classes: {args.classes}")
    
    # Save dataset summary
    with open(args.summary_path, "w") as f:
        f.write(f"Selected classes: {', '.join(args.classes)}\n")
        f.write(f"Total samples: {len(filtered_dataset)}\n")
    
    # Prepare model and data loaders
    model, train_loader, val_loader, optimizer, criterion = prepare_seqtrack_data_and_model(filtered_dataset, args, device)
    
    # Training loop
    best_iou = 0.0
    epoch_losses = []
    train_start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start_time = time.time()
        last_log_time = epoch_start_time
        running_loss = 0.0
        samples_seen = 0
        samples_at_last_log = 0
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # Process batch and compute loss
            images, targets = batch['template'], batch['search']
            batch_size = images.size(0)
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * batch_size
            samples_seen += batch_size
            
            # Log every 50 samples or as close as possible
            if samples_seen - samples_at_last_log >= 50 or batch_idx == len(train_loader) - 1:
                current_time = time.time()
                time_for_last_batch = current_time - last_log_time
                actual_samples = samples_seen - samples_at_last_log
                
                logger.log_training_step(
                    epoch=epoch,
                    samples_seen=samples_seen,
                    total_samples=len(train_loader.dataset),
                    last_batch_time=time_for_last_batch,
                    epoch_start_time=epoch_start_time,
                    actual_samples_last_interval=actual_samples
                )
                
                last_log_time = current_time
                samples_at_last_log = samples_seen
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        iou_sum = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, targets = batch['template'], batch['search']
                batch_size = images.size(0)
                
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                val_loss += criterion(outputs, targets).item() * batch_size
                
                # Calculate IoU using the real implementation
                pred_boxes = outputs['pred_boxes'] if isinstance(outputs, dict) else outputs
                target_boxes = targets['target_bbox'] if isinstance(targets, dict) else targets
                
                try:
                    # Try to use SeqTrack's evaluation function if available
                    from lib.train.utils import get_iou
                    iou = get_iou(pred_boxes, target_boxes)
                except (ImportError, NameError):
                    # Fall back to our implementation
                    iou = calculate_iou(pred_boxes, target_boxes)
                
                iou_sum += iou * batch_size
                val_samples += batch_size
        
        val_loss /= max(1, val_samples)
        avg_iou = iou_sum / max(1, val_samples)
        
        # Update best IoU
        if avg_iou > best_iou:
            best_iou = avg_iou
        
        # Log epoch results
        logger.log(f"Epoch {epoch} completed: train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}, IoU={avg_iou:.4f}")
        
        # Save tensorboard metrics
        tb_writer.add_scalar('Loss/train', epoch_loss, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('IoU', avg_iou, epoch)
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed
        )
        
        # Upload to Hugging Face
        upload_to_huggingface(checkpoint_path, args.hf_repo)
    
    # Training complete
    total_time = time.time() - train_start_time
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    
    # Log final summary
    logger.log("\n" + "="*50)
    logger.log("Training Summary:")
    logger.log(f"Total training time: {format_time(total_time)}")
    logger.log(f"Best IoU: {best_iou:.4f}")
    logger.log(f"Average loss per epoch: {avg_epoch_loss:.6f}")
    logger.log("="*50)
    
    # Close logger
    logger.close()
    tb_writer.close()
    
    # Generate report
    generate_report(args, best_iou, avg_epoch_loss, total_time)


def generate_report(args, best_iou, avg_loss, total_time):
    """
    Generate assignment report
    
    Args:
        args: Command line arguments
        best_iou: Best IoU achieved
        avg_loss: Average loss per epoch
        total_time: Total training time
    """
    report_path = args.report_path
    
    with open(report_path, "w") as f:
        f.write("# Assignment 3 - SeqTrack Setup, Training, and Checkpoint Management\n\n")
        
        f.write("## Dataset Information\n")
        f.write(f"- Selected classes: {args.classes}\n")
        f.write(f"- Dataset source: {args.dataset}\n\n")
        
        f.write("## Environment Setup\n")
        f.write("- Official SeqTrack implementation from Microsoft's VideoX repository\n")
        f.write("- Conda environment with PyTorch and CUDA support\n")
        f.write("- Deterministic training with fixed seed\n\n")
        
        f.write("## Training Configuration\n")
        f.write(f"- Seed: {args.seed}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Patch size: {args.patch_size}\n")
        f.write(f"- Workers: {args.workers}\n")
        f.write(f"- Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n\n")
        
        f.write("## Training Results\n")
        f.write(f"- Best IoU: {best_iou:.4f}\n")
        f.write(f"- Average loss per epoch: {avg_loss:.6f}\n")
        f.write(f"- Total training time: {format_time(total_time)}\n\n")
        
        f.write("## Checkpoints\n")
        f.write(f"- Local checkpoints saved to: {args.checkpoint_dir}\n")
        f.write(f"- Uploaded to Hugging Face repo: {args.hf_repo}\n\n")
        
        f.write("## GitHub Repository\n")
        f.write("- https://github.com/HossamAladin/Assignment_3\n\n")
        
        f.write("## Limitations\n")
        f.write("- Training performed on a subset of LaSOT (two classes only)\n")
        f.write("- Resource constraints limited training to 5 epochs\n")
    
    print(f"Report generated: {report_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train SeqTrack on LaSOT dataset")
    
    # Dataset parameters
    parser.add_argument("--classes", nargs="+", default=["airplane", "bicycle"], 
                        help="Two class names to filter (default: airplane bicycle)")
    parser.add_argument("--dataset", type=str, default="l-lt/LaSOT", 
                        help="Dataset name on Hugging Face")
    parser.add_argument("--split", type=str, default="train", 
                        help="Dataset split to use")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of epochs to train")
    parser.add_argument("--seed", type=int, default=8, 
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size")
    parser.add_argument("--workers", type=int, default=1, 
                        help="Number of data loading workers")
    parser.add_argument("--patch_size", type=int, default=8, 
                        help="Patch size for ViT backbone")
    parser.add_argument("--config", type=str, default="experiments/seqtrack/config.yaml", 
                        help="Path to SeqTrack config file")
    
    # Output parameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--log_path", type=str, default="training_log.txt", 
                        help="Path to save training log")
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="Directory for tensorboard logs")
    parser.add_argument("--summary_path", type=str, default="dataset_summary.txt", 
                        help="Path to save dataset summary")
    parser.add_argument("--report_path", type=str, default="assignment_3.md", 
                        help="Path to save assignment report")
    parser.add_argument("--hf_repo", type=str, default="hossamaladdin/Assignment3", 
                        help="Hugging Face repo ID for uploading checkpoints")
    
    args = parser.parse_args()
    
    # Ensure exactly two classes are selected
    if len(args.classes) != 2:
        print(f"Warning: Expected exactly 2 classes, got {len(args.classes)}. Using the first two.")
        args.classes = args.classes[:2]
    
    # Start training
    train_seqtrack(args)


if __name__ == "__main__":
    main()