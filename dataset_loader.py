"""
LaSOT Dataset Loader for SeqTrack Assignment 3
- Loads LaSOT dataset from Hugging Face
- Filters to two selected classes
- Saves dataset summary information
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from datasets import load_dataset


def load_lasot_dataset(dataset_name: str = "l-lt/LaSOT", split: str = "train") -> Tuple[any, str]:
    """
    Load LaSOT dataset from Hugging Face, trying multiple possible dataset IDs
    
    Args:
        dataset_name: Name/ID of the dataset on Hugging Face
        split: Dataset split to load (train, validation, test)
        
    Returns:
        Tuple of (dataset, actual_name_used)
    """
    # Try different possible dataset names
    dataset_options = [dataset_name, "l-lt/LaSOT", "LaSOT", "lasot", "lasot_full"]
    
    for name in dataset_options:
        try:
            print(f"Attempting to load dataset: {name}")
            dataset = load_dataset(name, split=split)
            print(f"Successfully loaded dataset: {name}")
            return dataset, name
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
    
    raise RuntimeError(f"Could not load LaSOT dataset. Please check dataset name and internet connection.")


def filter_dataset_by_classes(dataset, class_names: List[str]) -> any:
    """
    Filter dataset to only include samples from specified classes
    
    Args:
        dataset: The dataset to filter
        class_names: List of class names to keep
        
    Returns:
        Filtered dataset
    """
    class_names_lower = [c.lower() for c in class_names]
    
    def filter_fn(example):
        # Handle different possible field names for class/category
        for field in ["category", "class", "label"]:
            if field in example:
                category = str(example[field]).lower()
                return category in class_names_lower
        
        # If no recognized field is found, try to extract from other fields
        if "metadata" in example and isinstance(example["metadata"], dict):
            for field in ["category", "class", "label"]:
                if field in example["metadata"]:
                    category = str(example["metadata"][field]).lower()
                    return category in class_names_lower
        
        return False
    
    filtered_dataset = dataset.filter(filter_fn)
    return filtered_dataset


def get_dataset_stats(dataset, class_names: List[str]) -> Dict[str, int]:
    """
    Get statistics about the dataset
    
    Args:
        dataset: Dataset to analyze
        class_names: List of class names to count
        
    Returns:
        Dictionary with counts per class and total
    """
    # Convert to DataFrame for easier analysis
    try:
        # Try different field names for category
        categories = []
        for example in dataset:
            category = None
            for field in ["category", "class", "label"]:
                if field in example:
                    category = example[field]
                    break
            
            if category is None and "metadata" in example and isinstance(example["metadata"], dict):
                for field in ["category", "class", "label"]:
                    if field in example["metadata"]:
                        category = example["metadata"][field]
                        break
            
            categories.append(category or "unknown")
        
        df = pd.DataFrame({"category": categories})
        
        # Count samples per class
        counts = df["category"].value_counts().to_dict()
        
        # Prepare stats dictionary
        stats = {class_name: counts.get(class_name, 0) for class_name in class_names}
        stats["total"] = len(dataset)
        
        return stats
    except Exception as e:
        print(f"Error computing dataset stats: {e}")
        # Fallback: just count total
        fallback = {class_name: 0 for class_name in class_names}
        fallback["total"] = len(dataset) if hasattr(dataset, "__len__") else 0
        return fallback


def save_dataset_summary(stats: Dict[str, int], class_names: List[str], output_path: str) -> None:
    """
    Save dataset summary to a text file
    
    Args:
        stats: Dictionary with dataset statistics
        class_names: List of class names
        output_path: Path to save the summary file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"Selected classes: {', '.join(class_names)}\n\n")
        
        for class_name in class_names:
            f.write(f"Class: {class_name}, Samples: {stats.get(class_name, 0)}\n")
        
        f.write(f"\nTotal dataset size: {stats.get('total', 0)} samples\n")
    
    print(f"Dataset summary saved to {output_path}")


def main():
    """Main function to load and process the LaSOT dataset"""
    parser = argparse.ArgumentParser(description="Load and filter LaSOT dataset for SeqTrack training")
    parser.add_argument("--classes", nargs="+", default=["airplane", "bicycle"], 
                        help="Class names to filter (default: airplane bicycle)")
    parser.add_argument("--dataset", type=str, default="l-lt/LaSOT", 
                        help="Dataset name on Hugging Face")
    parser.add_argument("--split", type=str, default="train", 
                        help="Dataset split to use")
    parser.add_argument("--output", type=str, default="dataset_summary.txt", 
                        help="Output path for dataset summary")
    
    args = parser.parse_args()
    
    # Ensure exactly two classes are selected
    if len(args.classes) != 2:
        print(f"Warning: Expected exactly 2 classes, got {len(args.classes)}. Using the first two.")
        args.classes = args.classes[:2]
    
    print(f"Loading LaSOT dataset with classes: {args.classes}")
    
    # Load dataset
    dataset, actual_name = load_lasot_dataset(args.dataset, args.split)
    print(f"Loaded dataset '{actual_name}' with {len(dataset)} samples")
    
    # Filter by classes
    filtered_dataset = filter_dataset_by_classes(dataset, args.classes)
    print(f"Filtered dataset contains {len(filtered_dataset)} samples")
    
    # Get stats
    stats = get_dataset_stats(filtered_dataset, args.classes)
    
    # Print required information
    print(f"Class names: {args.classes}")
    print(f"Number of samples per class: {[stats.get(cls, 0) for cls in args.classes]}")
    print(f"Total dataset size: {stats.get('total', 0)}")
    
    # Save summary
    save_dataset_summary(stats, args.classes, args.output)
    
    return filtered_dataset


if __name__ == "__main__":
    main()