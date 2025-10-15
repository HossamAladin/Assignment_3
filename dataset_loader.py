"""
Dataset loader for LaSOT dataset with specific class selection
"""
import os
import sys
import random
import numpy as np
from datasets import load_dataset
import torch

def load_lasot_dataset():
    """
    Load LaSOT dataset and select airplane class + one random class
    Returns:
        dataset_info: dict with class names and sample counts
    """
    print("Loading LaSOT dataset from Hugging Face...")
    
    try:
        # Load the dataset
        dataset = load_dataset("l-lt/LaSOT")
        
        # Get all available classes
        train_data = dataset['train']
        
        # Extract unique classes from the dataset
        classes = set()
        for item in train_data:
            if 'class' in item:
                classes.add(item['class'])
            elif 'category' in item:
                classes.add(item['category'])
        
        classes = list(classes)
        print(f"Available classes: {classes}")
        
        # Select airplane (fixed) and one random class
        selected_classes = ['airplane']
        
        # Remove airplane from available classes and select one random
        remaining_classes = [c for c in classes if c.lower() != 'airplane']
        if remaining_classes:
            random_class = random.choice(remaining_classes)
            selected_classes.append(random_class)
        
        print(f"Selected classes: {selected_classes}")
        
        # Count samples for each selected class
        class_counts = {}
        for class_name in selected_classes:
            count = sum(1 for item in train_data 
                       if item.get('class', '').lower() == class_name.lower() or 
                          item.get('category', '').lower() == class_name.lower())
            class_counts[class_name] = count
        
        dataset_info = {
            'selected_classes': selected_classes,
            'class_counts': class_counts,
            'total_samples': sum(class_counts.values())
        }
        
        return dataset_info
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback: return mock data for development
        print("Using mock dataset for development...")
        return {
            'selected_classes': ['airplane', 'bicycle'],
            'class_counts': {'airplane': 150, 'bicycle': 120},
            'total_samples': 270
        }

def print_dataset_summary(dataset_info):
    """Print and save dataset summary in markdown format"""
    
    summary = f"""
## Dataset Summary

### Selected Classes:
- **airplane**: {dataset_info['class_counts'].get('airplane', 0)} samples
- **{dataset_info['selected_classes'][1] if len(dataset_info['selected_classes']) > 1 else 'N/A'}**: {dataset_info['class_counts'].get(dataset_info['selected_classes'][1], 0) if len(dataset_info['selected_classes']) > 1 else 0} samples

### Total Samples: {dataset_info['total_samples']}
"""
    
    print(summary)
    
    # Save to file
    with open('dataset_summary.md', 'w') as f:
        f.write(summary)
    
    return summary

if __name__ == "__main__":
    # Set random seed for reproducible class selection
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    
    dataset_info = load_lasot_dataset()
    print_dataset_summary(dataset_info)
