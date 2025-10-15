"""
Dataset loader for LaSOT dataset with specific class selection (airplane + one random).

This implementation downloads class-level ZIP archives from the Hugging Face
dataset repository and extracts only the two needed classes locally, then
constructs a PyTorch Dataset that yields (template, search, bbox) samples.
"""
import os
import re
import random
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from huggingface_hub import list_repo_files, hf_hub_download

def load_lasot_dataset() -> Dict:
    """Prepare LaSOT samples from class ZIP archives (airplane + one random or pinned).

    If LASOT_ZIP_ROOT is set and contains local ZIPs, we use those exclusively and
    avoid any network calls (no need for HF credentials). Otherwise, we list and
    pull class ZIPs from the Hugging Face dataset repo.

    Returns:
        dict with keys: selected_classes, class_counts, total_samples, samples, root_dir
    """
    # Prefer local ZIPs if provided
    local_zip_root = os.environ.get('LASOT_ZIP_ROOT')
    classes: List[str]
    local_zip_map: Dict[str, str] = {}
    if local_zip_root and os.path.isdir(local_zip_root):
        print(f"Indexing LaSOT local archives in: {local_zip_root}")
        for name in os.listdir(local_zip_root):
            if name.lower().endswith('.zip'):
                cls = os.path.splitext(name)[0]
                local_zip_map[cls] = os.path.join(local_zip_root, name)
        classes = sorted(local_zip_map.keys())
        if not classes:
            raise RuntimeError(f"No .zip archives found under {local_zip_root}")
    else:
        print("Indexing LaSOT archives on Hugging Face...")
        repo_id = "l-lt/LaSOT"
        files = list(list_repo_files(repo_id, repo_type="dataset"))
        zip_files = [f for f in files if f.lower().endswith('.zip')]
        classes = sorted(os.path.splitext(os.path.basename(f))[0] for f in zip_files)
        if not classes:
            raise RuntimeError("No .zip archives found in l-lt/LaSOT dataset")

    # Allow user to pin classes via env: LASOT_CLASSES="coin,hat" (no spaces)
    pinned = os.environ.get('LASOT_CLASSES')
    if pinned:
        requested = [c.strip() for c in pinned.split(',') if c.strip()]
        selected_classes = [c for c in requested if c in classes][:2]
        if len(selected_classes) < 2 and not selected_classes:
            # fall through to automatic selection if invalid
            selected_classes = []
    else:
        selected_classes = []

    # If not pinned, pick airplane + one random (if available)
    if not selected_classes:
        if 'airplane' in classes:
            selected_classes.append('airplane')
        else:
            selected_classes.append(classes[0])
        remaining = [c for c in classes if c not in selected_classes]
        if remaining:
            selected_classes.append(random.choice(remaining))
    print(f"Selected classes: {selected_classes}")

    # Download and extract
    # Allow overriding locations via env
    extract_root_env = os.environ.get('LASOT_EXTRACT_ROOT')
    cache_root = os.path.abspath(extract_root_env or os.path.join('.', '.cache', 'lasot_zip'))
    os.makedirs(cache_root, exist_ok=True)

    extracted_roots: Dict[str, str] = {}
    for cls in selected_classes:
        # Use local ZIP if available, otherwise fetch from HF
        zip_path = local_zip_map.get(cls)
        if zip_path is None:
            repo_id = "l-lt/LaSOT"
            zip_path = hf_hub_download(repo_id=repo_id, filename=f"{cls}.zip", repo_type='dataset')
        target_dir = os.path.join(cache_root, 'train', cls)
        if not os.path.isdir(target_dir) or not os.listdir(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            print(f"Extracting {cls}.zip -> {target_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(target_dir)
        extracted_roots[cls] = target_dir

    def find_sequences(root: str) -> List[Tuple[str, str]]:
        seqs = []
        for dirpath, dirnames, filenames in os.walk(root):
            if 'img' in dirnames and 'groundtruth.txt' in filenames:
                seqs.append((os.path.join(dirpath, 'img'), os.path.join(dirpath, 'groundtruth.txt')))
        return seqs

    def nat_sort_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    samples: List[Dict] = []
    class_counts: Dict[str, int] = {cls: 0 for cls in selected_classes}

    # Limit number of sequences per class (to accelerate) via LASOT_SEQS_PER_CLASS
    max_seqs_env = os.environ.get('LASOT_SEQS_PER_CLASS')
    max_seqs = int(max_seqs_env) if max_seqs_env and max_seqs_env.isdigit() else None

    for cls in selected_classes:
        seqs = find_sequences(extracted_roots[cls])
        if max_seqs is not None:
            seqs = seqs[:max_seqs]
        for img_dir, gt_path in seqs:
            try:
                with open(gt_path, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                bboxes = []
                for l in lines:
                    parts = re.split(r'[\s,]+', l)
                    if len(parts) >= 4:
                        x, y, w, h = map(float, parts[:4])
                        bboxes.append((x, y, w, h))
                img_files = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)
                             if fn.lower().endswith(('.jpg', '.jpeg', '.png'))]
                img_files.sort(key=nat_sort_key)
                if len(img_files) < 2 or len(bboxes) < 2:
                    continue
                # Generate up to 3 pairs per sequence
                for idx in [1, len(img_files)//2, len(img_files)-1]:
                    if idx < len(img_files) and idx < len(bboxes):
                        samples.append({
                            'template_path': img_files[0],
                            'search_path': img_files[idx],
                            'bbox': bboxes[idx],
                            'class_name': cls
                        })
                        class_counts[cls] += 1
            except Exception:
                continue

    dataset_info = {
        'selected_classes': selected_classes,
        'class_counts': class_counts,
        'total_samples': len(samples),
        'samples': samples,
        'root_dir': cache_root,
    }

    print(f"Prepared {dataset_info['total_samples']} samples")
    for cls, n in class_counts.items():
        print(f"  {cls}: {n} samples")
    return dataset_info

class LaSOTTrackingDataset(Dataset):
    """PyTorch Dataset yielding real (template, search, bbox) from extracted LaSOT."""

    def __init__(self, dataset_info, template_size: int = 256, search_size: int = 256):
        self.samples = dataset_info['samples']
        self.template_size = template_size
        self.search_size = search_size

        import torchvision.transforms as transforms
        self.template_tf = transforms.Compose([
            transforms.Resize((template_size, template_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.search_tf = transforms.Compose([
            transforms.Resize((search_size, search_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        template_img = Image.open(s['template_path']).convert('RGB')
        search_img = Image.open(s['search_path']).convert('RGB')

        template = self.template_tf(template_img)
        search = self.search_tf(search_img)
        bbox = torch.tensor(s['bbox'], dtype=torch.float32)

        return {
            'template_images': template,
            'search_images': search,
            'bbox': bbox,
            'class_name': s.get('class_name')
        }
    
    # transforms defined in __init__

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
