import os
import shutil
import random

# Source directories
src_dirs = {
    'O': '/workspaces/Commit-Issues/waste_data/DATASET/TRAIN/O',
    'R': '/workspaces/Commit-Issues/waste_data/DATASET/TRAIN/R'
}

# Destination directories
base_dst = '/workspaces/Commit-Issues/sample_data/TRAIN'

# Number of samples per class
num_samples = 50

os.makedirs(base_dst, exist_ok=True)

for label, src_dir in src_dirs.items():
    dst_dir = os.path.join(base_dst, label)
    os.makedirs(dst_dir, exist_ok=True)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
    sampled = random.sample(images, min(num_samples, len(images)))
    for img in sampled:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))

print('Sampled images copied to', base_dst)
