import os
import random
import shutil

source_drowsy = "dataset/Drowsy"
source_non_drowsy = "dataset/Non Drowsy"

target_drowsy = "dataset_n/Drowsy"
target_non_drowsy = "dataset_n/Non Drowsy"

os.makedirs(target_drowsy, exist_ok=True)
os.makedirs(target_non_drowsy, exist_ok=True)

# broj slika koje uzimas
num_images = 3000

# Drowsy
drowsy_images = random.sample(os.listdir(source_drowsy), num_images)

for image in drowsy_images:
    shutil.copy(
        os.path.join(source_drowsy, image),
        os.path.join(target_drowsy, image)
    )

# Non-drowsy
non_drowsy_images = random.sample(os.listdir(source_non_drowsy), num_images)

for image in non_drowsy_images:
    shutil.copy(
        os.path.join(source_non_drowsy, image),
        os.path.join(target_non_drowsy, image)
    )

print("Mali dataset napravljen!")