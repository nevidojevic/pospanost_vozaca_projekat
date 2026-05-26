# import os
# import random
# import shutil
#
# source_drowsy = "dataset/Drowsy"
# source_non_drowsy = "dataset/Non_Drowsy"
#
# target_drowsy = "dataset_n/Drowsy"
# target_non_drowsy = "dataset_n/Non_Drowsy"
#
# os.makedirs(target_drowsy, exist_ok=True)
# os.makedirs(target_non_drowsy, exist_ok=True)
#
# # broj slika koje uzimas
# num_images = 3000
#
# # Drowsy
# drowsy_images = random.sample(os.listdir(source_drowsy), num_images)
#
# for image in drowsy_images:
#     shutil.copy(
#         os.path.join(source_drowsy, image),
#         os.path.join(target_drowsy, image)
#     )
#
# # Non-drowsy
# non_drowsy_images = random.sample(os.listdir(source_non_drowsy), num_images)
#
# for image in non_drowsy_images:
#     shutil.copy(
#         os.path.join(source_non_drowsy, image),
#         os.path.join(target_non_drowsy, image)
#     )
#
# print("Mali dataset napravljen!")

# import os
# import shutil
#
# input_dir = "DriverDataset"
# output_dir = "driver_dataset"
#
# os.makedirs(output_dir, exist_ok=True)
#
# step = 10
#
# for class_folder in os.listdir(input_dir):
#     class_path = os.path.join(input_dir, class_folder)
#
#     if not os.path.isdir(class_path):
#         continue
#
#     images = sorted(os.listdir(class_path))
#
#     new_class_path = os.path.join(output_dir, class_folder)
#     os.makedirs(new_class_path, exist_ok=True)
#
#     for i, img_name in enumerate(images):
#         if i % step == 0:
#             src = os.path.join(class_path, img_name)
#             dst = os.path.join(new_class_path, img_name)
#             shutil.copy(src, dst)

import os
import shutil
import random

# ===== INPUT DATASET =====
input_dir = "DriverDataset"

drowsy_dir = os.path.join(input_dir, "Drowsy")
non_dir = os.path.join(input_dir, "Non_Drowsy")

# ===== IZVLAČENJE OSOBA =====
# osoba = prvo slovo imena fajla (A, B, C... / a, b, c...)

people = set()

for f in os.listdir(drowsy_dir):
    if f[0].isalpha():
        people.add(f[0].upper())

for f in os.listdir(non_dir):
    if f[0].isalpha():
        people.add(f[0].upper())

people = sorted(list(people))

# ===== SHUFFLE + SPLIT =====
random.seed(42)
random.shuffle(people)

split_idx = int(0.8 * len(people))

train_people = set(people[:split_idx])
test_people = set(people[split_idx:])

print("Train osobe:", train_people)
print("Test osobe:", test_people)

# ===== OUTPUT FOLDERS =====
output_dir = "dataset_split"

train_drowsy = os.path.join(output_dir, "train/Drowsy")
train_non = os.path.join(output_dir, "train/Non_Drowsy")

test_drowsy = os.path.join(output_dir, "test/Drowsy")
test_non = os.path.join(output_dir, "test/Non_Drowsy")

for path in [train_drowsy, train_non, test_drowsy, test_non]:
    os.makedirs(path, exist_ok=True)

# ===== FUNKCIJA ZA KOPIRANJE =====
def copy_images(src_dir, target_dir, people_set):
    for file in os.listdir(src_dir):
        if file[0].upper() in people_set:
            shutil.copy(
                os.path.join(src_dir, file),
                os.path.join(target_dir, file)
            )

# ===== TRAIN =====
copy_images(drowsy_dir, train_drowsy, train_people)
copy_images(non_dir, train_non, train_people)

# ===== TEST =====
copy_images(drowsy_dir, test_drowsy, test_people)
copy_images(non_dir, test_non, test_people)

print("Split gotov!")