import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

# =====================================================
# CNN MODEL
# =====================================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(16 * 9 * 9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x


# =====================================================
# UCITAVANJE MODELA
# =====================================================

model = CNN()

model.load_state_dict(
    torch.load("drowsiness_model.pth", map_location=torch.device("cpu"))
)

model.eval()

# =====================================================
# TRANSFORMACIJE
# =====================================================

# transform = transforms.Compose([
#     transforms.Resize((48, 48)),
#     transforms.ToTensor(),
#     # transforms.Normalize(
#     #     (0.5, 0.5, 0.5),
#     #     (0.5, 0.5, 0.5)
#     # )
#     transforms.Grayscale(num_output_channels=3)
# ])
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# =====================================================
# KLASE
# =====================================================

classes = ["DROWSY", "NATURAL"]

# =====================================================
# PUTANJA DO NOVE SLIKE
# =====================================================

image_path = "Drowsy_datset/train/NATURAL/images11.png"

# =====================================================
# UCITAVANJE SLIKE
# =====================================================

image = Image.open(image_path).convert("RGB")

# Sačuvaj original za prikaz
original_image = image.copy()

# Transformacija
image = transform(image)

# Dodavanje batch dimenzije
image = image.unsqueeze(0)

# =====================================================
# PREDIKCIJA
# =====================================================

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)
    print(outputs)
    print(probabilities)
    _, predicted = torch.max(outputs, 1)

predicted_class = classes[predicted.item()]

drowsy_prob = probabilities[0][0].item()
non_drowsy_prob = probabilities[0][1].item()

# =====================================================
# ISPIS REZULTATA
# =====================================================

print(f"Predikcija: {predicted_class}")
print(f"Drowsy verovatnoca: {drowsy_prob:.4f}")
print(f"Natural verovatnoca: {non_drowsy_prob:.4f}")

# =====================================================
# PRIKAZ SLIKE
# =====================================================

plt.imshow(original_image)
plt.axis("off")
plt.title(
    f"{predicted_class}\n"
    f"Drowsy={drowsy_prob:.2%} | "
    f"Natural={non_drowsy_prob:.2%}"
)
plt.show()