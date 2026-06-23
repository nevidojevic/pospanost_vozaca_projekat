import torch
import torch.nn as nn

from torchvision import transforms
import torchvision.models as models

from PIL import Image
import matplotlib.pyplot as plt

from ResNet import full_dataset

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# RESNET MODEL
# =====================================================

class ResNetModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):
        return self.model(x)

# =====================================================
# UCITAVANJE MODELA
# =====================================================

model = ResNetModel()

model.load_state_dict(
    torch.load(
        "src/resnet_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# =====================================================
# TRANSFORMACIJE
# MORAJU BITI ISTE KAO TEST TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# =====================================================
# KLASE
# =====================================================

classes = ["DROWSY", "NATURAL"]

# =====================================================
# PUTANJA DO SLIKE
# =====================================================

image_path = "test_images/Screenshot 2026-06-21 154250.png"

# =====================================================
# UCITAVANJE SLIKE
# =====================================================

image = Image.open(image_path).convert("RGB")

original_image = image.copy()

image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

# =====================================================
# PREDIKCIJA
# =====================================================

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    _, predicted = torch.max(outputs, 1)

predicted_class = classes[predicted.item()]

drowsy_prob = probabilities[0][0].item()
natural_prob = probabilities[0][1].item()

# =====================================================
# ISPIS
# =====================================================

print("Raw outputs:")
print(outputs)

print("\nProbabilities:")
print(probabilities)

print(f"\nPredikcija: {predicted_class}")
print(f"DROWSY: {drowsy_prob:.4f}")
print(f"NATURAL: {natural_prob:.4f}")

# =====================================================
# PRIKAZ
# =====================================================

plt.imshow(original_image)
plt.axis("off")

plt.title(
    f"{predicted_class}\n"
    f"Drowsy={drowsy_prob:.2%} | "
    f"Natural={natural_prob:.2%}"
)

plt.show()