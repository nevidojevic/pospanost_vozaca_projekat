import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# =====================================================
# MODEL (MORA BITI IDENTIČAN KAO U TRAININGU)
# =====================================================

class ResNetModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):
        return self.model(x)


# =====================================================
# LOAD MODEL
# =====================================================

model = ResNetModel()

model.load_state_dict(
    torch.load("drowsiness_resnet.pth", map_location="cpu")
)

model.eval()

# =====================================================
# TRANSFORM (MORA BITI ISTI KAO TRAINING)
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
# SLIKA
# =====================================================

image_path = "Drowsy_datset/train/DROWSY/images12.png"

image = Image.open(image_path).convert("RGB")
original_image = image.copy()

image = transform(image)
image = image.unsqueeze(0)

# =====================================================
# PREDIKCIJA
# =====================================================

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    predicted = torch.argmax(outputs, dim=1)

predicted_class = classes[predicted.item()]

drowsy_prob = probabilities[0][0].item()
natural_prob = probabilities[0][1].item()

# =====================================================
# ISPIS
# =====================================================

print("Logits:", outputs)
print("Probabilities:", probabilities)

print(f"\nPredikcija: {predicted_class}")
print(f"Drowsy: {drowsy_prob:.4f}")
print(f"Natural: {natural_prob:.4f}")

# =====================================================
# PRIKAZ
# =====================================================

plt.imshow(original_image)
plt.axis("off")
plt.title(
    f"{predicted_class}\n"
    f"Drowsy={drowsy_prob:.2%} | Natural={natural_prob:.2%}"
)
plt.show()