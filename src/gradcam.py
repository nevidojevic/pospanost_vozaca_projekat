import cv2
import numpy as np
import torch

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def run_gradcam(model, image_path, device):

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    pil_img = Image.open(image_path)

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    rgb_img = np.array(
        pil_img.resize((96, 96)).convert("RGB")
    ).astype(np.float32) / 255.0

    # poslednji konvolucioni blok ResNet18
    target_layer = model.model.layer4[-1]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    cv2.imshow(
        "GradCAM",
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()