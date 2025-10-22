import timm
import torch

# Initialize EfficientNet-B0 for binary classification (real vs fake)
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/efficientnet_b0_real_vs_fake.pth", weights_only=True))
model.eval()
