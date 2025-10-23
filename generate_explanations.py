import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import os
import timm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)  # Binary classification
    model.load_state_dict(torch.load("models/efficientnet_b0_real_vs_fake.pth", weights_only=True))
    model = model.to(device)
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Register hooks for EfficientNet
        target_layer = self.model.blocks[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class

def apply_gradcam(image_path, model, output_path):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    cam, pred_class = gradcam.generate_cam(img_tensor)
    
    # Create heatmap overlay
    img_resized = cv2.resize(np.array(img), (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    overlay = heatmap * 0.4 + img_resized * 0.6
    overlay = np.uint8(overlay)
    
    # Save result
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title(f'Grad-CAM: {"Real" if pred_class == 0 else "Fake"}')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    print(f'Grad-CAM saved to {output_path}')
    return pred_class

# LIME implementation
def predict_fn(images, model):
    batch = torch.stack([transform(Image.fromarray(img.astype('uint8'))) for img in images])
    batch = batch.to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    return probs.cpu().numpy()

def apply_lime(image_path, model, output_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    classifier_fn = lambda images: predict_fn(images, model)
    
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img_array,
        classifier_fn,
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )
    
    # Get the top predicted class
    pred_class = explanation.top_labels[0]
    
    # Get image and mask
    temp, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    plt.title(f'LIME Explanation: {"Real" if pred_class == 0 else "Fake"}')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    print(f'LIME explanation saved to {output_path}')
    return pred_class

if __name__ == '__main__':
    # Create explanations directory if it doesn't exist
    os.makedirs('explanations', exist_ok=True)
    
    # Load model
    print('Loading model...')
    model = load_model()
    
    # Process sample_real.jpg
    print('\nProcessing sample_real.jpg...')
    if os.path.exists('sample_real.jpg'):
        print('Generating Grad-CAM for real image...')
        apply_gradcam('sample_real.jpg', model, 'explanations/gradcam_real.png')
        
        print('Generating LIME explanation for real image...')
        apply_lime('sample_real.jpg', model, 'explanations/lime_real.png')
    else:
        print('Warning: sample_real.jpg not found')
    
    # Process sample_fake.jpg
    print('\nProcessing sample_fake.jpg...')
    if os.path.exists('sample_fake.jpg'):
        print('Generating Grad-CAM for fake image...')
        apply_gradcam('sample_fake.jpg', model, 'explanations/gradcam_fake.png')
        
        print('Generating LIME explanation for fake image...')
        apply_lime('sample_fake.jpg', model, 'explanations/lime_fake.png')
    else:
        print('Warning: sample_fake.jpg not found')
    
    print('\nDone! All explanations generated successfully.')
