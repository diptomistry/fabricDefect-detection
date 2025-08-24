#!/usr/bin/env python3
"""
Single Image Fabric Defect Tester
=================================

Test individual fabric images for defects using the trained PatchCore model.

Usage:
    python test_single_image.py <image_path>
    python test_single_image.py defect/IMG_0109.jpg
    python test_single_image.py noDefect/IMG_0125.jpg
    
Interactive mode:
    python test_single_image.py
"""

import os
import sys
import time
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Add the patchcore source to path
sys.path.append('patchcore-inspection/src')

import patchcore.patchcore as patchcore_module
import patchcore.backbones as backbones
import patchcore.sampler as sampler
import patchcore.common as common

def load_image_tensor(image_path, transform):
    """Load an image and convert to tensor without batch dimension."""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    return tensor  # Don't add batch dimension here

def initialize_model():
    """Initialize and train the PatchCore model."""
    print("🔄 Initializing PatchCore model...")
    
    # Paths and transform
    train_dir = os.path.join('.', 'noDefect')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Get training images
    train_image_paths = sorted(glob(os.path.join(train_dir, '*.jpg')))
    
    # Setup device and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = backbones.load("resnet50")
    backbone.eval()
    
    patchcore_model = patchcore_module.PatchCore(device=device)
    patchcore_model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2"],
        device=device,
        input_shape=(3, 128, 128),
        pretrain_embed_dimension=1024,
        target_embed_dimension=256,
        patchsize=3,
        patchstride=3,
        anomaly_score_num_nn=1,
        featuresampler=sampler.IdentitySampler(),
        nn_method=common.FaissNN(False, 1),
    )
    
    # Quick training with first 50 images for faster initialization
    print(f"🏃 Quick training on {min(50, len(train_image_paths))} normal images...")
    train_tensors = []
    for path in train_image_paths[:50]:
        tensor = load_image_tensor(path, transform).unsqueeze(0)
        train_tensors.append(tensor)
    
    start_time = time.time()
    patchcore_model.fit(train_tensors)
    train_time = time.time() - start_time
    
    print(f"✅ Model ready! Training completed in {train_time:.2f} seconds")
    return patchcore_model, transform

def test_single_image(model, transform, image_path, anomaly_threshold=0.3):
    """Test a single image for defects."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return None
        
    try:
        print(f"\n🔍 Testing: {os.path.basename(image_path)}")
        
        # Load and predict
        tensor = load_image_tensor(image_path, transform)
        batch_tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        start_time = time.time()
        prediction_result = model.predict(batch_tensor)
        pred_time = time.time() - start_time
        
        if isinstance(prediction_result, tuple):
            scores = prediction_result[0]
        else:
            scores = prediction_result
            
        score = scores[0] if len(scores) > 0 else 0.0
        
        # Classification
        is_anomaly = score > anomaly_threshold
        if score >= 1.0:
            confidence = "HIGH"
        elif score >= 0.7:
            confidence = "MEDIUM" 
        else:
            confidence = "LOW"
            
        status = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"
        
        print(f"📊 Results:")
        print(f"   Score: {score:.4f}")
        print(f"   Status: {status}")
        print(f"   Confidence: {confidence}")
        print(f"   Time: {pred_time:.3f}s")
        
        return {
            'score': score,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'prediction_time': pred_time
        }
        
    except Exception as e:
        print(f"❌ Error testing image: {e}")
        return None

def interactive_mode(model, transform):
    """Interactive mode for testing multiple images."""
    print("\n" + "="*60)
    print("🧪 INTERACTIVE FABRIC DEFECT DETECTION")
    print("="*60)
    print("Commands:")
    print("  • Enter image path (e.g., defect/IMG_0109.jpg)")
    print("  • 'list' - show available images")
    print("  • 'quit' or 'exit' - exit interactive mode")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n📁 Enter image path (or command): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            elif user_input.lower() == 'list':
                print("\n📂 Available images:")
                print("\n🟢 Normal images (noDefect/):")
                normal_images = glob("noDefect/*.jpg")[:10]  # Show first 10
                for img in normal_images:
                    print(f"   {img}")
                if len(glob("noDefect/*.jpg")) > 10:
                    print(f"   ... and {len(glob('noDefect/*.jpg')) - 10} more")
                    
                print("\n🔴 Defect images (defect/):")
                defect_images = glob("defect/*.jpg")
                for img in defect_images:
                    print(f"   {img}")
                continue
                
            elif user_input:
                test_single_image(model, transform, user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    print("🔬 Fabric Defect Detection - Single Image Tester")
    print("=" * 50)
    
    # Initialize model
    try:
        model, transform = initialize_model()
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Check if image path provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_single_image(model, transform, image_path)
    else:
        # Interactive mode
        interactive_mode(model, transform)

if __name__ == "__main__":
    main()
