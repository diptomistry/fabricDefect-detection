"""
Preprocessing Comparison Tool
============================

This script compares the original model with the enhanced preprocessing version
to demonstrate the impact of advanced image preprocessing on fabric defect detection.

Usage:
    python compare_preprocessing.py
"""

import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from enhanced_fabric_defect_detector import AdvancedPreprocessor

def compare_preprocessing_effects():
    """Compare original vs preprocessed images visually"""
    
    # Get sample images
    defect_dir = 'defect'
    normal_dir = 'noDefect' 
    
    # Sample images for comparison
    defect_img = os.path.join(defect_dir, 'IMG_0109.jpg')
    normal_img = os.path.join(normal_dir, 'IMG_0125.jpg')
    
    if not os.path.exists(defect_img) or not os.path.exists(normal_img):
        print("Sample images not found!")
        return
    
    # Initialize preprocessor
    preprocessor = AdvancedPreprocessor(
        enable_histogram_eq=True,
        enable_noise_reduction=True,
        enable_edge_enhancement=True,
        enable_contrast_enhancement=True
    )
    
    # Process images
    print("Processing sample images...")
    
    for img_path, img_type in [(defect_img, 'Defect'), (normal_img, 'Normal')]:
        # Load original
        original = Image.open(img_path).convert('RGB')
        original_resized = original.resize((128, 128))
        
        # Process with enhanced preprocessing
        processed = preprocessor.process_image(original)
        processed_resized = processed.resize((128, 128))
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_resized)
        axes[0].set_title(f'{img_type} - Original')
        axes[0].axis('off')
        
        axes[1].imshow(processed_resized)
        axes[1].set_title(f'{img_type} - Enhanced Preprocessing')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        output_path = f'preprocessing_comparison_{img_type.lower()}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison: {output_path}")
        plt.close()

def test_both_models():
    """Test both original and enhanced models"""
    print("\n" + "="*50)
    print("MODEL COMPARISON TEST")
    print("="*50)
    
    # Test with a few sample images
    test_images = [
        'defect/IMG_0109.jpg',
        'defect/IMG_0121.jpg', 
        'noDefect/IMG_0125.jpg',
        'noDefect/IMG_0126.jpg'
    ]
    
    print("Note: This is a conceptual comparison.")
    print("To run both models, you would:")
    print("1. Run: python fabric_defect_detector.py")
    print("2. Run: python enhanced_fabric_defect_detector.py")
    print("3. Compare the anomaly scores for the same images")
    
    print("\nExpected improvements with preprocessing:")
    print("• Better separation between normal/defect scores")
    print("• More consistent detection across lighting conditions")
    print("• Reduced false positives from noise")
    print("• Enhanced detection of subtle defects")

if __name__ == "__main__":
    print("Fabric Defect Detection - Preprocessing Comparison")
    print("=" * 55)
    
    # Check if required packages are available
    try:
        import cv2
        import matplotlib.pyplot as plt
        
        print("✅ All dependencies available")
        print("\n1. Creating visual comparisons...")
        compare_preprocessing_effects()
        
        print("\n2. Model comparison guidance...")
        test_both_models()
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nTo install missing packages:")
        print("pip install opencv-python matplotlib")
        
        print("\n2. Model comparison guidance...")
        test_both_models()
    
    print("\n" + "="*55)
    print("Comparison completed!")
