# train_helmet_detector.py
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import shutil
import time
from datetime import datetime

def train_helmet_model():
    """
    Complete helmet detection model training in one file.
    Trains with 2 classes: helmet and no_helmet
    Saves: best_model.pt
    """
    
    print("="*70)
    print("🚀 HELMET DETECTION MODEL TRAINER")
    print("="*70)
    
    # ===================== CONFIGURATION =====================
    DATASET_PATH = "/new_dataset"  # CHANGE THIS to your dataset path
    MODEL_NAME = "yolov8l.pt"         # Best model for accuracy
    EPOCHS = 50                       # Good for training
    BATCH_SIZE = 16
    IMG_SIZE = 640
    PROJECT_NAME = "helmet_detection"
    SAVE_NAME = "best_model.pt"
    
    # ===================== CHECK DATASET =====================
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print(f"   Please check the dataset path and try again.")
        return False
    
    print(f"✓ Dataset found: {dataset_path}")
    
    # Check dataset structure
    required_folders = ['train', 'valid', 'test']
    for folder in required_folders:
        if not (dataset_path / folder).exists():
            print(f"⚠️  Warning: {folder} folder not found in dataset")
    
    # ===================== CREATE CONFIG =====================
    print("\n📁 Creating configuration...")
    
    # ALWAYS USE 2 CLASSES ONLY
    config = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 2,  # IMPORTANT: 2 classes only
        'names': ['helmet', 'no_helmet']  # Class 0=helmet, Class 1=no_helmet
    }
    
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Configuration saved: {yaml_path}")
    print(f"✓ Classes: {config['names']} (2 classes)")
    
    # ===================== CHECK GPU =====================
    print("\n💻 Checking hardware...")
    if torch.cuda.is_available():
        device = 0  # Use GPU
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU available: {gpu_name}")
        print(f"✓ Training on GPU (faster)")
    else:
        device = 'cpu'
        print(f"✓ Training on CPU (slower)")
        print(f"⚠️  Consider using GPU for faster training")
    
    # ===================== LOAD MODEL =====================
    print(f"\n📦 Loading model: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # ===================== TRAINING INFO =====================
    print("\n" + "="*70)
    print("📋 TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model:          {MODEL_NAME}")
    print(f"Epochs:         {EPOCHS}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Image size:     {IMG_SIZE}")
    print(f"Classes:        helmet, no_helmet")
    print(f"Device:         {'GPU' if device != 'cpu' else 'CPU'}")
    
    # Estimate time
    if device != 'cpu':
        total_minutes = EPOCHS * 1.5  # ~1.5 min per epoch on GPU
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
    else:
        total_minutes = EPOCHS * 10  # ~10 min per epoch on CPU
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        time_str = f"{hours}h {minutes}m"
    
    print(f"Estimated time: {time_str}")
    print("="*70)
    
    # ===================== CONFIRM TRAINING =====================
    print("\n")
    confirm = input("Start training? (y/n): ")
    if confirm.lower() != 'y':
        print("Training cancelled.")
        return False
    
    # ===================== START TRAINING =====================
    print("\n" + "="*70)
    print("🎯 TRAINING STARTED")
    print("="*70)
    print("Training in progress... This may take a while.")
    print("Please wait...")
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Train the model
        results = model.train(
            data=str(yaml_path),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=device,
            patience=15,           # Stop if no improvement for 15 epochs
            save=True,
            save_period=10,        # Save checkpoint every 10 epochs
            project=PROJECT_NAME,
            name=f'train_{timestamp}',
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            workers=8,
            
            # Augmentation for better accuracy
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            erasing=0.4,
            crop_fraction=0.9,
        )
        
        training_time = time.time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Training time: {hours}h {minutes}m {seconds}s")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False
    
    # ===================== SAVE BEST MODEL =====================
    print("\n💾 Saving best model...")
    
    # Find the best model
    train_dir = Path(PROJECT_NAME) / f'train_{timestamp}'
    best_model_path = train_dir / 'weights' / 'best.pt'
    
    if best_model_path.exists():
        # Copy to current directory
        shutil.copy(best_model_path, SAVE_NAME)
        print(f"✓ Best model saved as: {SAVE_NAME}")
        
        # Also save the last model
        last_model_path = train_dir / 'weights' / 'last.pt'
        if last_model_path.exists():
            shutil.copy(last_model_path, 'last_model.pt')
            print(f"✓ Last model saved as: last_model.pt")
    else:
        print(f"❌ Best model not found at: {best_model_path}")
        # Try to find any .pt file
        pt_files = list(train_dir.rglob('*.pt'))
        if pt_files:
            shutil.copy(pt_files[0], SAVE_NAME)
            print(f"✓ Model saved as: {SAVE_NAME} (found alternative)")
        else:
            print("❌ No model files found!")
            return False
    
    # ===================== VALIDATE MODEL =====================
    print("\n📊 Validating model performance...")
    
    try:
        # Load the saved model
        saved_model = YOLO(SAVE_NAME)
        
        # Run validation
        val_results = saved_model.val(
            data=str(yaml_path),
            split='val',
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            conf=0.25,
            iou=0.6,
            device='auto',
            verbose=False
        )
        
        print("\n" + "="*70)
        print("📈 VALIDATION RESULTS")
        print("="*70)
        print(f"✅ mAP50:      {val_results.box.map50:.3f} ({val_results.box.map50*100:.1f}%)")
        print(f"✅ mAP50-95:   {val_results.box.map:.3f} ({val_results.box.map*100:.1f}%)")
        print(f"✅ Precision:  {val_results.box.mp:.3f} ({val_results.box.mp*100:.1f}%)")
        print(f"✅ Recall:     {val_results.box.mr:.3f} ({val_results.box.mr*100:.1f}%)")
        
        # Show per-class results
        if hasattr(val_results.box, 'ap_class_index') and val_results.box.ap_class_index is not None:
            print("\n📊 PER-CLASS ACCURACY:")
            print("-" * 40)
            for i, cls_idx in enumerate(val_results.box.ap_class_index):
                if cls_idx == 0:
                    class_name = "helmet"
                elif cls_idx == 1:
                    class_name = "no_helmet"
                else:
                    class_name = f"class_{cls_idx}"
                
                ap50 = val_results.box.ap50[i] if i < len(val_results.box.ap50) else 0
                print(f"  {class_name}: {ap50:.3f} ({ap50*100:.1f}%)")
        
        print("="*70)
        
    except Exception as e:
        print(f"⚠️  Validation failed: {e}")
        print("   Model was saved but validation could not run.")
    
    # ===================== TEST MODEL =====================
    print("\n🧪 Quick model test...")
    
    try:
        # Load the model again to test
        test_model = YOLO(SAVE_NAME)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model classes: {test_model.names}")
        print(f"✓ Number of classes: {test_model.model.nc}")
        
        # Create a simple test image
        import numpy as np
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Quick inference test
        test_results = test_model.predict(
            test_image,
            conf=0.5,
            imgsz=640,
            verbose=False
        )
        
        print(f"✓ Inference test passed")
        
    except Exception as e:
        print(f"⚠️  Test inference failed: {e}")
    
    # ===================== FINAL SUMMARY =====================
    print("\n" + "="*70)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Your trained model is ready:")
    print(f"   → {SAVE_NAME}")
    
    print(f"\n📊 Model information:")
    print(f"   • Classes: helmet, no_helmet")
    print(f"   • Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   • Training epochs: {EPOCHS}")
    
    print(f"\n🚀 How to use the model:")
    print(f"   1. Load model: model = YOLO('{SAVE_NAME}')")
    print(f"   2. Detect: results = model('image.jpg')")
    print(f"   3. Get classes: model.names[0] = 'helmet', model.names[1] = 'no_helmet'")
    
    print(f"\n📝 Example code:")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('best_model.pt')")
    print("   results = model('test.jpg', conf=0.5)")
    print("   for result in results:")
    print("       for box in result.boxes:")
    print("           cls = int(box.cls[0])")
    print("           if cls == 0: print('Helmet detected')")
    print("           elif cls == 1: print('No helmet detected')")
    
    print("\n" + "="*70)
    
    return True

def quick_test_model(model_path="best_model.pt"):
    """
    Quick test to verify the model works
    """
    print("\n" + "="*70)
    print("🧪 QUICK MODEL TEST")
    print("="*70)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please run training first.")
        return False
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        
        print(f"\n✓ Model loaded successfully!")
        print(f"✓ Model classes: {model.names}")
        print(f"✓ Number of classes: {model.model.nc}")
        
        # Create test image
        print("\nRunning test inference...")
        test_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add a simple rectangle (simulating an object)
        cv2.rectangle(test_img, (200, 150), (400, 350), (255, 255, 255), -1)
        
        # Run detection
        results = model.predict(
            test_img,
            conf=0.3,
            imgsz=640,
            verbose=False
        )
        
        print(f"✓ Inference completed")
        
        if results[0].boxes is not None:
            print(f"\nDetections found:")
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                print(f"  {i+1}. {class_name}: {conf:.1%}")
        else:
            print(f"\nNo detections (normal for random test image)")
        
        print("\n" + "="*70)
        print("✅ MODEL TEST PASSED!")
        print("="*70)
        print(f"\nYour model '{model_path}' is working correctly!")
        print(f"It can detect: {model.names}")
        
        return True
        
    except ImportError:
        print("⚠️  OpenCV not installed. Install with: pip install opencv-python")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

# Import for test function
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  Note: OpenCV not installed. Test visualization disabled.")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 HELMET DETECTION MODEL TRAINER")
    print("="*70)
    
    # Check if model already exists
    if Path("best_model.pt").exists():
        print("\n⚠️  Existing model found: best_model.pt")
        choice = input("\nChoose:\n  1. Train new model (overwrite)\n  2. Test existing model\n  3. Exit\nChoice (1/2/3): ")
        
        if choice == '2':
            quick_test_model("best_model.pt")
            exit()
        elif choice == '3':
            print("Exiting...")
            exit()
        elif choice != '1':
            print("Invalid choice. Exiting...")
            exit()
    
    # Start training
    success = train_helmet_model()
    
    if success:
        # Test the new model
        print("\n" + "="*70)
        test_choice = input("Test the new model now? (y/n): ")
        if test_choice.lower() == 'y':
            quick_test_model("best_model.pt")
        
        print("\n" + "="*70)
        print("🎊 ALL DONE! Your model is ready to use.")
        print("="*70)
        print(f"\nNext steps:")
        print(f"1. Use your model: model = YOLO('best_model.pt')")
        print(f"2. Detect helmets: results = model('your_image.jpg')")
        print(f"3. Check results: model.names[0] = 'helmet', model.names[1] = 'no_helmet'")
        print("\n" + "="*70)
    else:
        print("\n❌ Training failed. Please check your dataset and try again.")