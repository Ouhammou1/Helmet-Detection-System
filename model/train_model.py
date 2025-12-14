# complete_helmet_trainer.py
import yaml
import os
import glob
import shutil
import time
import gc
from pathlib import Path
from ultralytics import YOLO

class CompleteHelmetTrainer:
    def __init__(self):
        self.dataset_path = Path("new_dataset/")
        self.model_output = "best_model.pt"
        
        # Memory-safe training settings for Apple Silicon
        self.epochs = 50
        self.batch_size = 2      # VERY small for memory safety
        self.img_size = 320      # Reduced from 640
        self.patience = 20
        
        # Clear memory warning
        os.environ['POLARS_SKIP_CPU_CHECK'] = '1'
        
    def fix_dataset_labels(self):
        """Fix all label files with class ID issues"""
        print("🔧 Fixing dataset labels...")
        
        for split in ['train', 'valid', 'test']:
            labels_dir = self.dataset_path / split / 'labels'
            if not labels_dir.exists():
                continue
                
            label_files = list(labels_dir.glob('*.txt'))
            fixed_count = 0
            removed_count = 0
            
            for file_path in label_files:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                corrected_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                        
                    class_id = int(parts[0])
                    
                    # Fix class ID issues
                    if class_id == 2:
                        # Option 1: Map class 2 to class 1 (if it was 'person')
                        parts[0] = '1'
                        removed_count += 1
                    
                    # Ensure class is 0 or 1
                    if int(parts[0]) in [0, 1]:
                        corrected_lines.append(' '.join(parts))
                
                # Write back if changed
                if len(corrected_lines) != len(lines):
                    with open(file_path, 'w') as f:
                        f.write('\n'.join(corrected_lines))
                    fixed_count += 1
            
            if fixed_count > 0:
                print(f"✓ {split}: Fixed {fixed_count} files, corrected {removed_count} class 2 labels")
        
        print("✅ Dataset labels fixed!")
        return True
    
    def check_dataset_stats(self):
        """Check dataset statistics"""
        print("\n📊 Checking dataset statistics...")
        
        for split in ['train', 'valid']:
            images_dir = self.dataset_path / split / 'images'
            labels_dir = self.dataset_path / split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                image_count = len(list(images_dir.glob('*')))
                label_count = len(list(labels_dir.glob('*.txt')))
                
                # Count objects by class
                class_counts = {0: 0, 1: 0}
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                if class_id in class_counts:
                                    class_counts[class_id] += 1
                
                print(f"{split.upper()}:")
                print(f"  Images: {image_count}, Labels: {label_count}")
                print(f"  Objects: helmet={class_counts[0]}, no_helmet={class_counts[1]}")
                
                if class_counts[0] == 0 or class_counts[1] == 0:
                    print(f"⚠️  Warning: One class has zero samples!")
        
        return True
    
    def create_config(self):
        """Create data.yaml configuration file"""
        print("\n📁 Creating dataset configuration...")
        
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'nc': 2,
            'names': ['helmet', 'no_helmet']
        }
        
        yaml_path = self.dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Config created: {yaml_path}")
        print(f"✓ Classes: {config['names']}")
        
        return str(yaml_path)
    
    def train_model(self):
        """Main training function with memory safety"""
        print("\n" + "="*70)
        print("🚀 STARTING COMPLETE TRAINING PROCESS")
        print("="*70)
        
        # Step 1: Fix dataset
        self.fix_dataset_labels()
        
        # Step 2: Check dataset
        self.check_dataset_stats()
        
        # Step 3: Create config
        yaml_path = self.create_config()
        
        # Step 4: Load appropriate model
        print(f"\n📦 Loading model...")
        
        # Try to use a small model for memory safety
        try:
            model = YOLO('yolov8n.pt')  # Nano model - smallest
            print("✓ Using: YOLOv8n (Nano - 3.2M parameters)")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
        
        # Step 5: Configure training
        print("\n" + "="*70)
        print("⚙️  TRAINING CONFIGURATION")
        print("="*70)
        print(f"Epochs:         {self.epochs}")
        print(f"Batch size:     {self.batch_size} (memory-safe)")
        print(f"Image size:     {self.img_size}x{self.img_size}")
        print(f"Device:         CPU (Apple Silicon optimized)")
        print(f"Workers:        0 (no multiprocessing)")
        print("="*70)
        
        print("\n⚠️  IMPORTANT: Training with memory-safe settings")
        print("   This prevents 'killed' errors on Apple Silicon")
        
        # Step 6: Clear memory before training
        gc.collect()
        
        # Step 7: Start training
        print("\n🎯 Starting training...")
        start_time = time.time()
        
        try:
            # MEMORY-SAFE TRAINING SETTINGS
            results = model.train(
                data=yaml_path,
                epochs=self.epochs,
                imgsz=self.img_size,
                batch=self.batch_size,
                device='cpu',
                workers=0,           # Critical: no multiprocessing
                patience=self.patience,
                save=True,
                save_period=10,
                project='complete_training',
                name='helmet_model',
                exist_ok=True,
                pretrained=True,
                verbose=True,        # Show progress
                seed=42,
                
                # BASIC augmentation only (memory-safe)
                augment=True,
                hsv_h=0.01,
                hsv_s=0.5,
                hsv_v=0.3,
                degrees=5.0,
                translate=0.1,
                fliplr=0.5,
                
                # DISABLE memory-intensive features
                mosaic=0.0,         # NO mosaic
                mixup=0.0,          # NO mixup
                copy_paste=0.0,     # NO copy-paste
                cache=False,        # NO caching
                
                # Learning rate
                lr0=0.01,
                lrf=0.01,
            )
            
            training_time = time.time() - start_time
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            
            print("\n" + "="*70)
            print(f"✅ TRAINING COMPLETE! ({hours}h {minutes}m)")
            print("="*70)
            
            # Save the model
            self.save_model()
            
            # Validate
            self.validate_model(yaml_path)
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            self.save_last_model()
            return False
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            return False
    
    def save_model(self):
        """Save the best trained model"""
        print("\n💾 Saving model...")
        
        train_dir = Path('complete_training/helmet_model/weights')
        
        if train_dir.exists():
            # Try to save best model
            best_model = train_dir / 'best.pt'
            if best_model.exists():
                shutil.copy(best_model, self.model_output)
                print(f"✅ Best model saved: {self.model_output}")
                
                # Also save last model as backup
                last_model = train_dir / 'last.pt'
                if last_model.exists():
                    shutil.copy(last_model, 'last_model.pt')
                    print(f"✓ Backup saved: last_model.pt")
                
                return True
        
        print("⚠️  Could not find trained model")
        return False
    
    def save_last_model(self):
        """Save last model if training was interrupted"""
        train_dir = Path('complete_training/helmet_model/weights')
        if train_dir.exists():
            last_model = train_dir / 'last.pt'
            if last_model.exists():
                shutil.copy(last_model, self.model_output)
                print(f"✓ Last model saved as: {self.model_output}")
                return True
        return False
    
    def validate_model(self, yaml_path):
        """Validate the trained model"""
        print("\n📊 Validating model performance...")
        
        if not Path(self.model_output).exists():
            print("❌ Model not found for validation")
            return
        
        try:
            model = YOLO(self.model_output)
            
            print(f"✓ Model loaded: {model.names}")
            
            # Quick validation
            val_results = model.val(
                data=yaml_path,
                split='val',
                imgsz=self.img_size,
                batch=4,
                conf=0.25,
                iou=0.6,
                device='cpu',
                verbose=False
            )
            
            print("\n" + "="*60)
            print("📈 VALIDATION RESULTS")
            print("="*60)
            print(f"mAP50:     {val_results.box.map50:.4f} ({val_results.box.map50*100:.1f}%)")
            print(f"Precision: {val_results.box.mp:.4f} ({val_results.box.mp*100:.1f}%)")
            print(f"Recall:    {val_results.box.mr:.4f} ({val_results.box.mr*100:.1f}%)")
            print("="*60)
            
            # Save results
            with open('training_results.txt', 'w') as f:
                f.write(f"Model: {self.model_output}\n")
                f.write(f"mAP50: {val_results.box.map50:.4f}\n")
                f.write(f"Precision: {val_results.box.mp:.4f}\n")
                f.write(f"Recall: {val_results.box.mr:.4f}\n")
            
            print(f"✓ Results saved: training_results.txt")
            
        except Exception as e:
            print(f"⚠️  Validation error: {e}")
    
    def create_usage_example(self):
        """Create example code for using the model"""
        print("\n" + "="*70)
        print("💻 HOW TO USE YOUR TRAINED MODEL")
        print("="*70)
        
        example_code = '''
from ultralytics import YOLO
import cv2

# 1. Load your trained model
model = YOLO('best_model.pt')

# 2. Detect in an image
results = model('your_image.jpg', conf=0.5)

# 3. Process and display results
for result in results:
    # Get detections
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]
            
            if cls_id == 0:
                print(f"✅ Helmet detected: {confidence:.1%}")
            else:
                print(f"❌ No helmet: {confidence:.1%}")
    
    # Save annotated image
    result.save('result.jpg')
    print("✓ Result saved as result.jpg")

# 4. For webcam detection
# model.predict(source=0, show=True, conf=0.5)
'''
        
        print(example_code)
        
        # Save to file
        with open('usage_example.py', 'w') as f:
            f.write(example_code)
        
        print("✓ Example code saved: usage_example.py")
        print("="*70)

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🎯 COMPLETE HELMET DETECTION TRAINER")
    print("="*70)
    print("\nThis script will:")
    print("1. 🔧 Fix dataset label issues (class 2 errors)")
    print("2. 📊 Check dataset statistics")
    print("3. 🚀 Train for 50 epochs with memory-safe settings")
    print("4. 💾 Save best_model.pt")
    print("5. 📈 Validate and create usage example")
    print("="*70)
    
    # Check if model exists
    if Path("best_model.pt").exists():
        print("\n⚠️  Existing model found: best_model.pt")
        choice = input("Overwrite and train new? (y/n): ")
        if choice.lower() != 'y':
            print("Exiting...")
            return
    
    # Create trainer
    trainer = CompleteHelmetTrainer()
    
    # Confirm
    print(f"\nTraining configuration:")
    print(f"  Epochs: {trainer.epochs}")
    print(f"  Batch: {trainer.batch_size}")
    print(f"  Image size: {trainer.img_size}")
    print(f"  Estimated time: 2-4 hours on CPU")
    
    confirm = input("\nStart complete training process? (y/n): ")
    if confirm.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Run complete training
    print("\n" + "="*70)
    print("🚀 STARTING COMPLETE TRAINING PROCESS")
    print("="*70)
    
    success = trainer.train_model()
    
    if success:
        trainer.create_usage_example()
        
        print("\n" + "="*70)
        print("🎉 TRAINING PROCESS COMPLETE!")
        print("="*70)
        print(f"\n📁 Output files:")
        print(f"  → best_model.pt          (your trained model)")
        print(f"  → last_model.pt          (backup model)")
        print(f"  → training_results.txt   (performance metrics)")
        print(f"  → usage_example.py       (how to use the model)")
        print(f"  → complete_training/     (full training logs)")
        print("\n🚀 To test your model immediately:")
        print("   python usage_example.py")
        print("="*70)
    else:
        print("\n❌ Training failed. Check the error messages above.")

if __name__ == "__main__":
    # Clear memory and run
    gc.collect()
    main()