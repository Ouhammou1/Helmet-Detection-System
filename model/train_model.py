# train_optimized_accuracy.py
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil
import os
import time
import gc

# Fix Polars warning but keep performance
os.environ['POLARS_SKIP_CPU_CHECK'] = '1'

class OptimizedHelmetTrainer:
    def __init__(self):
        self.dataset_path = Path("new_dataset/")
        self.model_output = "best_model.pt"
        
        # OPTIMAL SETTINGS FOR ACCURACY (balanced for Apple Silicon)
        self.epochs = 100  # More epochs = better accuracy
        self.batch_size = 8  # Balance between memory and accuracy
        self.img_size = 640  # Full size for best accuracy
        self.patience = 30  # Early stopping
        
    def setup_environment(self):
        """Setup optimal environment for Apple Silicon"""
        print("🛠️  Setting up optimized environment...")
        
        # Clear memory before starting
        gc.collect()
        
        # Set optimal threads for Apple Silicon
        try:
            # On Apple Silicon, control threads for better performance
            os.environ['OMP_NUM_THREADS'] = '4'
            os.environ['MKL_NUM_THREADS'] = '4'
            print("✓ Thread optimization set for Apple Silicon")
        except:
            pass
        
        # Check if using MPS (Apple Metal Performance Shaders)
        import torch
        if torch.backends.mps.is_available():
            print("✅ Apple MPS (Metal) available!")
            self.device = 'mps'  # Use Apple Metal for GPU acceleration
        elif torch.cuda.is_available():
            print("✅ CUDA GPU available")
            self.device = 0
        else:
            print("⚠️  Using CPU (try to enable MPS)")
            self.device = 'cpu'
        
        return True
    
    def create_optimized_config(self):
        """Create optimized training configuration"""
        print("\n📁 Creating optimized config...")
        
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': 2,
            'names': ['helmet', 'no_helmet']
        }
        
        yaml_path = self.dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Config created with {config['nc']} classes")
        return str(yaml_path)
    
    def select_best_model(self):
        """Select the best model size for accuracy"""
        print("\n🤔 Selecting optimal model architecture...")
        
        # Try different models (larger = more accurate but needs more memory)
        models_to_try = [
            ('yolov8l.pt', 'Large (best accuracy, needs 8GB+ RAM)'),
            ('yolov8m.pt', 'Medium (good balance)'),
            ('yolov8s.pt', 'Small (efficient)'),
            ('yolov8n.pt', 'Nano (lightweight)')
        ]
        
        # Check available memory
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            print(f"📊 Available RAM: {ram_gb:.1f} GB")
            
            if ram_gb >= 16:
                print("✅ RAM ≥16GB: Using Large model for max accuracy")
                return 'yolov8l.pt'
            elif ram_gb >= 8:
                print("✅ RAM ≥8GB: Using Medium model")
                return 'yolov8m.pt'
            elif ram_gb >= 4:
                print("⚠️  RAM 4-8GB: Using Small model")
                return 'yolov8s.pt'
            else:
                print("⚠️  RAM <4GB: Using Nano model")
                return 'yolov8n.pt'
                
        except:
            print("⚠️  Can't detect RAM, using Medium model")
            return 'yolov8m.pt'
    
    def train_with_accuracy_optimization(self):
        """Train with maximum accuracy settings"""
        print("\n" + "="*70)
        print("🎯 ACCURACY-OPTIMIZED TRAINING")
        print("="*70)
        
        # Setup
        self.setup_environment()
        yaml_path = self.create_optimized_config()
        model_name = self.select_best_model()
        
        print(f"\n📦 Loading model: {model_name}")
        model = YOLO(model_name)
        
        # Training configuration for maximum accuracy
        train_args = {
            'data': yaml_path,
            'epochs': self.epochs,
            'imgsz': self.img_size,
            'batch': self.batch_size,
            'device': self.device,
            'patience': self.patience,
            'save': True,
            'save_period': 10,
            'project': 'accuracy_optimized',
            'name': 'max_accuracy',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'seed': 42,
            
            # CRITICAL: Augmentation for accuracy (re-enabled!)
            'augment': True,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 15.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.001,
            'flipud': 0.0,
            'fliplr': 0.5,
            
            # ENABLE advanced augmentations (for accuracy)
            'mosaic': 0.5,    # Reduced but enabled
            'mixup': 0.1,     # Reduced but enabled
            'copy_paste': 0.1,  # Reduced but enabled
            
            # Learning rate for accuracy
            'lr0': 0.01,      # Initial learning rate
            'lrf': 0.01,      # Final learning rate factor
            
            # Regularization to prevent overfitting
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Memory optimization (critical for Apple Silicon)
            'workers': 2,      # Limited but not zero
            'single_cls': False,
            'rect': False,
            'cache': False,    # Disable cache to save memory
            'persistent_workers': False,
        }
        
        print("\n" + "="*70)
        print("⚙️  OPTIMIZED TRAINING CONFIGURATION")
        print("="*70)
        print(f"Model:          {model_name}")
        print(f"Epochs:         {self.epochs}")
        print(f"Batch size:     {self.batch_size}")
        print(f"Image size:     {self.img_size}")
        print(f"Device:         {self.device}")
        print(f"Augmentation:   ENABLED (mosaic, mixup, copy-paste)")
        print(f"Workers:        2 (optimized for Apple Silicon)")
        print("="*70)
        
        # Start training with memory monitoring
        print("\n🚀 Starting training with accuracy optimization...")
        print("   Training will automatically save checkpoints\n")
        
        start_time = time.time()
        
        try:
            # Train with periodic garbage collection
            results = self.train_with_memory_management(model, train_args)
            
            training_time = time.time() - start_time
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            
            print("\n" + "="*70)
            print(f"✅ TRAINING COMPLETE! ({hours}h {minutes}m)")
            print("="*70)
            
            # Save best model
            self.save_best_model()
            
            # Validate
            self.validate_model(yaml_path)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            print("\nTrying fallback strategy...")
            return self.fallback_training(yaml_path)
    
    def train_with_memory_management(self, model, train_args):
        """Train with intelligent memory management"""
        import gc
        
        print("🧠 Using intelligent memory management...")
        
        # Modify args for memory safety
        train_args['verbose'] = True
        train_args['plots'] = False  # Disable plots to save memory
        
        # Train with periodic garbage collection
        results = model.train(**train_args)
        
        # Force garbage collection after training
        gc.collect()
        
        return results
    
    def fallback_training(self, yaml_path):
        """Fallback training if primary fails"""
        print("\n" + "="*70)
        print("🔄 FALLBACK TRAINING STRATEGY")
        print("="*70)
        
        # Use smaller model but keep accuracy settings
        print("Using smaller model with optimal accuracy settings...")
        
        model = YOLO('yolov8m.pt')  # Medium model as fallback
        
        # Train with reduced batch but full augmentation
        train_args = {
            'data': yaml_path,
            'epochs': 50,  # Reduced epochs
            'imgsz': 640,
            'batch': 4,    # Smaller batch
            'device': self.device,
            'patience': 20,
            'save': True,
            'project': 'fallback_training',
            'name': 'fallback',
            'verbose': False,
            
            # Keep augmentations for accuracy
            'augment': True,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'fliplr': 0.5,
            'degrees': 10.0,
            
            # Disable heavy augmentations to save memory
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            
            'workers': 0,  # No workers for stability
        }
        
        try:
            results = model.train(**train_args)
            self.save_best_model_from_dir('fallback_training/fallback/weights')
            return True
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            return False
    
    def save_best_model(self):
        """Save the best model"""
        print("\n💾 Saving best model...")
        
        train_dir = Path('accuracy_optimized/max_accuracy/weights')
        if train_dir.exists():
            best_model = train_dir / 'best.pt'
            if best_model.exists():
                shutil.copy(best_model, self.model_output)
                print(f"✅ Best model saved: {self.model_output}")
                
                # Also save metrics
                for file in ['results.csv', 'args.yaml']:
                    source = train_dir.parent / file
                    if source.exists():
                        shutil.copy(source, file)
                        print(f"✓ {file} saved")
                
                return True
        
        print("⚠️  Could not find best model, trying last model...")
        return self.save_best_model_from_dir('accuracy_optimized/max_accuracy/weights')
    
    def save_best_model_from_dir(self, dir_path):
        """Save best or last model from directory"""
        train_dir = Path(dir_path)
        if train_dir.exists():
            # Try best model first
            for model_file in ['best.pt', 'last.pt']:
                source = train_dir / model_file
                if source.exists():
                    shutil.copy(source, self.model_output)
                    print(f"✅ Model saved: {self.model_output}")
                    return True
        return False
    
    def validate_model(self, yaml_path):
        """Validate model performance"""
        print("\n📊 Validating model accuracy...")
        
        if not Path(self.model_output).exists():
            print("❌ Model not found for validation")
            return
        
        try:
            model = YOLO(self.model_output)
            
            # Run comprehensive validation
            val_results = model.val(
                data=yaml_path,
                split='val',
                imgsz=self.img_size,
                batch=8,
                conf=0.25,
                iou=0.6,
                device='auto',
                verbose=False
            )
            
            print("\n" + "="*70)
            print("🎯 ACCURACY REPORT")
            print("="*70)
            print(f"mAP50:     {val_results.box.map50:.4f} ({val_results.box.map50*100:.1f}%)")
            print(f"mAP50-95:  {val_results.box.map:.4f} ({val_results.box.map*100:.1f}%)")
            print(f"Precision: {val_results.box.mp:.4f} ({val_results.box.mp*100:.1f}%)")
            print(f"Recall:    {val_results.box.mr:.4f} ({val_results.box.mr*100:.1f}%)")
            
            # Save validation results
            with open('accuracy_report.txt', 'w') as f:
                f.write("Accuracy Report\n")
                f.write("="*50 + "\n")
                f.write(f"Model: {self.model_output}\n")
                f.write(f"mAP50: {val_results.box.map50:.4f} ({val_results.box.map50*100:.1f}%)\n")
                f.write(f"Precision: {val_results.box.mp:.4f} ({val_results.box.mp*100:.1f}%)\n")
                f.write(f"Recall: {val_results.box.mr:.4f} ({val_results.box.mr*100:.1f}%)\n")
            
            print(f"\n📄 Report saved: accuracy_report.txt")
            print("="*70)
            
        except Exception as e:
            print(f"⚠️  Validation error: {e}")
    
    def create_accuracy_tips(self):
        """Create tips for improving accuracy"""
        print("\n" + "="*70)
        print("💡 ACCURACY IMPROVEMENT TIPS")
        print("="*70)
        
        tips = """
        1. **More Data**: Add more diverse helmet/no-helmet images
        2. **Better Annotations**: Ensure precise bounding boxes
        3. **Class Balance**: Equal number of helmet/no-helmet samples
        4. **Higher Resolution**: Use 640x640 or higher if memory allows
        5. **More Epochs**: Train for 100-200 epochs
        6. **Advanced Augmentation**: Enable mosaic, mixup, copy-paste
        7. **Larger Model**: Use yolov8l.pt or yolov8x.pt
        8. **Hyperparameter Tuning**: Adjust learning rate, optimizer
        9. **Transfer Learning**: Start from custom pretrained weights
        10. **Ensemble**: Combine multiple models
        """
        
        print(tips)
        
        with open('accuracy_tips.txt', 'w') as f:
            f.write(tips)

def main():
    """Main function"""
    print("\n" + "="*70)
    print("🎯 MAXIMUM ACCURACY HELMET DETECTION TRAINING")
    print("="*70)
    print("\nThis will train with optimal settings for accuracy")
    print("while managing memory on Apple Silicon.\n")
    
    # Check existing model
    if Path("best_model.pt").exists():
        print("⚠️  Existing model found")
        choice = input("\nOverwrite and train new model? (y/n): ")
        if choice.lower() != 'y':
            print("Keeping existing model.")
            
            # Test existing model
            from ultralytics import YOLO
            model = YOLO("best_model.pt")
            print(f"\nExisting model classes: {model.names}")
            return
    
    # Create trainer
    trainer = OptimizedHelmetTrainer()
    
    # Start training
    print("\n" + "="*70)
    print("⚡ Starting accuracy-optimized training...")
    print("="*70 + "\n")
    
    success = trainer.train_with_accuracy_optimization()
    
    if success:
        trainer.create_accuracy_tips()
        
        print("\n" + "="*70)
        print("🎉 TRAINING SUCCESSFUL!")
        print("="*70)
        print(f"\n📁 Model saved: best_model.pt")
        print(f"📊 Accuracy report: accuracy_report.txt")
        print(f"💡 Tips: accuracy_tips.txt")
        
        print("\n🚀 To use your high-accuracy model:")
        print("""
from ultralytics import YOLO

# Load your trained model
model = YOLO('best_model.pt')

# Detect with high confidence
results = model('test.jpg', conf=0.5)

# Process results
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:
            print(f"✅ Helmet: {float(box.conf[0]):.1%}")
        else:
            print(f"❌ No Helmet: {float(box.conf[0]):.1%}")
        """)
        
        print("="*70)
    else:
        print("\n❌ Training failed. Try the fallback options.")

if __name__ == "__main__":
    # Run with memory optimization
    main()