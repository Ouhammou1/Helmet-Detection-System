import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from ultralytics import YOLO
import uuid
from werkzeug.utils import secure_filename
import base64
import threading
import queue
import time
from moviepy.editor import VideoFileClip

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {
    'png', 'jpg', 'jpeg', 'bmp', 'gif',  # Images
    'mp4', 'avi', 'mov', 'mkv', 'webm'   # Videos
}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create uploads directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Video processing queue
video_processing_queue = queue.Queue()
video_results = {}
processing_lock = threading.Lock()

# Load YOLOv8 model
try:
    model = YOLO('best_model.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process image with clear HELMET DETECTED/NO HELMET DETECTED labels"""
    if model is None:
        return {"error": "Model not loaded"}, []
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not read image"}, []
    
    # Run inference
    results = model(img, conf=0.25)
    
    # Process results
    detections = []
    helmet_count = 0
    no_helmet_count = 0
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(confidence),
                    "class": class_name,
                    "class_id": class_id
                })
                
                if class_name == "Helmet":
                    helmet_count += 1
                else:
                    no_helmet_count += 1
    
    # Draw bounding boxes with clear labels
    output_img = img.copy()
    height, width = output_img.shape[:2]
    
    # Add top header for status
    header_height = 60
    header = np.zeros((header_height, width, 3), dtype=np.uint8)
    
    # Determine header color and text
    if no_helmet_count > 0:
        header_color = (0, 0, 255)  # Red
        status_text = f"ALERT: {no_helmet_count} NO HELMET DETECTED"
    elif helmet_count > 0:
        header_color = (0, 255, 0)  # Green
        status_text = f"SAFE: {helmet_count} HELMET DETECTED"
    else:
        header_color = (255, 165, 0)  # Orange
        status_text = "NO DETECTIONS"
    
    header[:] = header_color
    
    # Add status text to header
    font_scale = min(width / 800, 1.5)  # Scale font based on image width
    thickness = max(2, int(width / 400))
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
    text_x = max(10, (width - text_size[0]) // 2)
    text_y = (header_height + text_size[1]) // 2
    cv2.putText(header, status_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness)
    
    # Add header to image
    output_img = np.vstack([header, output_img])
    
    # Draw each detection with clear labels
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        y1 += header_height
        y2 += header_height
        
        if det["class"] == "Helmet":
            color = (0, 255, 0)  # Green
            label_text = "HELMET DETECTED"
        else:
            color = (0, 0, 255)  # Red
            label_text = "NO HELMET DETECTED"
        
        # Draw thick bounding box
        box_thickness = max(2, int(min(x2-x1, y2-y1) / 50))
        cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thickness)
        
        # Draw label background
        confidence_text = f"{label_text} {det['confidence']:.0%}"
        label_font_scale = max(0.5, min(width / 1000, 0.8))
        label_thickness = max(1, int(width / 500))
        label_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_DUPLEX, label_font_scale, label_thickness)[0]
        
        # Label background
        label_bg_y1 = max(int(y1) - label_size[1] - 10, 0)
        label_bg_y2 = int(y1)
        label_bg_x2 = min(int(x1) + label_size[0] + 15, width)
        
        cv2.rectangle(output_img,
                     (int(x1), label_bg_y1),
                     (label_bg_x2, label_bg_y2),
                     color, -1)
        
        # Label text
        text_y_pos = max(label_bg_y1 + label_size[1] - 5, 15)
        cv2.putText(output_img, confidence_text,
                   (int(x1) + 8, text_y_pos),
                   cv2.FONT_HERSHEY_DUPLEX, label_font_scale, (255, 255, 255), label_thickness)
    
    # Add footer with statistics
    footer_height = 40
    footer = np.zeros((footer_height, width, 3), dtype=np.uint8)
    footer[:] = (40, 40, 40)
    
    stats_text = f"Total: {len(detections)} | Helmets: {helmet_count} | No Helmets: {no_helmet_count}"
    footer_font_scale = min(width / 1000, 0.7)
    stats_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, footer_font_scale, 1)[0]
    stats_x = max(10, (width - stats_size[0]) // 2)
    stats_y = (footer_height + stats_size[1]) // 2
    cv2.putText(footer, stats_text, (stats_x, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, footer_font_scale, (220, 220, 220), 1)
    
    output_img = np.vstack([output_img, footer])
    
    # Save processed image
    output_path = image_path.replace('.', '_processed.')
    cv2.imwrite(output_path, output_img)
    
    result_data = {
        "detections": detections,
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "total_detections": len(detections)
    }
    
    return result_data, output_path

def process_video_worker():
    """Background worker for video processing"""
    while True:
        try:
            task = video_processing_queue.get()
            if task is None:
                break
                
            video_id, video_path = task
            
            with processing_lock:
                video_results[video_id] = {
                    'status': 'processing',
                    'progress': 0
                }
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{video_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 100))  # Extra height for header/footer
            
            frame_count = 0
            helmet_total = 0
            no_helmet_total = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = model(frame, conf=0.25)
                
                frame_helmet = 0
                frame_no_helmet = 0
                
                # Create header
                header_height = 50
                header = np.zeros((header_height, width, 3), dtype=np.uint8)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = model.names[class_id]
                            
                            color = (0, 255, 0) if class_name == "Helmet" else (0, 0, 255)
                            label = "HELMET DETECTED" if class_name == "Helmet" else "NO HELMET DETECTED"
                            
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f"{label} {confidence:.0%}", 
                                       (int(x1), int(y1)-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            if class_name == "Helmet":
                                frame_helmet += 1
                                helmet_total += 1
                            else:
                                frame_no_helmet += 1
                                no_helmet_total += 1
                
                # Add header with frame info
                if frame_no_helmet > 0:
                    header_color = (0, 0, 255)
                    header_text = f"ALERT: {frame_no_helmet} NO HELMET"
                elif frame_helmet > 0:
                    header_color = (0, 255, 0)
                    header_text = f"SAFE: {frame_helmet} HELMET"
                else:
                    header_color = (255, 165, 0)
                    header_text = "NO DETECTIONS"
                
                header[:] = header_color
                cv2.putText(header, header_text, (10, 35), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(header, f"Frame: {frame_count}", (width - 150, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Combine header and frame
                frame_with_header = np.vstack([header, frame])
                
                # Add footer
                footer_height = 30
                footer = np.zeros((footer_height, width, 3), dtype=np.uint8)
                footer[:] = (50, 50, 50)
                footer_text = f"Total: {frame_helmet + frame_no_helmet} | H: {frame_helmet} | NH: {frame_no_helmet}"
                cv2.putText(footer, footer_text, (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                full_frame = np.vstack([frame_with_header, footer])
                out.write(full_frame)
                frame_count += 1
                
                progress = (frame_count / total_frames) * 100
                with processing_lock:
                    video_results[video_id]['progress'] = progress
            
            cap.release()
            out.release()
            
            # Compress video
            compressed_path = compress_video(output_path)
            
            with processing_lock:
                video_results[video_id]['status'] = 'completed'
                video_results[video_id]['processed_video_path'] = compressed_path
                video_results[video_id]['summary'] = {
                    'total_frames': frame_count,
                    'helmet_count': helmet_total,
                    'no_helmet_count': no_helmet_total
                }
            
            os.remove(video_path)
            os.remove(output_path)
            
            video_processing_queue.task_done()
            
        except Exception as e:
            print(f"Error processing video: {e}")
            with processing_lock:
                if video_id in video_results:
                    video_results[video_id]['status'] = 'error'
                    video_results[video_id]['error'] = str(e)

def compress_video(input_path):
    """Compress video for web playback"""
    output_path = input_path.replace('.mp4', '_compressed.mp4')
    try:
        video = VideoFileClip(input_path)
        video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        video.close()
        return output_path
    except:
        return input_path

# Start video worker
video_worker = threading.Thread(target=process_video_worker, daemon=True)
video_worker.start()

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            if is_video_file(filename):
                video_id = str(uuid.uuid4())
                video_processing_queue.put((video_id, filepath))
                
                return jsonify({
                    "success": True,
                    "type": "video",
                    "video_id": video_id,
                    "message": "Video uploaded for processing"
                })
            else:
                result, processed_path = process_image(filepath)
                
                if "error" in result:
                    return jsonify(result), 500
                
                with open(processed_path, 'rb') as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                os.remove(filepath)
                if processed_path != filepath:
                    os.remove(processed_path)
                
                return jsonify({
                    "success": True,
                    "type": "image",
                    **result,
                    "image": img_base64
                })
        
        return jsonify({"error": "Invalid file type"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/status/<video_id>', methods=['GET'])
def get_video_status(video_id):
    """Get video processing status"""
    with processing_lock:
        if video_id not in video_results:
            return jsonify({"error": "Video not found"}), 404
        
        result = video_results[video_id].copy()
        
        if result['status'] == 'completed' and result.get('processed_video_path'):
            try:
                with open(result['processed_video_path'], 'rb') as f:
                    video_data = f.read()
                    video_base64 = base64.b64encode(video_data).decode('utf-8')
                
                result['video_data'] = video_base64
                
                os.remove(result['processed_video_path'])
                result['processed_video_path'] = None
                
            except Exception as e:
                result['error'] = f"Error reading video: {str(e)}"
        
        return jsonify(result)

@app.route('/api/video/stream/<video_id>')
def stream_video(video_id):
    """Stream processed video"""
    with processing_lock:
        if video_id not in video_results or video_results[video_id]['status'] != 'completed':
            return jsonify({"error": "Video not ready"}), 404
        
        video_path = video_results[video_id].get('processed_video_path')
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 404
    
    def generate():
        with open(video_path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        
        os.remove(video_path)
        with processing_lock:
            if video_id in video_results:
                del video_results[video_id]
    
    return Response(generate(), mimetype='video/mp4')

@app.route('/api/webcam/frame', methods=['POST'])
def process_webcam_frame():
    """Process webcam frame"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"}), 400
        
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        results = model(img, conf=0.25)
        
        detections = []
        helmet_count = 0
        no_helmet_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidence),
                        "class": class_name,
                        "class_id": class_id
                    })
                    
                    if class_name == "Helmet":
                        helmet_count += 1
                    else:
                        no_helmet_count += 1
        
        return jsonify({
            "success": True,
            "detections": detections,
            "helmet_count": helmet_count,
            "no_helmet_count": no_helmet_count
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    return send_from_directory('../frontend', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)