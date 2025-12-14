// Configuration
const API_BASE_URL = 'http://localhost:5000';
let isProcessing = false;
let isWebcamActive = false;
let webcamStream = null;
let animationFrameId = null;
let currentVideoId = null;
let videoCheckInterval = null;
let fpsCounter = 0;
let lastFpsUpdate = 0;
let currentTab = 'image-tab';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const processBtn = document.getElementById('processBtn');
const uploadArea = document.getElementById('uploadArea');
const fileDisplay = document.getElementById('fileDisplay');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const previewCanvas = document.getElementById('previewCanvas');
const webcamBtn = document.getElementById('webcamBtn');
const stopBtn = document.getElementById('stopBtn');
const webcamVideo = document.getElementById('webcamVideo');
const webcamCanvas = document.getElementById('webcamCanvas');
const webcamPlaceholder = document.getElementById('webcamPlaceholder');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const detectionsBody = document.getElementById('detectionsBody');
const noDetections = document.getElementById('noDetections');
const helmetCount = document.getElementById('helmetCount');
const noHelmetCount = document.getElementById('noHelmetCount');
const avgConfidence = document.getElementById('avgConfidence');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

// Video elements
const videoInput = document.getElementById('videoInput');
const videoBrowseBtn = document.getElementById('videoBrowseBtn');
const videoUploadArea = document.getElementById('videoUploadArea');
const processVideoBtn = document.getElementById('processVideoBtn');
const videoFileDisplay = document.getElementById('videoFileDisplay');
const videoFileName = document.getElementById('videoFileName');
const videoFileSize = document.getElementById('videoFileSize');
const processedVideo = document.getElementById('processedVideo');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const processingProgress = document.getElementById('processingProgress');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');

// Tab elements
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

// Webcam stats
const fpsCounterEl = document.getElementById('fpsCounter');
const currentDetectionsEl = document.getElementById('currentDetections');
const webcamHelmetCountEl = document.getElementById('webcamHelmetCount');

// Event Listeners
document.addEventListener('DOMContentLoaded', init);
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
processBtn.addEventListener('click', processImage);
webcamBtn.addEventListener('click', toggleWebcam);
stopBtn.addEventListener('click', stopWebcam);

// Video event listeners
videoBrowseBtn.addEventListener('click', () => videoInput.click());
videoInput.addEventListener('change', handleVideoSelect);
processVideoBtn.addEventListener('click', processVideo);

// Tab switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');
        switchTab(tabId);
    });
});

// Drag and drop
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);

videoUploadArea.addEventListener('dragover', handleDragOver);
videoUploadArea.addEventListener('dragleave', handleDragLeave);
videoUploadArea.addEventListener('drop', handleVideoDrop);

// Initialize application
async function init() {
    checkServerStatus();
    setInterval(checkServerStatus, 30000);
    switchTab('image-tab');
}

// Tab switching
function switchTab(tabId) {
    tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    
    tabPanes.forEach(pane => {
        pane.classList.toggle('active', pane.id === tabId);
    });
    
    currentTab = tabId;
    
    if (tabId !== 'webcam-tab' && isWebcamActive) {
        stopWebcam();
    }
    
    if (tabId !== 'video-tab') {
        stopVideoProcessing();
    }
}

// Server status
async function checkServerStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            const data = await response.json();
            statusDot.className = 'status-dot online';
            statusText.textContent = `Server online | Model: ${data.model_loaded ? 'Loaded' : 'Not loaded'}`;
        } else {
            throw new Error('Server not responding');
        }
    } catch (error) {
        statusDot.className = 'status-dot';
        statusText.textContent = 'Server offline - Please start backend server';
    }
}

// File handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && validateFile(file, 'image')) {
        displayFileInfo(file, 'image');
        processBtn.disabled = false;
    }
}

function handleVideoSelect(e) {
    const file = e.target.files[0];
    if (file && validateFile(file, 'video')) {
        displayFileInfo(file, 'video');
        processVideoBtn.disabled = false;
    }
}

function validateFile(file, type) {
    const imageTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/gif'];
    const videoTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    const maxSize = 100 * 1024 * 1024;
    
    const validTypes = type === 'image' ? imageTypes : videoTypes;
    const errorMessage = type === 'image' 
        ? 'Please upload an image (JPG, PNG, BMP, GIF).' 
        : 'Please upload a video (MP4, AVI, MOV, MKV, WEBM).';
    
    if (!validTypes.includes(file.type)) {
        showNotification(errorMessage, 'error');
        return false;
    }
    
    if (file.size > maxSize) {
        showNotification('File too large. Maximum size is 100MB.', 'error');
        return false;
    }
    
    return true;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function displayFileInfo(file, type) {
    if (type === 'image') {
        fileDisplay.style.display = 'flex';
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        
        // Show preview placeholder
        previewPlaceholder.style.display = 'flex';
        previewCanvas.style.display = 'none';
        
        // Remove any existing displayed image
        const existingImg = document.querySelector('.preview-container img');
        if (existingImg) {
            existingImg.remove();
        }
    } else {
        videoFileDisplay.style.display = 'flex';
        videoFileName.textContent = file.name;
        videoFileSize.textContent = formatFileSize(file.size);
        
        videoPlaceholder.style.display = 'flex';
        processedVideo.style.display = 'none';
    }
}

function clearFile() {
    fileInput.value = '';
    processBtn.disabled = true;
    fileDisplay.style.display = 'none';
    previewPlaceholder.style.display = 'flex';
    previewCanvas.style.display = 'none';
    
    // Remove displayed image if exists
    const existingImg = document.querySelector('.preview-container img');
    if (existingImg) {
        existingImg.remove();
    }
    
    resetDetections();
}

function clearVideoFile() {
    videoInput.value = '';
    processVideoBtn.disabled = true;
    videoFileDisplay.style.display = 'none';
    videoPlaceholder.style.display = 'flex';
    processedVideo.style.display = 'none';
    processingProgress.style.display = 'none';
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (!file) return;
    
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');
    
    if (isImage && validateFile(file, 'image')) {
        fileInput.files = e.dataTransfer.files;
        displayFileInfo(file, 'image');
        processBtn.disabled = false;
        switchTab('image-tab');
    } else if (isVideo && validateFile(file, 'video')) {
        videoInput.files = e.dataTransfer.files;
        displayFileInfo(file, 'video');
        processVideoBtn.disabled = false;
        switchTab('video-tab');
    }
}

function handleVideoDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && validateFile(file, 'video')) {
        videoInput.files = e.dataTransfer.files;
        displayFileInfo(file, 'video');
        processVideoBtn.disabled = false;
    }
}

// Image processing - FIXED VERSION
async function processImage() {
    if (!fileInput.files[0] || isProcessing) return;
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    isProcessing = true;
    processBtn.disabled = true;
    processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    showLoading(true, 'Detecting helmets...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success && data.type === 'image') {
            // Hide placeholder
            previewPlaceholder.style.display = 'none';
            previewCanvas.style.display = 'none';
            
            // Remove any existing image
            const existingImg = document.querySelector('.preview-container img');
            if (existingImg) {
                existingImg.remove();
            }
            
            // Create and display the processed image
            const img = new Image();
            img.onload = function() {
                img.style.maxWidth = '100%';
                img.style.maxHeight = '100%';
                img.style.objectFit = 'contain';
                img.style.borderRadius = '8px';
                img.style.boxShadow = '0 4px 15px rgba(0,0,0,0.3)';
                img.alt = 'Processed image with detection labels';
                
                document.querySelector('.preview-container').appendChild(img);
            };
            
            img.src = `data:image/jpeg;base64,${data.image}`;
            
            // Update detection results
            updateDetectionResults(data);
            
            // Show status notification
            showDetectionStatus(data);
            
        } else {
            throw new Error(data.error || 'Processing failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        isProcessing = false;
        processBtn.disabled = false;
        processBtn.innerHTML = '<i class="fas fa-play"></i> Process Image';
        showLoading(false);
    }
}

// Update detection results
function updateDetectionResults(data) {
    const detections = data.detections || [];
    const helmetCountValue = data.helmet_count || 0;
    const noHelmetCountValue = data.no_helmet_count || 0;
    
    // Update counters with animation
    animateCounter(helmetCount, parseInt(helmetCount.textContent) || 0, helmetCountValue);
    animateCounter(noHelmetCount, parseInt(noHelmetCount.textContent) || 0, noHelmetCountValue);
    
    // Calculate average confidence
    const totalConfidence = detections.reduce((sum, d) => sum + d.confidence, 0);
    const avgConf = detections.length > 0 ? (totalConfidence / detections.length) * 100 : 0;
    animateCounter(avgConfidence, parseFloat(avgConfidence.textContent) || 0, avgConf, true);
    
    // Update stats card colors
    updateStatsColors(helmetCountValue, noHelmetCountValue);
    
    // Update detections table
    updateDetectionsTable(detections);
}

function animateCounter(element, start, end, isFloat = false) {
    const duration = 1000;
    const startTime = Date.now();
    const endTime = startTime + duration;
    
    function update() {
        const now = Date.now();
        const progress = Math.min((now - startTime) / duration, 1);
        const current = start + (end - start) * progress;
        
        if (isFloat) {
            element.textContent = `${current.toFixed(1)}%`;
        } else {
            element.textContent = Math.round(current);
        }
        
        if (now < endTime) {
            requestAnimationFrame(update);
        }
    }
    
    update();
}

function updateStatsColors(helmetCount, noHelmetCount) {
    const helmetCard = document.querySelector('.stat-card.helmet');
    const noHelmetCard = document.querySelector('.stat-card.no-helmet');
    const confidenceCard = document.querySelector('.stat-card.confidence');
    
    // Reset
    helmetCard.classList.remove('stat-highlight', 'stat-warning');
    noHelmetCard.classList.remove('stat-highlight', 'stat-warning');
    confidenceCard.classList.remove('stat-highlight', 'stat-warning');
    
    if (helmetCount > 0) {
        helmetCard.classList.add('stat-highlight');
    }
    
    if (noHelmetCount > 0) {
        noHelmetCard.classList.add('stat-warning');
    }
    
    if (helmetCount > 0 || noHelmetCount > 0) {
        confidenceCard.classList.add('stat-highlight');
    }
}

function updateDetectionsTable(detections) {
    if (detections.length > 0) {
        detectionsBody.innerHTML = '';
        noDetections.style.display = 'none';
        
        detections.forEach((det, index) => {
            const row = document.createElement('tr');
            const confidence = Math.round(det.confidence * 100);
            const displayClass = det.class === 'Helmet' ? 'Helmet' : 'No Helmet';
            
            row.innerHTML = `
                <td>
                    <div class="status-badge ${det.class === 'Helmet' ? 'status-helmet' : 'status-nohelmet'}">
                        <i class="fas ${det.class === 'Helmet' ? 'fa-hard-hat' : 'fa-user-slash'}"></i>
                        ${displayClass} Detected
                    </div>
                </td>
                <td class="confidence-cell">${confidence}%</td>
                <td>[${det.bbox.map(v => Math.round(v)).join(', ')}]</td>
                <td>${displayClass}</td>
            `;
            
            row.style.animationDelay = `${index * 0.1}s`;
            row.classList.add('fade-in');
            detectionsBody.appendChild(row);
        });
    } else {
        detectionsBody.innerHTML = '';
        noDetections.style.display = 'flex';
    }
}

function showDetectionStatus(data) {
    const helmetCount = data.helmet_count || 0;
    const noHelmetCount = data.no_helmet_count || 0;
    const totalDetections = data.total_detections || 0;
    
    if (noHelmetCount > 0) {
        showNotification(`⚠ ALERT: ${noHelmetCount} person(s) without helmet detected!`, 'warning');
    } else if (helmetCount > 0) {
        showNotification(`✓ SAFE: ${helmetCount} helmet(s) detected`, 'success');
    } else if (totalDetections === 0) {
        showNotification('No detections found in the image', 'info');
    }
}

function resetDetections() {
    helmetCount.textContent = '0';
    noHelmetCount.textContent = '0';
    avgConfidence.textContent = '0%';
    detectionsBody.innerHTML = '';
    noDetections.style.display = 'flex';
    
    // Reset stats colors
    const cards = document.querySelectorAll('.stat-card');
    cards.forEach(card => {
        card.classList.remove('stat-highlight', 'stat-warning');
    });
}

// Video processing
async function processVideo() {
    if (!videoInput.files[0] || isProcessing) return;
    
    const file = videoInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    isProcessing = true;
    processVideoBtn.disabled = true;
    processVideoBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    showLoading(true, 'Uploading video...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success && data.type === 'video') {
            currentVideoId = data.video_id;
            startVideoProgressCheck();
            showLoading(false);
            showNotification('Video uploaded! Processing started.', 'success');
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification(`Error: ${error.message}`, 'error');
        isProcessing = false;
        processVideoBtn.disabled = false;
        processVideoBtn.innerHTML = '<i class="fas fa-play-circle"></i> Process Video';
        showLoading(false);
    }
}

function startVideoProgressCheck() {
    if (videoCheckInterval) clearInterval(videoCheckInterval);
    
    processingProgress.style.display = 'block';
    processVideoBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    
    videoCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/video/status/${currentVideoId}`);
            const data = await response.json();
            
            if (data.status === 'completed') {
                clearInterval(videoCheckInterval);
                onVideoProcessingComplete(data);
            } else if (data.status === 'error') {
                clearInterval(videoCheckInterval);
                showNotification(`Error: ${data.error}`, 'error');
                resetVideoProcessing();
            } else {
                updateVideoProgress(data);
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }, 1000);
}

function updateVideoProgress(data) {
    const progress = data.progress || 0;
    progressBar.style.width = `${progress}%`;
    progressText.textContent = `${Math.round(progress)}%`;
}

function onVideoProcessingComplete(data) {
    isProcessing = false;
    processVideoBtn.disabled = false;
    processVideoBtn.innerHTML = '<i class="fas fa-play-circle"></i> Process Video';
    processingProgress.style.display = 'none';
    
    // Show video
    videoPlaceholder.style.display = 'none';
    processedVideo.style.display = 'block';
    processedVideo.src = `${API_BASE_URL}/api/video/stream/${currentVideoId}`;
    
    showNotification('Video processing completed!', 'success');
}

function resetVideoProcessing() {
    isProcessing = false;
    processVideoBtn.disabled = false;
    processVideoBtn.innerHTML = '<i class="fas fa-play-circle"></i> Process Video';
    processingProgress.style.display = 'none';
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
}

function stopVideoProcessing() {
    if (videoCheckInterval) {
        clearInterval(videoCheckInterval);
        videoCheckInterval = null;
    }
    currentVideoId = null;
}

// Webcam handling
async function toggleWebcam() {
    if (isWebcamActive) {
        stopWebcam();
    } else {
        await startWebcam();
    }
}

async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        
        webcamVideo.srcObject = webcamStream;
        webcamVideo.style.display = 'block';
        webcamCanvas.style.display = 'block';
        webcamPlaceholder.style.display = 'none';
        webcamBtn.disabled = true;
        stopBtn.disabled = false;
        isWebcamActive = true;
        
        lastFpsUpdate = Date.now();
        processWebcamFrame();
        
        showNotification('Webcam started!', 'success');
    } catch (error) {
        console.error('Error:', error);
        showNotification('Could not access webcam.', 'error');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    webcamVideo.style.display = 'none';
    webcamCanvas.style.display = 'none';
    webcamPlaceholder.style.display = 'flex';
    webcamBtn.disabled = false;
    stopBtn.disabled = true;
    isWebcamActive = false;
    
    fpsCounterEl.textContent = '0';
    currentDetectionsEl.textContent = '0';
    webcamHelmetCountEl.textContent = '0';
    
    showNotification('Webcam stopped.', 'info');
}

async function processWebcamFrame() {
    if (!isWebcamActive) return;
    
    const startTime = performance.now();
    const canvas = webcamCanvas;
    const ctx = canvas.getContext('2d');
    
    if (canvas.width !== webcamVideo.videoWidth || canvas.height !== webcamVideo.videoHeight) {
        canvas.width = webcamVideo.videoWidth;
        canvas.height = webcamVideo.videoHeight;
    }
    
    ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/webcam/frame`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        if (data.success) {
            drawWebcamDetections(data.detections, ctx);
            updateWebcamStats(data);
            updatePerformanceMetrics(startTime);
        }
    } catch (error) {
        console.error('Error:', error);
    }
    
    animationFrameId = requestAnimationFrame(processWebcamFrame);
}

function drawWebcamDetections(detections, ctx) {
    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const color = det.class === 'Helmet' ? '#2ecc71' : '#e74c3c';
        const label = det.class === 'Helmet' ? 'HELMET DETECTED' : 'NO HELMET DETECTED';
        const confidence = Math.round(det.confidence * 100);
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        ctx.fillStyle = color;
        ctx.font = 'bold 16px Arial';
        const text = `${label} ${confidence}%`;
        const textWidth = ctx.measureText(text).width;
        
        ctx.fillRect(x1, y1 - 30, textWidth + 15, 30);
        
        ctx.fillStyle = 'white';
        ctx.fillText(text, x1 + 8, y1 - 8);
    });
}

function updateWebcamStats(data) {
    const helmetCount = data.helmet_count || 0;
    currentDetectionsEl.textContent = data.detections.length;
    webcamHelmetCountEl.textContent = helmetCount;
}

function updatePerformanceMetrics(startTime) {
    const endTime = performance.now();
    const processingTime = endTime - startTime;
    
    fpsCounter++;
    const now = Date.now();
    
    if (now - lastFpsUpdate >= 1000) {
        const fps = Math.round(1000 / (processingTime || 1));
        fpsCounterEl.textContent = fps;
        fpsCounter = 0;
        lastFpsUpdate = now;
    }
}

// Utility functions
function showLoading(show, text = 'Processing...') {
    if (show) {
        loadingText.textContent = text;
        loadingOverlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    } else {
        loadingOverlay.classList.remove('active');
        document.body.style.overflow = 'auto';
    }
}

function showNotification(message, type = 'info') {
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    let icon = 'fa-info-circle';
    if (type === 'success') icon = 'fa-check-circle';
    if (type === 'error') icon = 'fa-exclamation-circle';
    if (type === 'warning') icon = 'fa-exclamation-triangle';
    
    notification.innerHTML = `<i class="fas ${icon}"></i><span>${message}</span>`;
    
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '15px 20px',
        borderRadius: '10px',
        color: 'white',
        fontWeight: '600',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        zIndex: '10000',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        animation: 'slideIn 0.3s ease-out'
    });
    
    const colors = {
        success: '#2ecc71',
        error: '#e74c3c',
        info: '#3498db',
        warning: '#f39c12'
    };
    notification.style.background = colors[type] || colors.info;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .fade-in {
        animation: slideIn 0.3s ease-out;
    }
`;
document.head.appendChild(style);