# USB 摄像头
CAMERA_DEVICE=/dev/video1 INPUT_FORMAT=mjpeg docker compose -f docker-compose.jetson.yaml up -d  

# 板载摄像头
CAMERA_DEVICE=/dev/video0 INPUT_FORMAT=mjpeg docker compose -f docker-compose.jetson.yaml up -d  