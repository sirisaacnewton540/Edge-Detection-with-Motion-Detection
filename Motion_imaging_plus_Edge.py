import cv2
import numpy as np

print('Tasks are as follows:\n1) Motion Edge Detection\n2) Motion Imaging')
n = int(input(('Enter the task to perform: ')))

# Video Capture
cap = cv2.VideoCapture(0)

# Read the capture and get the first frame.
ret, frame1 = cap.read()

# Convert frame to gray scale.
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Initialize the motion threshold
motion_threshold = 2

# Function to update threshold from trackbar
def update_threshold(x):
    global motion_threshold
    motion_threshold = x

# Create an image with the same dimensions as the frame for later drawing purposes.
mask = np.zeros_like(frame1)
mask[..., 1] = 255  # Saturation to maximum.

# Create windows and trackbars for Motion Detection
if n == 1:
    cv2.namedWindow("Motion Detection")
    cv2.createTrackbar('Sensitivity', 'Motion Detection', motion_threshold, 50, update_threshold)

# Function to perform motion detection and highlight moving parts
def motion_detection(frame, prev_gray, threshold):
    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude and angle of the flow vectors
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold to create motion mask
    motion_mask = mag > threshold  # Use dynamic threshold
    
    # Convert motion mask to uint8 for contour detection
    motion_mask = np.uint8(motion_mask)

    # Find contours of motion
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw green contours around moving parts
    frame_with_contours = frame.copy()
    cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 2)
    
    # Check if any contours are detected
    if len(contours) > 0:
        # Display "Motion Detected" on the frame
        cv2.putText(frame_with_contours, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame_with_contours, next_gray

# Function for motion imaging (visualizing motion using HSV color space)
def motion_img(frame, prev_gray):
    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    mask[..., 0] = ang * 180 / np.pi / 4
    mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    cv2.imshow('Original', frame)
    cv2.imshow('Motion Imaging', rgb)
    
    return next_gray

# Main loop to capture frames and detect motion
while cap.isOpened():
    # Read frame from camera
    ret, frame = cap.read()
    
    if not ret:
        break

    if n == 1:
        # Get the current threshold value from trackbar
        current_threshold = cv2.getTrackbarPos('Sensitivity', 'Motion Detection')
        # Perform motion detection and highlight moving parts
        motion_frame, prev_gray = motion_detection(frame, prev_gray, current_threshold)
        cv2.imshow('Motion Detection', motion_frame)
    elif n == 2:
        # Perform motion imaging
        prev_gray = motion_img(frame, prev_gray)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
