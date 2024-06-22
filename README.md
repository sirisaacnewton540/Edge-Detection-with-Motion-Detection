# Edge Detection and Motion Imaging using Contouring Techniques

## Overview

This project implements a real-time edge detection and motion imaging system using contouring techniques. The system captures video from a webcam, processes the frames to detect edges of motion, and displays the results interactively. The project leverages OpenCV for computer vision tasks and provides an interactive interface for threshold adjustment in edge detection.

## Features

- Real-time edge detection with adjustable threshold
- Real-time motion imaging using optical flow
- Visually Appealing edge detection of the movinf pixels in the frame
  
## Usage

1. Run the script

2. The script will start the webcam and begin processing frames for the moving objects in the frames and highlighting the edges using green contours. Press `q` to quit the application.

## Project Structure

- `edge_motion_detection.py`: Main script for edge detection and motion imaging.

## Code Description

1. **Edge Detection:**
    - Converts frames to grayscale.
    - Applies Laplacian, SobelX, and SobelY operations.
    - Uses Canny edge detection with an adjustable threshold via a trackbar.
    - Displays the original and processed frames.

2. **Motion Imaging:**
    - Converts frames to grayscale.
    - Computes optical flow between consecutive frames using Farneback method.
    - Visualizes motion using HSV color space.
    - Displays the original and motion-imaging frames.

## Example

1. Run the script:
   ```sh
   python edge_motion_detection.py
   ```

2. Select the task:
   - Enter `1` for Edge Detection. Adjust the threshold using the trackbar that appears.
   - Enter `2` for Motion Imaging. Observe the motion visualization in real-time.

## Dependencies

- OpenCV: For computer vision tasks.
- NumPy: For numerical operations.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or issues, please open an issue or submit a pull request.

## Acknowledgements

- OpenCV library for facilitating computer vision tasks.
- The open-source community for continuous improvements and support.
