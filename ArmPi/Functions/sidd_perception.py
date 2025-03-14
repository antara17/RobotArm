#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi/')

import cv2
import numpy as np
import time
import math
import Camera
from LABConfig import color_range  # Assumes LABConfig defines color_range
from CameraCalibration.CalibrationConfig import square_length

# Original drawing color definitions
range_rgb = {
    'red':   (0, 0, 255),
    'blue':  (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

class BlockDetector:
    def __init__(self, target_color, size=(640, 480), square_length=square_length):
        """
        target_color: tuple of colors to detect, e.g. ('red', 'green', 'blue')
        """
        self.target_color = target_color  
        self.size = size
        self.square_length = square_length
        self.color_range = color_range
        self.range_rgb = range_rgb
        # For movement comparison (optional)
        self.last_x, self.last_y = 0, 0

    def preprocess(self, img):
        # Resize and apply Gaussian blur
        resized = cv2.resize(img, self.size, interpolation=cv2.INTER_NEAREST)
        blurred = cv2.GaussianBlur(resized, (11, 11), 11)
        return blurred

    def convert_color_space(self, img):
        # Convert image to LAB color space
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    def get_contours_for_color(self, img_lab, color):
        # Get the threshold values for the current color
        lower, upper = self.color_range[color]
        mask = cv2.inRange(img_lab, lower, upper)
        # Apply morphological operations: open then close to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours, mask

    def detect_block(self, img):
        """
        Processes the image to detect blocks for all target colors.
        Returns:
            - Annotated image (with contours and coordinate labels drawn)
            - A list of detected blocks, each as a tuple: (color, world_x, world_y)
        """
        annotated_img = img.copy()
        proc_img = self.preprocess(img)
        img_lab = self.convert_color_space(proc_img)
        detections = []  # list to hold detections from all colors

        # Loop over each color in the target_color tuple
        for color in self.target_color:
            contours, mask = self.get_contours_for_color(img_lab, color)
            for cnt in contours:
                area = math.fabs(cv2.contourArea(cnt))
                if area > 2500:  # Use the same area threshold as original
                    rect = cv2.minAreaRect(cnt)
                    box = np.int0(cv2.boxPoints(rect))
                    # Get ROI from the box
                    roi = self.getROI(box)
                    # Get center in image coordinates
                    img_centerx, img_centery = self.getCenter(rect, roi)
                    # Convert image coordinates to world coordinates
                    world_x, world_y = self.convertCoordinate(img_centerx, img_centery)
                    
                    # Draw the contour and label using the corresponding color
                    cv2.drawContours(annotated_img, [box], -1, self.range_rgb[color], 2)
                    cv2.putText(annotated_img, f'({world_x},{world_y})',
                                (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.range_rgb[color], 1)
                    
                    # Add detection to the list: (color, world_x, world_y)
                    detections.append((color, world_x, world_y))
                    # Update last seen coordinates (optional, if needed for motion detection)
                    self.last_x, self.last_y = world_x, world_y
        return annotated_img, detections

    def getROI(self, box):
        # Here we simply return the box as a placeholder.
        return box

    def getCenter(self, rect, roi):
        # Here we use the center from the rectangle as a placeholder.
        center = rect[0]
        return int(center[0]), int(center[1])

    def convertCoordinate(self, x, y):
        # Placeholder conversion from image coordinates to world coordinates.
        scale_factor = 0.1  # Example conversion factor; adjust as needed.
        return round(x * scale_factor, 2), round(y * scale_factor, 2)

def main():
    # Initialize the BlockDetector with target colors for simultaneous detection
    detector = BlockDetector(target_color=('red', 'green', 'blue'))
    
    my_camera = Camera.Camera()
    my_camera.camera_open()
    
    while True:
        frame = my_camera.frame
        if frame is not None:
            # Process the frame to detect blocks of all target colors
            annotated_frame, detections = detector.detect_block(frame)
            cv2.imshow('Block Detection', annotated_frame)
            
            for det in detections:
                print(f"Detected {det[0]} block at world coordinates: ({det[1]}, {det[2]})")
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to exit
                break
                
    my_camera.camera_close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()