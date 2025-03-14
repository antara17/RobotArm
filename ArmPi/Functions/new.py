#!/usr/bin/python3
# coding=utf8
import sys
import cv2
import numpy as np
import time
import math

sys.path.append('/home/pi/ArmPi/')

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
                    # Get ROI from the box (here simply using the box as a placeholder)
                    roi = box
                    # Use the center from the rectangle as a placeholder.
                    center = rect[0]
                    img_centerx, img_centery = int(center[0]), int(center[1])
                    # Convert image coordinates to world coordinates.
                    # This conversion factor is an example; adjust as needed.
                    scale_factor = 0.1  
                    world_x = round(img_centerx * scale_factor, 2)
                    world_y = round(img_centery * scale_factor, 2)
                    
                    # Draw the contour and label using the corresponding color
                    cv2.drawContours(annotated_img, [box], -1, self.range_rgb[color], 2)
                    cv2.putText(annotated_img, f'({world_x},{world_y})',
                                (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.range_rgb[color], 1)
                    
                    # Add detection to the list: (color, world_x, world_y)
                    detections.append((color, world_x, world_y))
                    # Update last seen coordinates (optional)
                    self.last_x, self.last_y = world_x, world_y
        return annotated_img, detections

class Perception:
    def __init__(self):
        # Initialize detection attributes with defaults
        self.current_detected_color = "None"
        self.object_x = 0
        self.object_y = 0
        self.rotation_angle = 0  # New version does not compute rotation; set to 0 or modify if needed
        
        # Initialize BlockDetector for red, green, and blue blocks
        self.detector = BlockDetector(target_color=('red', 'green', 'blue'))
        
        # Initialize the camera
        self.camera = Camera.Camera()
        self.camera.camera_open()

    def find_objects(self):
        """Continuously capture frames, detect blocks, and update detection attributes."""
        while True:
            frame = self.camera.frame
            if frame is not None:
                annotated_frame, detections = self.detector.detect_block(frame)
                cv2.imshow('Block Detection', annotated_frame)
                
                if detections:
                    # For this example, we take the first detected block.
                    detected_color, world_x, world_y = detections[0]
                    self.current_detected_color = detected_color
                    self.object_x = world_x
                    self.object_y = world_y
                    # Rotation angle is not computed in this version; set to 0.
                    self.rotation_angle = 0
                else:
                    self.current_detected_color = "None"
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC key to exit the loop
                    break
            else:
                # If no frame is available, wait briefly
                time.sleep(0.01)
                
        self.camera.camera_close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # For testing purposes, run the Perception module directly.
    perception_system = Perception()
    perception_system.find_objects()