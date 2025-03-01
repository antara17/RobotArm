#!/usr/bin/python3
# coding=utf8
import sys
import cv2
import numpy as np
import math
import time
sys.path.append('/home/pi/ArmPi/')
import Camera
from LABConfig import *
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
from CameraCalibration.CalibrationConfig import *

class Perception:
    def __init__(self):
        # Color definitions
        self.color_display_values = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255)
        }
        
        # Target colors to detect
        self.target_colors = ['red', 'green', 'blue']
        
        # Setup camera
        self.camera = Camera.Camera()
        self.camera.camera_open()
        
        # Image processing parameters
        self.image_dimensions = (640, 480)
        self.blur_kernel_size = (11, 11)
        self.filter_kernel_size = (6, 6)
        self.gaussian_std = 11
        
        # Object detection parameters
        self.region_of_interest = ()
        self.detected_contour = None
        self.detected_contour_area = 0
        self.detected_color = None
        self.min_contour_area = 2500
        
        # Position tracking
        self.object_x = 0
        self.object_y = 0
        self.color_code_map = {"red": 1, "green": 2, "blue": 3}
        self.color_decode_map = {1: "red", 2: "green", 3: "blue"}
        self.color_history = []
        self.position_history = []
        
        # Timing and state variables
        self.movement_threshold = 0.5
        self.last_detection_time = time.time()
        self.stability_time_required = 1.0
        self.current_detected_color = "None"
        self.display_color = self.color_display_values['black']
        self.rotation_angle = 0
        
        # Import color ranges from configuration
        self.color_range = color_range

    def find_objects(self):
        """Main loop for object detection"""
        while True:
            frame = self.camera.frame

            if frame is not None:
                processed_frame = self.process_frame(frame)
                cv2.imshow('Frame', processed_frame)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break
            
        self.camera.camera_close()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """Process a single camera frame to detect objects"""
        height, width = frame.shape[:2]

        # Draw calibration crosshairs
        cv2.line(frame, (0, int(height / 2)), (width, int(height / 2)), (0, 0, 200), 1)
        cv2.line(frame, (int(width / 2), 0), (int(width / 2), height), (0, 0, 200), 1)

        # Image preprocessing
        resized_image = cv2.resize(frame, self.image_dimensions, interpolation=cv2.INTER_NEAREST)
        blurred_image = cv2.GaussianBlur(resized_image, self.blur_kernel_size, self.gaussian_std)
        
        # Convert to LAB color space for better color detection
        lab_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2LAB)

        # Find objects of interest
        self.detect_color_objects(lab_image)

        # Process detected object if large enough
        if self.detected_contour_area > self.min_contour_area:
            rect = cv2.minAreaRect(self.detected_contour)
            box = np.int0(cv2.boxPoints(rect))

            self.region_of_interest = getROI(box)
            img_x, img_y = getCenter(rect, self.region_of_interest, self.image_dimensions, square_length)
            world_x, world_y = convertCoordinate(img_x, img_y, self.image_dimensions)

            # Draw contour and coordinates
            cv2.drawContours(frame, [box], -1, self.color_display_values[self.detected_color], 2)
            cv2.putText(frame, f'({world_x}, {world_y})', 
                       (min(box[0, 0], box[2, 0]), box[2, 1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.color_display_values[self.detected_color], 1) 

            # Calculate movement distance
            distance = math.sqrt((world_x - self.object_x)**2 + (world_y - self.object_y)**2)
            self.object_x, self.object_y = world_x, world_y

            # Record color detection
            if self.detected_color in self.color_code_map:
                color_code = self.color_code_map[self.detected_color]
            else:
                color_code = 0
            
            self.color_history.append(color_code)

            # Process object stability
            if distance < self.movement_threshold:
                self.position_history.extend((world_x, world_y))
                self.check_object_stability(rect)
            else:
                self.last_detection_time = time.time()
                self.position_history = []
            
            # Determine consensus color after collecting enough samples
            if len(self.color_history) == 3:
                average_color_code = int(round(np.mean(np.array(self.color_history))))
                
                if average_color_code in self.color_decode_map:
                    self.current_detected_color = self.color_decode_map[average_color_code]
                    self.display_color = self.color_display_values[self.current_detected_color]
                else:
                    self.current_detected_color = 'None'
                    self.display_color = self.color_display_values['black']
                
                self.color_history = []
        else:
            self.display_color = (0, 0, 0)
            self.current_detected_color = "None"

        # Display current detected color
        cv2.putText(frame, f'Color: {self.current_detected_color}', 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.65, self.display_color, 2)
        
        return frame
            
    def check_object_stability(self, rect):
        """Check if object has been stable for the required time"""
        if time.time() - self.last_detection_time > self.stability_time_required:
            self.rotation_angle = rect[2]
            self.position_history = []
            self.last_detection_time = time.time()

    def detect_color_objects(self, lab_image):
        """Detect objects of target colors"""
        self.detected_contour = None
        self.detected_contour_area = 0
        self.detected_color = None
        
        for color in self.color_range:
            if color in self.target_colors:
                # Create mask for current color
                color_mask = cv2.inRange(lab_image, 
                                        self.color_range[color][0], 
                                        self.color_range[color][1])
                
                # Clean up the mask
                opened_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, 
                                              np.ones(self.filter_kernel_size, np.uint8))
                cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, 
                                               np.ones(self.filter_kernel_size, np.uint8))
                
                # Find contours in the mask
                contours = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_NONE)[-2]

                try:
                    # Find largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    largest_area = cv2.contourArea(largest_contour)

                    # Update best contour if current is larger
                    if largest_area > self.detected_contour_area:
                        self.detected_contour_area = largest_area
                        self.detected_contour = largest_contour
                        self.detected_color = color
                except:
                    continue