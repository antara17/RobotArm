#!/usr/bin/python3
# coding=utf8
import sys
import cv2
import numpy as np
import math
import time
sys.path.append('/home/pi/ArmPi/')
import Camera
from LABConfig import color_range
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
from CameraCalibration.CalibrationConfig import square_length
import HiwonderSDK.Board as Board
import json

from vosk import Model, KaldiRecognizer
import sounddevice as sd
import threading

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
        # For backwards compatibility, we keep these but now use a list of detections.
        self.detected_contour = None
        self.detected_contour_area = 0
        self.detected_color = None
        self.min_contour_area = 2500
        
        # New list to hold all detections (each as a tuple: (color, contour, area))
        self.detections = []
        
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

        # NEW: a variable to filter detections by requested color.
        self.target_color_filter = None

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

        # Find objects of interest for all target colors
        self.detect_color_objects(lab_image)

        # Default: no detection updated this frame.
        detection_updated = False
        
        # If a target color filter is set, try to update detection info only for that color.
        if self.target_color_filter is not None:
            for (color, contour, area) in self.detections:
                if color == self.target_color_filter and area > self.min_contour_area:
                    rect = cv2.minAreaRect(contour)
                    box = np.int0(cv2.boxPoints(rect))
                    self.region_of_interest = getROI(box)
                    img_x, img_y = getCenter(rect, self.region_of_interest, self.image_dimensions, square_length)
                    world_x, world_y = convertCoordinate(img_x, img_y, self.image_dimensions)
                    
                    # Draw contour and coordinates using the color's display value
                    cv2.drawContours(frame, [box], -1, self.color_display_values[color], 2)
                    cv2.putText(frame, f'({world_x}, {world_y})', 
                                (min(box[0, 0], box[2, 0]), box[2, 1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                self.color_display_values[color], 1)
                    
                    # Update detection info for the requested color
                    self.detected_contour = contour
                    self.detected_color = color
                    self.detected_contour_area = area
                    distance = math.sqrt((world_x - self.object_x)**2 + (world_y - self.object_y)**2)
                    self.object_x, self.object_y = world_x, world_y
                    self.color_history.append(self.color_code_map.get(color, 0))
                    if distance < self.movement_threshold:
                        self.position_history.extend((world_x, world_y))
                        self.check_object_stability(rect)
                    else:
                        self.last_detection_time = time.time()
                        self.position_history = []
                    
                    self.current_detected_color = color
                    self.display_color = self.color_display_values[color]
                    detection_updated = True
                    break  # use the first matching detection

        # If no target filter is set or if no matching detection found, fallback to consensus (or none)
        if not detection_updated:
            if self.color_history and len(self.color_history) >= 3:
                average_color_code = int(round(np.mean(np.array(self.color_history))))
                if average_color_code in self.color_decode_map:
                    self.current_detected_color = self.color_decode_map[average_color_code]
                    self.display_color = self.color_display_values[self.current_detected_color]
                else:
                    self.current_detected_color = 'None'
                    self.display_color = self.color_display_values['black']
                self.color_history = []
            else:
                self.current_detected_color = "None"
                self.display_color = self.color_display_values['black']
        
        # Display current detected color
        cv2.putText(frame, f'Color: {self.current_detected_color}', 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.65, self.display_color, 2)
        
        # Reset detections list and temporary single-detection variables for next frame
        self.detections = []
        self.detected_contour = None
        self.detected_contour_area = 0
        self.detected_color = None
        
        return frame
            
    def check_object_stability(self, rect):
        """Check if object has been stable for the required time"""
        if time.time() - self.last_detection_time > self.stability_time_required:
            self.rotation_angle = rect[2]
            self.position_history = []
            self.last_detection_time = time.time()

    def detect_color_objects(self, lab_image):
        """Detect objects of target colors and store them in self.detections"""
        self.detections = []
        for color in self.color_range:
            if color in self.target_colors:
                color_mask = cv2.inRange(lab_image, 
                                         self.color_range[color][0], 
                                         self.color_range[color][1])
                opened_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, 
                                               np.ones(self.filter_kernel_size, np.uint8))
                cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, 
                                                np.ones(self.filter_kernel_size, np.uint8))
                contours = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_NONE)[-2]
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > self.min_contour_area:
                        self.detections.append((color, cnt, area))

class Motion:
    def __init__(self, perception):
        # Destination coordinates for sorted blocks (x, y, z)
        self.destination_positions = {
            'red': (-14.5, 11.5, 1.5),
            'green': (-14.5, 5.5, 1.5),
            'blue': (-14.5, -0.5, 1.5)
        }
        
        # Reference to the perception system
        self.perception = perception
        
        # Arm control parameters
        self.arm_controller = ArmIK()
        
        # Timing parameters
        self.movement_speed_divisor = 1000
        self.pause_duration = 0.5
        
        # Servo parameters
        self.gripper_servo_id = 1
        self.wrist_servo_id = 2
        self.gripper_closed_position = 500
        self.gripper_open_position = 280
        
        # Motion planning parameters
        self.approach_height = 7
        self.grasp_height = 1.0

    def set_indicator_leds(self, color):
        """Set the LED indicators to match the detected block color"""
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "None": (0, 0, 0)
        }
        rgb_values = color_map.get(color, (0, 0, 0))
        Board.RGB.setPixelColor(0, Board.PixelColor(*rgb_values))
        Board.RGB.setPixelColor(1, Board.PixelColor(*rgb_values))
        Board.RGB.show()

    def move_to_ready_position(self):
        """Move the arm to a neutral ready position"""
        # Set gripper to partially open
        Board.setBusServoPulse(self.gripper_servo_id, 
                               self.gripper_closed_position - 50, 
                               self.gripper_open_position)
        
        # Reset wrist orientation
        Board.setBusServoPulse(self.wrist_servo_id, 
                               self.gripper_closed_position, 
                               self.gripper_closed_position)
        
        # Move to home coordinates with -30Â° orientation
        self.arm_controller.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)
        time.sleep(self.pause_duration)

    def execute_cycle(self, requested_color):
        """Execute one complete pick-and-place cycle for the requested block color."""
        # Use the requested color for the cycle
        block_color = requested_color
        self.set_indicator_leds(block_color)
        
        # Use the perception's detected coordinates
        target_x = self.perception.object_x
        target_y = self.perception.object_y
        target_orientation = self.perception.rotation_angle
        
        # STEP 1: Move to position above block
        move_result = self.arm_controller.setPitchRangeMoving(
            (target_x, target_y, self.approach_height), 
            -90, -90, 0
        )  
        if move_result:
            time.sleep(move_result[2] / self.movement_speed_divisor)
            
            # STEP 2: Prepare gripper orientation and open it
            gripper_angle = getAngle(target_x, target_y, target_orientation)
            Board.setBusServoPulse(
                self.gripper_servo_id, 
                self.gripper_closed_position - self.gripper_open_position, 
                self.gripper_closed_position
            )
            Board.setBusServoPulse(
                self.wrist_servo_id, 
                gripper_angle, 
                self.gripper_closed_position
            )
            time.sleep(self.pause_duration)
            
            # STEP 3: Lower arm to grasp position
            self.arm_controller.setPitchRangeMoving(
                (target_x, target_y, self.grasp_height), 
                -90, -90, 0, 1000
            )
            time.sleep(self.pause_duration)
            
            # STEP 4: Close gripper to grab block
            Board.setBusServoPulse(
                self.gripper_servo_id, 
                self.gripper_closed_position, 
                self.gripper_closed_position
            )
            time.sleep(self.pause_duration)
            
            # STEP 5: Reset wrist rotation and lift block
            Board.setBusServoPulse(
                self.wrist_servo_id, 
                self.gripper_closed_position, 
                self.gripper_closed_position
            )
            self.arm_controller.setPitchRangeMoving(
                (target_x, target_y, self.approach_height), 
                -90, -90, 0, 1000
            )
            time.sleep(2 * self.pause_duration)
            
            # STEP 6: Move to sorting location based on requested color
            destination = self.destination_positions[block_color]
            move_result = self.arm_controller.setPitchRangeMoving(
                (destination[0], destination[1], 12), 
                -90, -90, 0
            )   
            time.sleep(move_result[2] / self.movement_speed_divisor)
                            
            # STEP 7: Adjust wrist orientation for placement
            placement_angle = getAngle(destination[0], destination[1], -90)
            Board.setBusServoPulse(
                self.wrist_servo_id, 
                placement_angle, 
                self.gripper_closed_position
            )
            time.sleep(self.pause_duration)
            
            # STEP 8: Lower arm to pre-placement position
            self.arm_controller.setPitchRangeMoving(
                (destination[0], destination[1], destination[2] + 3), 
                -90, -90, 0, 500
            )
            time.sleep(self.pause_duration)
                                    
            # STEP 9: Lower arm to final placement position
            self.arm_controller.setPitchRangeMoving(
                destination, 
                -90, -90, 0, 1000
            )
            time.sleep(self.pause_duration)
            
            # STEP 10: Open gripper to release block
            Board.setBusServoPulse(
                self.gripper_servo_id, 
                self.gripper_closed_position - self.gripper_open_position, 
                self.gripper_closed_position
            )
            time.sleep(self.pause_duration)
            
            # STEP 11: Raise arm after placement
            self.arm_controller.setPitchRangeMoving(
                (destination[0], destination[1], 12), 
                -90, -90, 0, 800
            )
            time.sleep(self.pause_duration)
            
            # STEP 12: Return to ready position
            self.move_to_ready_position()
            
            # STEP 13: Reset status for next cycle
            self.perception.current_detected_color = 'None'
            self.set_indicator_leds('None')
            time.sleep(3 * self.pause_duration)

def wait_for_pickup_command():
    """
    Listen continuously until a command like 
    'pick up red block', 'pick up green block' or 'pick up blue block' is recognized.
    Returns the detected color (as a lowercase string) if found.
    """
    model = Model("vosk-model-small-en-us-0.15")
    try:
        device_info = sd.query_devices(None, 'input')
        samplerate = int(device_info['default_samplerate'])
    except Exception as e:
        print("Error querying device:", e)
        sys.exit(1)
    
    recognizer = KaldiRecognizer(model, samplerate)
    valid_colors = ["red", "green", "blue", "read"]
    print("Listening for a pickup command (e.g., 'pick up red block')...")
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None,
                           dtype='int16', channels=1) as stream:
        while True:
            data, _ = stream.read(4000)
            data_bytes = bytes(data)
            if recognizer.AcceptWaveform(data_bytes):
                result = json.loads(recognizer.Result())
                recognized_text = result.get("text", "").lower()
                print("Heard:", recognized_text)
                # Check if the command contains "pick up" and one of the valid colors
                if "pick up" in recognized_text:
                    for color in valid_colors:
                        if color in recognized_text:
                            print(f"Command to pick up {color} block recognized.")
                            return color

if __name__ == "__main__":
    # Start the perception system in its own thread so that it continuously detects objects.
    perception_system = Perception()
    perception_thread = threading.Thread(target=perception_system.find_objects)
    perception_thread.daemon = True  # run in background
    perception_thread.start()
    
    # Initialize the motion system
    motion_system = Motion(perception_system)
    
    while True:
        requested_color = wait_for_pickup_command()
        # Set the target filter so that the perception system prioritizes this color
        perception_system.target_color_filter = requested_color
        
        # Wait for up to 5 seconds for the requested block to be detected
        timeout = 5
        start_time = time.time()
        while time.time() - start_time < timeout:
            if perception_system.current_detected_color == requested_color:
                break
            time.sleep(0.1)
        
        if perception_system.current_detected_color == requested_color:
            motion_system.execute_cycle(requested_color)
        else:
            print(f"Requested {requested_color} block not detected. Please place a {requested_color} block and try again.")
        
        # Reset the target filter for the next cycle
        perception_system.target_color_filter = None