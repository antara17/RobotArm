#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi/')
import cv2
import time
import threading
import math
import numpy as np
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from LABConfig import *
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board
from CameraCalibration.CalibrationConfig import *
from new import Perception

# Define the motion control class for the robot arm
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
    valid_colors = ["red", "green", "blue"]
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
    
    # Main loop: wait for a pickup command, then wait for the corresponding block to be detected,
    # and finally perform one cycle of motion for that color.
    while True:
        requested_color = wait_for_pickup_command()
        # Wait for up to 5 seconds for the requested block to be detected by the perception system
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