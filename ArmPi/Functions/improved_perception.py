import cv2
import numpy as np

class BlockPerception:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        # Define color ranges in HSV (except red, which is better in LAB)
        self.color_ranges = {
            'red': {
                'space': 'LAB',
                'lower': np.array([20, 150, 100], dtype="uint8"),
                'upper': np.array([255, 200, 180], dtype="uint8")
            },
            'blue': {
                'space': 'HSV',
                'lower': np.array([90, 50, 50], dtype="uint8"),
                'upper': np.array([130, 255, 255], dtype="uint8")
            },
            'green': {
                'space': 'HSV',
                'lower': np.array([35, 50, 50], dtype="uint8"),
                'upper': np.array([85, 255, 255], dtype="uint8")
            },
            'black': {
                'space': 'HSV',
                'lower': np.array([0, 0, 0], dtype="uint8"),
                'upper': np.array([180, 255, 30], dtype="uint8")
            },
            'white': {
                'space': 'HSV',
                'lower': np.array([0, 0, 200], dtype="uint8"),
                'upper': np.array([180, 40, 255], dtype="uint8")
            }
        }

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame.")
            return None
        return frame

    #Convert the frame to the appropriate color space and apply color filtering
    def process_frame(self, frame, color_name):
        color_info = self.color_ranges[color_name]

        if color_info['space'] == 'LAB':
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        else:  # HSV
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = color_info['lower'], color_info['upper']
        mask = cv2.inRange(converted, lower, upper)
        return mask

    def detect_block(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 500:  # Ignore small objects
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)

    def label_block(self, frame, x, y, w, h, color_name):
        color_bgr = {
            'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0),
            'black': (0, 0, 0), 'white': (255, 255, 255)
        }
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color_name], 2)
        cv2.putText(frame, color_name.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_name], 2)

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    perception = BlockPerception()

    while True:
        frame = perception.capture_frame()
        if frame is None:
            continue

        for color_name in perception.color_ranges.keys():
            mask = perception.process_frame(frame, color_name)
            block = perception.detect_block(mask)

            if block:
                x, y, w, h = block
                perception.label_block(frame, x, y, w, h, color_name)
                break  # Stop processing once a block is found

        cv2.imshow("Block Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    perception.release_resources()
