import cv2
import numpy as np
from djitellopy import Tello  # Example SDK for DJI Tello drone

# Initialize Drone
drone = Tello()
drone.connect()
drone.streamon()

# Function for Object Detection (e.g., detecting red objects)
def detect_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define range for red color detection
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return x, y, w, h, mask
    return None, None, None, None, mask

# Main Loop
try:
    while True:
        # Get the frame from the drone's camera
        frame = drone.get_frame_read().frame
        frame = cv2.resize(frame, (640, 480))
        
        # Detect object
        x, y, w, h, mask = detect_object(frame)
        
        if x is not None:
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(f"Object detected at: x={x}, y={y}, w={w}, h={h}")
            
            # Autonomous commands (example: center the drone on the object)
            if x + w/2 < 320:
                drone.move_left(20)
            elif x + w/2 > 320:
                drone.move_right(20)
            else:
                print("Object centered")

        # Display the frame
        cv2.imshow("Drone Camera", frame)
        cv2.imshow("Mask", mask)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Safe landing and cleanup
    drone.land()
    drone.streamoff()
    cv2.destroyAllWindow()
