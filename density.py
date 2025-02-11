import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def detect_vehicles(image, model):
    results = model(image)
    return results

def detect_divider(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Assuming the divider is a vertical line on the left side of the image
            if abs(x1 - x2) < 20:  # Vertical line condition
                return x1  # Return the x-coordinate of the divider
    
    return None  # Return None if no divider found

def filter_vehicles_by_lane(boxes, divider_x):
    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if x1 > divider_x:  # Check if the vehicle is on the right side of the divider
            filtered_boxes.append(box)
    return filtered_boxes

def draw_lane_bounding_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return image

# Load YOLO model
model = YOLO('yolov8n.pt')

# Load the input image
image_path = 'D:\\python project\\car project\\Density of Car\\IMG-20240821-WA0080.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print(f"Failed to load image from {image_path}")
else:
    print(f"Image loaded successfully from {image_path}")
    
    # Step 1: Detect vehicles
    results = detect_vehicles(image, model)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    # Step 2: Detect the divider
    divider_x = detect_divider(image)

    if divider_x is not None:
        # Step 3: Filter vehicles by lane
        filtered_boxes = filter_vehicles_by_lane(boxes, divider_x)

        # Step 4: Draw bounding boxes for the filtered vehicles
        image_with_lanes = draw_lane_bounding_boxes(image, filtered_boxes)
        
        # Save the result to a file
        output_path = 'D:\\python project\\car project\\Density of Car\\IMG-20240821-WA0080.jpg'
        cv2.imwrite(output_path, image_with_lanes)
        
        # Display the result using matplotlib
        image_rgb = cv2.cvtColor(image_with_lanes, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()
    else:
        print("Divider not detected.")
