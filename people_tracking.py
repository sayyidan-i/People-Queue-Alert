import cv2
from ultralytics import YOLO, YOLOv10

# Load the YOLOv10 model
model = YOLOv10.from_pretrained("jameslahm/yolov10s")

# Function to calculate percentage of bounding box overlap with the ROI
def calculate_overlap_percentage(person_box, roi_box):
    x1 = max(person_box[0], roi_box[0])
    y1 = max(person_box[1], roi_box[1])
    x2 = min(person_box[2], roi_box[2])
    y2 = min(person_box[3], roi_box[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate person bounding box area
    person_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])

    # Return percentage of overlap
    return (intersection_area / person_area) if person_area != 0 else 0

# Open the video file
input_video_path = 'input_video\store(1).mp4'
cap = cv2.VideoCapture(input_video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Define the codec and create VideoWriter object
output_video_path = 'output_video/tracked_output.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define the ROI (Region of Interest)
roi_x1, roi_y1, roi_x2, roi_y2 = 180, 50, 400, 300
roi_box = [roi_x1, roi_y1, roi_x2, roi_y2]

# Initialize variables for time counting
person_count_threshold = 3
alert_duration_seconds = 60  # Threshold for alert in seconds
elapsed_time = 0  # Initialize elapsed time in seconds
alert_displayed = False  # Flag to track if alert has been shown
overlap_threshold = 0.5  # Only count a person if 50% or more of their bounding box overlaps with the ROI

# Blinking variables
blink_duration_frames = 20 
frame_counter = 0

# Frame miss detection tolerance variables for alert
miss_detection_tolerance_frames = 60  # Allow up to 60 frames or 2 second of missed detections before resetting. 
missed_detection_count_alert = 0  # Count of consecutive frames where person count falls below the threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame, verbose=False)  # Directly using the frame in track

    # Count the number of people in the ROI based on overlap percentage
    person_count = 0
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy()  # Convert to NumPy array
            x1, y1, x2, y2 = map(int, xyxy[0])
            person_box = [x1, y1, x2, y2]

            # Check if the detected object is a person (usually class 0)
            if box.cls == 0:  
                # Calculate overlap percentage between person bounding box and the ROI
                overlap_percentage = calculate_overlap_percentage(person_box, roi_box)
                if overlap_percentage >= overlap_threshold:
                    person_count += 1
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Draw the bounding box
                    # Add background rectangle for label
                    label_background = (x1, y1 - 25, x1 + 60, y1)
                    cv2.rectangle(frame, (label_background[0], label_background[1]), 
                                  (label_background[2], label_background[3]), (255, 255, 0), -1)
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    print(f"Person Count: {person_count}")
    
    # Draw the ROI rectangle
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)  # Draw ROI in blue

    # Display the count of people in the ROI
    count_text = f"People Queuing: {person_count}"
    
    # Calculate text size for background
    (text_width, text_height), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (roi_x1, roi_y1 - text_height - 5), (roi_x1 + text_width, roi_y1), (0, 0, 255), -1)  # Background rectangle
    cv2.putText(frame, count_text, (roi_x1, roi_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White text on red background

    # Check for alert conditions with tolerance for missed detections
    if person_count >= person_count_threshold:
        missed_detection_count_alert = 0  # Reset missed detection count if above threshold
        # Increment elapsed time based on fps (1 frame = 1/fps seconds)
        elapsed_time += 1 / fps
    else:
        missed_detection_count_alert += 1
        if missed_detection_count_alert >= miss_detection_tolerance_frames:
            # Reset elapsed time if missed detection tolerance exceeded
            elapsed_time = 0

    # Convert elapsed_time to integer seconds for display
    elapsed_seconds = int(elapsed_time)

    # Show elapsed time below the ROI
    elapsed_time_text = f"{elapsed_seconds} sec"
    
    # Calculate text size for background
    (elapsed_text_width, elapsed_text_height), _ = cv2.getTextSize(elapsed_time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (roi_x1, roi_y2 + 20 - elapsed_text_height - 10), 
                  (roi_x1 + elapsed_text_width, roi_y2 + 30), (255, 0, 0), -1)  # Background rectangle
    cv2.putText(frame, elapsed_time_text, (roi_x1, roi_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text on red background

    # Check for alert condition
    if elapsed_seconds >= alert_duration_seconds:
        frame_counter += 1
        if frame_counter // blink_duration_frames % 2 == 0:
            alert_text = "Warning: Long Queue Detected! People Waiting for Too Long!"
            # Calculate text size for background
            (alert_text_width, alert_text_height), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            alert_x = (frame_width - alert_text_width) // 2  # Centered horizontally
            alert_y = frame_height - 30  # Positioned 30 pixels from the bottom
            cv2.rectangle(frame, (alert_x, alert_y - alert_text_height - 5), 
                        (alert_x + alert_text_width, alert_y), (0, 0, 255), -1)  # Background rectangle
            cv2.putText(frame, alert_text, (alert_x, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White alert text on red background

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
