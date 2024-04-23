import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

image = "3.JPG"
im = cv2.imread("3.JPG")
# cv2.imshow("image",im)
cv2.waitKey(0)
results = model(image, stream=True, conf=0.40)

# Display the annotated images with labeled license plates and vehicles
for res in results:
    annotated_frame = res.plot()  # Plot detection results
    cv2.imshow("Annotated Frame", annotated_frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

# Close all OpenCV windows
cv2.destroyAllWindows()
