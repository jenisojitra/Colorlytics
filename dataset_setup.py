import cv2
import numpy as np
from tkinter import Tk, filedialog

# Hide the main Tkinter window
Tk().withdraw()

# 1. Select the image file
print("Please select the metal defect image...")
file_path = filedialog.askopenfilename(
    title="Select Metal Defect Image",
    filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
)

if not file_path:
    print("No file selected. Exiting.")
    exit()

# 2. Load the image
image = cv2.imread(file_path)
if image is None:
    print("Error: Could not load image.")
    exit()

# 3. Convert to Grayscale & Blur to smooth out normal metal grain
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Canny Edge Detection
# Adjust 50 and 150 thresholds if it catches too much noise or misses cracks
edges = cv2.Canny(blurred, 50, 150)

# 5. Dilate and Erode (Closing) to connect broken edge lines of the cracks
kernel = np.ones((3, 3), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 6. Find Contours (the outlines of the cracks)
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 7. Filter out tiny speckles/noise and draw the defects
output_image = image.copy()
defect_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    # Ignore tiny noise spots (adjust the minimum area limit if needed)
    if area > 20: 
        defect_count += 1
        # Draw a red bounding box around the detected defect
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 8. Determine Status
status = "Defective Steel" if defect_count > 0 else "Normal Steel"
print(f"Detected Defective Regions: {defect_count}")
print(f"Status: {status}")

# Annotate screen
cv2.putText(output_image, f"Defects Found: {defect_count} ({status})", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 9. Display the processing stages
cv2.imshow("Original Image", image)
cv2.imshow("Edge Map (What the AI sees)", closed_edges)
cv2.imshow("Final Defect Detection", output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()