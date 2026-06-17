import cv2 as cv
import os

# Create folder for tags
os.makedirs("data/tags", exist_ok=True)

# Select AprilTag 36h11 dictionary
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)

# List of target IDs needed for the project
target_ids = [16, 17, 18, 19, 20, 24, 25, 27]

# Generate and save each tag
for tag_id in target_ids:
    # Generate a 400x400 pixel tag image (with 1-pixel margin boundary)
    tag_img = cv.aruco.generateImageMarker(dictionary, tag_id, 400)
    
    file_path = f"data/tags/tag_36h11_{tag_id}.png"
    cv.imwrite(file_path, tag_img)
    print(f"Generated: {file_path}")

print("All tags saved successfully in 'data/tags/' folder.")
