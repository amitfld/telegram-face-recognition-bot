import face_recognition
import os

# Paths
image_folder = "images"
image1_path = os.path.join(image_folder, "image1.png")
image2_path = os.path.join(image_folder, "image2.png")
image3_path = os.path.join(image_folder, "image3.png")

# Load images
image1 = face_recognition.load_image_file(image1_path)
image2 = face_recognition.load_image_file(image2_path)
image3 = face_recognition.load_image_file(image3_path)

# Get encodings
encodings1 = face_recognition.face_encodings(image1)
encodings2 = face_recognition.face_encodings(image2)
encodings3 = face_recognition.face_encodings(image3)

# Check if faces are detected
if not encodings1 or not encodings2 or not encodings3:
    print("Face not found in one or more images.")
    exit()

# Get first face from each image
face1 = encodings1[0]
face2 = encodings2[0]
face3 = encodings3[0]

# Compare
sim_1_2 = face_recognition.face_distance([face1], face2)[0]
sim_1_3 = face_recognition.face_distance([face1], face3)[0]

print(f"Similarity of image1 to image2: {1 - sim_1_2:.4f}")
print(f"Similarity of image1 to image3: {1 - sim_1_3:.4f}")

if sim_1_2 < sim_1_3:
    print("Image1 is more similar to Image2.")
else:
    print("Image1 is more similar to Image3.")
