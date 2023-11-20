import numpy as np
import cv2
import os


# Function to compute the distance between two images
def cosine_distance(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def pix2pix_comparision(img, img2,distance_measure):
    #just compute the difference between the two images by treating them as vectors

    #flatten the images
    img = img.flatten()
    img2 = img2.flatten()

    #compute the difference
    return distance_measure(img, img2)


# Function to perform LBP operation on a single pixel
def lbp_operation(pixel_values):
    center = pixel_values[1, 1]
    binary = (pixel_values >= center).astype(int)
    binary_vector = binary.flatten()
    # Remove the center value
    binary_vector = np.delete(binary_vector, 4)
    return int("".join(str(x) for x in binary_vector), 2)

# Function to perform LBP on an image
def local_binary_pattern(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Padding image to handle border pixels
    padded_img = np.pad(gray, (1, 1), mode='constant', constant_values=0)
    lbp_img = np.zeros_like(gray)
    
    # Iterate through each pixel in the image
    for i in range(1, padded_img.shape[0] - 1):
        for j in range(1, padded_img.shape[1] - 1):
            lbp_img[i-1, j-1] = lbp_operation(padded_img[i-1:i+2, j-1:j+2])
    
    return lbp_img

# Function to load images, convert them to grayscale, and apply LBP
def load_and_apply_lbp(dataset_path):
    lbp_images = []
    for img_file in os.listdir(dataset_path)[:10]:
        if img_file.endswith('.jpg'):
            img_path = os.path.join(dataset_path, img_file)
            img = cv2.imread(img_path)
            lbp_img = local_binary_pattern(img)
            lbp_images.append(lbp_img)
    return lbp_images

# Mocking the dataset path
dataset_path = "./gt_detections" # This path is for illustration purposes

# Load and apply LBP to all images in the dataset
lbp_images = load_and_apply_lbp(dataset_path)

#show lbp images
for img in lbp_images:
    cv2.imshow("lbp", img)
    print(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

# Display the LBP of the first image
lbp_images[0] if lbp_images else "No images found or LBP operation failed"
