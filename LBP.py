# lbp for ear indentification

import cv2
import numpy as np
import os

from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import torch

import argparse

# P = number of circularly symmetric neighbour set points (quantization of the angular space)
# R = radius of circle


def extract_lbp_features(images, filenames, P, R, method="default"):
    lbp_features = []
    print("Extracting LBP features for", len(images), "images")
    for i, img in enumerate(images):
        lbp = local_binary_pattern(
            img, P, R, method=method
        )  # take the worst method so i can improve it

        # Calculate the featsogram with 2^P bins for standard LBP
        # n_bins = 2**P
        # (feats, _) = np.featsogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        # # Normalize the featsogram to values between 0 and 1
        # feats = feats.astype("float")
        # feats /= feats.sum() + 1e-7

        # #plot featsogram of lbp features
        # plt.feats(feats, bins=50)
        # plt.show()

        # print(len(feats))

        feats = lbp.ravel()

        lbp_features.append({"img": img, "feats": feats, "filename": filenames[i]})

    return lbp_features


def rotate_to_minimum(value, bits=8):

    #xhexk if bits afe True or False arays and if they are convert them to ints
    binary_string = "".join(['1' if bit else '0' for bit in value])
    
    #to array
    value = np.array([int(x) for x in binary_string])
    

    return min(
        int("".join(str(bit) for bit in np.roll(value, i)), 2) for i in range(bits)
    )


def is_uniform(pattern):
    return np.sum(pattern != np.roll(pattern, -1)) <= 2


def get_lbp_value(center_pixel, neighbors, method, P, R):
    binary_result = neighbors >= center_pixel
    if method == "uniform":
        return (
            np.packbits(binary_result.astype(int))[0]
            if is_uniform(binary_result)
            else P + 1
        )
    elif method == "rotation":
        return rotate_to_minimum(binary_result, bits=P)
    else:  # default method
        return np.packbits(binary_result.astype(int))[0]


def circular_lbp(img, P, R, method="default"):
    rows, cols = img.shape
    lbp_image = np.zeros((rows - 2 * R, cols - 2 * R), dtype=np.uint8)
    for r in range(R, rows - R):
        for c in range(R, cols - R):
            center_pixel = img[r, c]
            neighbors = [
                img[
                    r + int(R * np.sin(2 * np.pi * p / P)),
                    c + int(R * np.cos(2 * np.pi * p / P)),
                ]
                for p in range(P)
            ]
            neighbors = np.array(neighbors, dtype=center_pixel.dtype)
            lbp_image[r - R, c - R] = get_lbp_value(
                center_pixel, neighbors, method, P, R
            )
    return lbp_image


def compute_histogram(lbp_image, method, P, R):
    if method == "uniform":
        # For uniform patterns, the number of bins is P + 2
        # P bins for uniform patterns and 1 for non-uniform patterns, 1 for the background
        hist = np.histogram(lbp_image.ravel(), bins=P + 2, range=(0, P + 2))[0]
    else:
        # For non-uniform patterns, the number of bins is 2^P
        hist = np.histogram(lbp_image.ravel(), bins=2**P, range=(0, 2**P))[0]
    return hist


def extract_lbp_features_mine(
    images, filenames, P, R, method="default", save=True, histograms=False
):
    lbp_features = []
    print("Extracting LBP features for", len(images), "images")
    if save:
        os.makedirs("lbp_features", exist_ok=True)
    for i, img in enumerate(images):
        lbp_image = circular_lbp(img, P, R, method)
        if save:
            cv2.imwrite(os.path.join("lbp_features", filenames[i]), lbp_image)
        features = lbp_image.ravel()
        if histograms:
            features = compute_histogram(lbp_image, method, P, R)
        lbp_features.append({"img": img, "feats": features, "filename": filenames[i]})
    return lbp_features


def parse_args():
    """
    Parses command line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="LBP")
    parser.add_argument(
        "--dataset", help="Path to ear images", default="./gt_detections", type=str
    )
    parser.add_argument(
        "--lbptype",
        help="LBP type",
        default="default",
        type=str,  # choices=['default', 'uniform', 'rotation', 'lib']
    )

    parser.add_argument("--radius", help="Radius", default=1, type=int)
    parser.add_argument("--neighbors", help="Neighbors", default=8, type=int)
    parser.add_argument("--resize_to", help="Resize to", default=128, type=int)

    parser.add_argument("--use_lib", action='store_true', help="Use library for LBP")
    parser.add_argument("--histograms", action='store_true', help="Use histograms")
    parser.add_argument("--save", action='store_true', help="Save the result")
    parser.add_argument("--use_raw", action='store_true', help="Use raw comparison")

    return parser.parse_args()


def raw_image_to_lbp(images, filenames):
    # convert images to vectors
    res = []
    print("Extracting LBP features for", len(images), "images")

    for i, img in enumerate(images):
        img_f = img.flatten()
        res.append({"img": img, "filename": filenames[i], "feats": img_f})

    return res


# load dataset of images
def main():
    args = parse_args()

    dataset_path = args.dataset
    dataset = os.listdir(dataset_path)
    # keep png files
    dataset = [f for f in dataset if f[-3:] == "png"]
    sorted(dataset)

    print("Dataset size:", len(dataset))
    # print(dataset)
    # file_names = [f for f in dataset]

    # # convert to grayscale
    # dataset = [
    #     cv2.imread(os.path.join(dataset_path, f), cv2.IMREAD_GRAYSCALE) for f in dataset
    # ]

    # # rescale images to 128x128
    # dataset = [cv2.resize(img, (args.resize_to, args.resize_to)) for img in dataset]

    # # lbp_features = extract_lbp_features_simple(dataset, file_names, P, R)
    # lbp_features = extract_lbp_features(dataset, file_names, P, R)

    identities = []
    with open("identities.txt", "r") as f:
        for line in f:
            # 0501.png 001
            file, identity = line.split()

            # lbp = [i["feats"] for i in lbp_features if i["filename"] == file]

            img= cv2.imread(os.path.join(dataset_path, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (args.resize_to, args.resize_to))


            identities.append({"file": file, "identity": identity, "feats": None, "img": img})

    # get how many unique identities we have
    print(f"Found {len(set([i['identity'] for i in identities]))} unique identities")
    print(
        f"Found {len(identities)/len(set([i['identity'] for i in identities]))} images per identity"
    )

    test_identities = [
        "008",
        "004",
        "043",
        "022",
        "035",
        "034",
        "014",
        "028",
        "042",
        "025",
    ]
    print(f"Test identities: {test_identities}")

    test_set = []
    for identity in test_identities:
        test_set += [i for i in identities if i["identity"] == identity]

    print(f"Test set: len: {len(test_set)}")
    # print(test_set[0])


    #trst set is all the images that are not in the test set
    train_set = [i for i in identities if i["file"] not in [i["file"] for i in test_set]]


    print(f"Train set: len: {len(train_set)}")
    # print(train_set[0])

    train_set_images = [i["img"] for i in train_set]
    train_set_filenames = [i["file"] for i in train_set]

    # Extract LBP features
    P = args.neighbors  # Number of circularly symmetric neighbour set points
    R = args.radius  # Radius of circle
    histograms = args.histograms
    method = args.lbptype

    lbp_features = None
    if args.use_raw == False:
        if args.use_lib:

            if method == "default":
                method = "default"
            elif method == "uniform":
                method = "uniform"
            elif method == "rotation":
                method = "ror"

            lbp_features = extract_lbp_features(train_set_images, train_set_filenames, P, R, method)

        else:
            lbp_features = extract_lbp_features_mine(
                train_set_images, train_set_filenames, P, R, method, histograms=histograms
            )
    else:
        lbp_features = raw_image_to_lbp(train_set_images, train_set_filenames)

    for sample in train_set:
        feats = [i["feats"] for i in lbp_features if i["filename"] == sample["file"]]
        if not len(feats) == 1:
            print("Error")
            exit()

        sample["feats"] = feats[0]

    # Convert featsograms to a NumPy array for efficient computation
    features_array = np.array([i["feats"] for i in train_set])
    # remove inner dimension
    features_array = np.squeeze(features_array)

    print(f"Features array shape: {features_array.shape}")

    # Calculate the cosine similarity matrix (efficient vectorized operation)
    similarity_matrix = cosine_similarity(features_array, features_array)

    # Fill the diagonal with zeros to ignore self-similarity
    np.fill_diagonal(similarity_matrix, 0)

    # Initialize variables to track the closest matches and accuracy
    closest_matches = np.argmax(similarity_matrix, axis=1)
    correct_matches = 0

    # Use tqdm to show progress for checking matches
    for i in range(len(closest_matches)):
        # Get the index of the closest match
        closest_index = closest_matches[i]
        # Check if the closest match has the same identity
        if train_set[i]["identity"] == train_set[closest_index]["identity"]:
            correct_matches += 1
            print(
                f"Correct match: {train_set[i]['file']} with {train_set[closest_index]['file']}"
            )

    # Calculate accuracy based on the number of correct matches
    accuracy = correct_matches / len(train_set)


    print("----------------------------------------")
    print(f"Accuracy: {accuracy}")

    # Save result matrix to txt
    np.savetxt("result_matrix.txt", similarity_matrix, fmt="%.5f")


if __name__ == "__main__":
    main()
