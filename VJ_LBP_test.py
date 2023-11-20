import cv2
import os
import torch
from torchmetrics.detection import IntersectionOverUnion
import argparse
import random

from LBP import *
import pandas as pd
import tqdm


def select_largest_detection(left_ears, right_ears):
    """
    Selects the largest detection from the left and right ear detections.

    :param left_ears: A list of bounding boxes for left ear detections
    :param right_ears: A list of bounding boxes for right ear detections
    :return: The largest bounding box and the side ('left' or 'right') it belongs to
    """
    largest_bbox = None
    largest_area = 0
    side = None

    # Check left ears
    for bbox in left_ears:
        x, y, w, h = bbox
        area = w * h
        if area > largest_area:
            largest_bbox = bbox
            largest_area = area
            side = "left"

    # Check right ears
    for bbox in right_ears:
        x, y, w, h = bbox
        area = w * h
        if area > largest_area:
            largest_bbox = bbox
            largest_area = area
            side = "right"

    return largest_bbox, side


def convert_to_yolo_format(bbox, image_dims):
    """
    Converts bounding box to YOLO format.

    :param bbox: Tuple (x, y, w, h) as provided by detectMultiScale
    :param image_dims: Tuple (image_width, image_height)
    :return: String in YOLO format
    """
    x_center = (bbox[0] + bbox[2] / 2) / image_dims[0]
    y_center = (bbox[1] + bbox[3] / 2) / image_dims[1]
    w_norm = bbox[2] / image_dims[0]
    h_norm = bbox[3] / image_dims[1]

    return x_center, y_center, w_norm, h_norm


def read_ground_truth(file_path):
    """
    Reads the ground truth bounding boxes from a file.

    :param file_path: Path to the ground truth .txt file
    :return:tuple representing ground truth bounding box
    """
    ground_truths = []
    with open(file_path, "r") as f:
        for line in f:
            clas, x_center, y_center, width, height = [float(n) for n in line.split()]
            ground_truths.append((clas, x_center, y_center, width, height))
            # only one gt per image
            break

    return ground_truths[0]


def parse_args():
    """
    Parses command line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Detect ears in images")
    parser.add_argument("--dataset", help="path to ear images", default="./ears")
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.02,
        help="Scale factor for ear detection",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=2,
        help="Minimum number of neighbors for ear detection",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=30,
        help="Minimum height size of ear for ear detection",
    )
    parser.add_argument("--save", action="store_true", help="Save the result")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the image

    # get images from dir
    files = os.listdir(args.dataset)
    # use only png files
    files = [f for f in files if f[-3:] == "png"]

    # sort
    files.sort()

    # print(files)

    # print(f"Found {len(files)} images")

    # open identities file and read the identities
    identities = []
    with open("identities.txt", "r") as f:
        for line in f:
            # 0501.png 001
            file, identity = line.split()
            identities.append({"file": file, "identity": identity})

    # get how many unique identities we have
    # print(f"Found {len(set([i['identity'] for i in identities]))} unique identities")

    # #get how many images we have per identity
    # print(f"Found {len(identities)/len(set([i['identity'] for i in identities]))} images per identity")

    # #shuffle the identities and chose 10 for test set
    # unique_identities = list(set([i['identity'] for i in identities]))
    # random.seed(42)
    # test_identities = random.sample(unique_identities, 10)

    # print(f"Test identities: {test_identities}")

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
    # print(f"Test identities: {test_identities}")

    test_set = []
    for identity in test_identities:
        test_set += [i["file"] for i in identities if i["identity"] == identity]

    print(f"Test set: len: {len(test_set)}")

    train_set = [i["file"] for i in identities if i["file"] not in test_set]
    print(f"Train set: len: {len(train_set)}")

    experiment = []
    for identity in test_identities:
        experiment += [i for i in identities if i["identity"] == identity]

    # add gt_bbox and detected_bbox to experiment
    for i in experiment:
        i["gt_bbox"] = None
        i["detected_bbox"] = None
        i["gt_img"] = None
        i["detected_img"] = None

    ious = []
    ioius_of_detected = []

    scale_factor = args.scale_factor
    min_neighbors = args.min_neighbors

    # IMPORTANT We have Yolo format which is cx,cy,w,h also we dont care about the class
    iou_metric = IntersectionOverUnion(
        class_metrics=True, box_format="cxcywh", respect_labels=False
    )

    # Load pre-trained ear classifiers
    left_ear_cascade = cv2.CascadeClassifier("haarcascade_mcs_leftear.xml")
    right_ear_cascade = cv2.CascadeClassifier("haarcascade_mcs_rightear.xml")
    print(
        f"Starting detection with scale_factor: {scale_factor}, min_neighbors: {min_neighbors}, min_size: {args.min_size}"
    )
    # DO hyperparameter search only on train set
    for i, sample in enumerate(tqdm.tqdm(experiment)):
        file = sample["file"]
        image_path = os.path.join(args.dataset, file)
        src = cv2.imread(image_path)
        src_copy = src.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        min_size_for_ear = args.min_size
        min_size_width_for_ear = int(min_size_for_ear * 0.61)
        # Based of Average ratio of images: .61

        # Detect left ear
        left_ears = left_ear_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size_width_for_ear, min_size_for_ear),
        )
      
        # Detect right ear
        right_ears = right_ear_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size_width_for_ear, min_size_for_ear),
        )
       
        # save in Yolo format

        src_width = src.shape[1]
        src_height = src.shape[0]

        # only one gt per image so we can select the largest detection from the left and right ear detections
        largest_bbox, side = select_largest_detection(left_ears, right_ears)

       
        gt = read_ground_truth(os.path.join(args.dataset, file[:-4] + ".txt"))

        # 0 is left ear, 1 is right ear
        clas, x_center, y_center, width, height = gt
        gt_bbox = (x_center, y_center, width, height)

        experiment[i]["gt_bbox"] = gt_bbox
        experiment[i]["detected_bbox"] = largest_bbox


        x, y, w, h = gt_bbox
        x = int((x - w / 2) * src_width)
        y = int((y - h / 2) * src_height)
        w = int(w * src_width)
        h = int(h * src_height)

        # save gt detection
        cv2.imwrite(
            f"gt_test_detections/{file[:-4]}_gt.png", src_copy[y : y + h, x : x + w]
        )

        experiment[i]["gt_img"] = f"gt_test_detections/{file[:-4]}_gt.png"

        iou = 0.0
        if not largest_bbox is None:
            # print(f"gt: {gt_bbox}")
            x, y, w, h = largest_bbox
            largest_bbox_yolo = convert_to_yolo_format(
                (x, y, w, h), (src_width, src_height)
            )
            # print(f"detected: {largest_bbox_yolo}")

            pred_label_tensor = None
            if side == "left":
                pred_label_tensor = torch.tensor([0])
            elif side == "right":
                pred_label_tensor = torch.tensor([1])

            experiment[i]["detected_bbox"] = largest_bbox_yolo

            preds = [
                {
                    "boxes": torch.tensor([largest_bbox_yolo]),
                    "scores": torch.tensor([1.0]),  # confidence score
                    "labels": pred_label_tensor,
                }
            ]
            target = [
                {"boxes": torch.tensor([gt_bbox]), "labels": torch.tensor([clas])}
            ]

            # print(f"preds: {preds}")
            # print(f"target: {target}")
            iou_torch = iou_metric(preds, target)
            iou = iou_torch["iou"]
            ioius_of_detected.append(iou)
            # print(f"iou_torch: {iou_torch}")

            # draw gt add anotation on tehe box
            x, y, w, h = gt_bbox
            x = int((x - w / 2) * src_width)
            y = int((y - h / 2) * src_height)
            w = int(w * src_width)
            h = int(h * src_height)

            # save gt detection
            cv2.imwrite(
                f"gt_test_detections/{file[:-4]}_gt.png", src_copy[y : y + h, x : x + w]
            )

            experiment[i]["gt_img"] = f"gt_test_detections/{file[:-4]}_gt.png"

            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(src, f"gt", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            x, y, w, h = largest_bbox_yolo
            # draw detected
            x = int((x - w / 2) * src_width)
            y = int((y - h / 2) * src_height)
            w = int(w * src_width)
            h = int(h * src_height)
            cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(
                src, f"detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )

            # add iou
            cv2.putText(
                src,
                f"IoU_torch: {iou_torch['iou']}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            # save image
            cv2.imwrite(f"res_test/{file[:-4]}_detected.png", src)

            # save the region of interest
            cv2.imwrite(
                f"test_detections/{file[:-4]}.png", src_copy[y : y + h, x : x + w]
            )

            experiment[i]["detected_img"] = f"test_detections/{file[:-4]}.png"

        ious.append(iou)

    # print(f"Average IoU: {sum(ious)/len(ious)}")
    # print(f"Average IoU of detected: {sum(ioius_of_detected)/len(ioius_of_detected)}")

    results = {
        "detected": len(ioius_of_detected),
        "not_detected": len(ious) - len(ioius_of_detected),
        "avg_iou": str(sum(ious) / len(ious)),
        "avg_iou_detected": str(sum(ioius_of_detected) / len(ioius_of_detected)),
    }

    #

    print(results)

    # run recognition experiments on the test set
    # save experiment to df
    df = pd.DataFrame(experiment)
    df.to_csv("experiment.csv")

    detected_imgs = []
    detected_imgs_filenames = []
    gt_imgs = []
    gt_imgs_filenames = []

    for sample in experiment.copy():
        # extract the detected image
        if not sample["detected_bbox"] is None:
            
            detected_img = cv2.imread(sample["detected_img"])
            detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY)
            detected_img = cv2.resize(detected_img, (24, 24))
            detected_imgs.append(detected_img)
            detected_imgs_filenames.append(sample["detected_img"])
        else:
            #do not add to the list
            pass


       


        gt_img = cv2.imread(sample["gt_img"])
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        gt_img = cv2.resize(gt_img, (24, 24))

     

        gt_imgs.append(gt_img)
        gt_imgs_filenames.append(sample["gt_img"])

    print("detected_imgs: ", len(detected_imgs))
    print("gt_imgs: ", len(gt_imgs))

    # 108: 0.1975 - histograms: False, lbptype: rotation, neighbors: 8, radius: 2, resize_to: 24, use_lib: False, use_raw: False

    lbp_features = extract_lbp_features_mine(
        detected_imgs, detected_imgs_filenames, 8, 2, "rotation", histograms=False
    )

    lbp_features_gt = extract_lbp_features_mine(
        gt_imgs, gt_imgs_filenames, 8, 2, "rotation", histograms=False
    )

     # Convert featsograms to a NumPy array for efficient computation
    features_array = np.array([i["feats"] for i in lbp_features])
    gt_features_array = np.array([i["feats"] for i in lbp_features_gt])
    # remove inner dimension
    features_array = np.squeeze(features_array)
    gt_features_array = np.squeeze(gt_features_array)

    print(f"Features array shape: {features_array.shape}")
    print(f"GT Features array shape: {gt_features_array.shape}")

    # Calculate the cosine similarity matrix (efficient vectorized operation)
    similarity_matrix = cosine_similarity(features_array, features_array)
    gt_similarity_matrix = cosine_similarity(gt_features_array, gt_features_array)

    # Fill the diagonal with zeros to ignore self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    np.fill_diagonal(gt_similarity_matrix, 0)

    # Initialize variables to track the closest matches and accuracy
    closest_matches = np.argmax(similarity_matrix, axis=1)
    gt_closest_matches = np.argmax(gt_similarity_matrix, axis=1)

    correct_matches_gt = 0
    correct_matches = 0

    # Use tqdm to show progress for checking matches
    for i in range(len(gt_closest_matches)):
        # Get the index of the closest match
        closest_index = gt_closest_matches[i]
        # Check if the closest match has the same identity
        if experiment[i]["identity"] == experiment[closest_index]["identity"]:
            correct_matches += 1

    # Calculate accuracy based on the number of correct matches
    accuracy = correct_matches / len(experiment)
    print(f"Accuracy using ground truth VJ detections: {accuracy} {correct_matches}/{len(experiment)}")



    experiment_only_detected = []
    for sample in experiment.copy():
        if not sample["detected_bbox"] is None:
            experiment_only_detected.append(sample)

    print(f"number of detected ears: {len(experiment_only_detected)}/{len(experiment)}")



    #we already heva len(experiment) - len(experiment_only_detected) not detected

    correct_matches_gt = 0

    for i in range(len(closest_matches)):
        # Get the index of the closest match
        closest_index = closest_matches[i]
        # Check if the closest match has the same identity
        if experiment_only_detected[i]["identity"] == experiment_only_detected[closest_index]["identity"]:
            correct_matches_gt += 1


    # Calculate accuracy based on the number of correct matches take into account only detected
    accuracy = correct_matches_gt / len(experiment_only_detected)
    print(f"Accuracy using best VJ detections when considering only detected ears: {accuracy} {correct_matches_gt}/{len(experiment_only_detected)}")

    #accuracy when take into account all images
    accuracy = correct_matches_gt / len(experiment)
    print(f"Accuracy using best VJ detections when considering all images: {accuracy} {correct_matches_gt}/{len(experiment)}")





    #calculate accuracy using raw images

    raw_feats_gt = raw_image_to_lbp(gt_imgs, gt_imgs_filenames)
    raw_feats = raw_image_to_lbp(detected_imgs, detected_imgs_filenames)

    raw_feats_gt=[i["feats"] for i in raw_feats_gt]
    raw_feats=[i["feats"] for i in raw_feats]

    print(f"Raw Features array shape: {len(raw_feats)}")
    print(f"Raw GT Features array shape: {len(raw_feats_gt)}")

    # Calculate the cosine similarity matrix (efficient vectorized operation)
    similarity_matrix = cosine_similarity(raw_feats, raw_feats)
    gt_similarity_matrix = cosine_similarity(raw_feats_gt, raw_feats_gt)

    # Fill the diagonal with zeros to ignore self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    np.fill_diagonal(gt_similarity_matrix, 0)

    # Initialize variables to track the closest matches and accuracy
    closest_matches = np.argmax(similarity_matrix, axis=1)
    gt_closest_matches = np.argmax(gt_similarity_matrix, axis=1)

    correct_matches_gt = 0
    correct_matches = 0

    # Use tqdm to show progress for checking matches
    for i in range(len(gt_closest_matches)):
        # Get the index of the closest match
        closest_index = gt_closest_matches[i]
        # Check if the closest match has the same identity
        if experiment[i]["identity"] == experiment[closest_index]["identity"]:
            correct_matches += 1


    # Calculate accuracy based on the number of correct matches
    accuracy = correct_matches / len(experiment)

    print(f"Accuracy with ground truth raw ear image comparison: {accuracy} {correct_matches}/{len(experiment)}")

    #accuracy when take into account only detected

    correct_matches_gt = 0

    for i in range(len(closest_matches)):
        # Get the index of the closest match
        closest_index = closest_matches[i]
        # Check if the closest match has the same identity
        if experiment_only_detected[i]["identity"] == experiment_only_detected[closest_index]["identity"]:
            correct_matches_gt += 1

    
    # Calculate accuracy based on the number of correct matches take into account only detected
    accuracy = correct_matches_gt / len(experiment_only_detected)

    print(f"Accuracy with best VJ, raw ear image comparison when considering only detected ears: {accuracy} {correct_matches_gt}/{len(experiment_only_detected)}")

    #accuracy when take into account all images
    accuracy = correct_matches_gt / len(experiment)

    print(f"Accuracy with best VJ, raw ear image comparison when considering all images: {accuracy} {correct_matches_gt}/{len(experiment)}")

















    

    return results


if __name__ == "__main__":
    main()
