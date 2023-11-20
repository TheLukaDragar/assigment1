import cv2
import os
import torch
from torchmetrics.detection import IntersectionOverUnion
import argparse
import random




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

    return x_center,y_center,w_norm,h_norm

def read_ground_truth(file_path):
    """
    Reads the ground truth bounding boxes from a file.

    :param file_path: Path to the ground truth .txt file
    :return:tuple representing ground truth bounding box
    """
    ground_truths = []
    with open(file_path, 'r') as f:
            
        for line in f:
            clas, x_center, y_center, width, height = [float(n) for n in line.split()]
            ground_truths.append((clas,x_center, y_center, width, height))
            #only one gt per image
            break

    return ground_truths[0]


def parse_args():
    """
    Parses command line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Detect ears in images')
    parser.add_argument('--dataset', help='path to ear images', default='./ears')
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


    #open identities file and read the identities
    identities = []
    with open("identities.txt", 'r') as f:
        for line in f:
            #0501.png 001
            file, identity = line.split()
            identities.append({
                "file": file,
                "identity": identity
            })






        
    #get how many unique identities we have
    # print(f"Found {len(set([i['identity'] for i in identities]))} unique identities")

    # #get how many images we have per identity
    # print(f"Found {len(identities)/len(set([i['identity'] for i in identities]))} images per identity")



   
    


    files = [ x['file'] for x in identities]




    ious = []
    ioius_of_detected = []



    for file in files:
        image_path = os.path.join(args.dataset, file)
        src = cv2.imread(image_path)
        src_copy = src.copy()

        





        
        # save in Yolo format

        src_width = src.shape[1]
        src_height = src.shape[0]

        #only one gt per image so we can select the largest detection from the left and right ear detections
      

        gt = read_ground_truth(os.path.join(args.dataset, file[:-4] + ".txt"))


        #0 is left ear, 1 is right ear
        clas, x_center, y_center, width, height = gt
        gt_bbox = (x_center, y_center, width, height)

        #save gt rois too
        x, y, w, h = gt_bbox
        x = int((x - w/2) * src_width)
        y = int((y - h/2) * src_height)
        w = int(w * src_width)
        h = int(h * src_height)


        cv2.imwrite(f"gt_detections/{file[:-4]}.png", src_copy[y:y+h, x:x+w])





if __name__ == "__main__":
    if os.path.exists("gt_detections"):
        os.system("rm -rf gt_detections")
    os.mkdir("gt_detections")

    main()




        



