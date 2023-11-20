#!/bin/bash

# Directory containing images
image_directory="./gt_detections"

# Initialize sum of ratios and image count
sum_ratios=0
count=0

# Iterate over all image files in the directory
for image in "$image_directory"/*.{jpg,jpeg,png,gif,bmp}; do
  if [ -f "$image" ]; then
    # Extract image width and height using ImageMagick
    dimensions=$(identify -format "%w %h" "$image")
    width=$(echo $dimensions | cut -d' ' -f1)
    height=$(echo $dimensions | cut -d' ' -f2)

    # Calculate the width-to-height ratio for the current image
    ratio=$(echo "scale=2; $width / $height" | bc)

    # Add the ratio to the sum of ratios
    sum_ratios=$(echo "scale=2; $sum_ratios + $ratio" | bc)
    ((count++))
  fi
done

# Calculate the average ratio
if [ $count -gt 0 ]; then
  average_ratio=$(echo "scale=2; $sum_ratios / $count" | bc)
  echo "Average ratio of images: $average_ratio"
else
  echo "No images found in the specified directory."
fi
