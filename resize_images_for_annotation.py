import cv2
import sys
import os

# Load the image
def convert_image_for_annotation(from_directory_name, file_name, image_width, image_height):
    # Adaptive Equalizer
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Reads in image
    image = cv2.imread(f'{from_directory_name}/{file_name}')

    # Converts image
    resized_image = cv2.resize(image, (image_width, image_height))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    equalized_image = clahe.apply(gray_image)

    return equalized_image

def save_image_in_directory(target_directory_name, file_name, image):
    cv2.imwrite(f'{target_directory_name}/{file_name}', image)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("First argument is name of from directory. Second argument is name of target directory")
        exit()

    from_directory_name = sys.argv[1]
    target_directory_name = sys.argv[2]
    image_width = int(sys.argv[3])
    image_height = int(sys.argv[4])

    from_directory_files = os.listdir(from_directory_name)

    # For each image in from directory, convert and save in target directory
    for file_name in from_directory_files:
        image = convert_image_for_annotation(from_directory_name, file_name, image_width, image_height)
        save_image_in_directory(target_directory_name, file_name, image)
