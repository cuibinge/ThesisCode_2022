# import os
# import numpy as np
# from scipy.ndimage import convolve
# from skimage import io, color
# from skimage.util import img_as_ubyte
#
#
# def mask_edge_extraction(mask):
#     """
#     Extract the edges from the binary mask using a 3x3 neighborhood filter.
#
#     Args:
#     - mask (np.array): Binary mask where 0 represents the background and 1 represents the object.
#
#     Returns:
#     - edge_mask (np.array): The extracted edge mask.
#     """
#     # Define a 3x3 kernel that checks the 8-connected neighbors of a pixel
#     kernel = np.array([[1, 1, 1],
#                        [1, 0, 1],
#                        [1, 1, 1]])
#
#     # Find the neighbors' sum using convolution
#     neighbor_sum = convolve(mask, kernel, mode='constant', cval=0)
#
#     # Extract edge pixels where the current pixel is 0, but has non-zero neighbors
#     edge_mask = np.where((mask == 0) & (neighbor_sum > 0), 1, 0)
#
#     return edge_mask
#
#
# def load_binary_image(image_path):
#     """
#     Load a binary image from the file system and convert it to a binary mask (0 and 1).
#
#     Args:
#     - image_path (str): Path to the binary image file.
#
#     Returns:
#     - mask (np.array): The binary mask.
#     """
#     # Load the image
#     image = io.imread(image_path)
#
#     # Convert the image to grayscale if it is not
#     if len(image.shape) == 3:
#         image = color.rgb2gray(image)
#
#     # Convert the grayscale image to binary (thresholding)
#     binary_mask = np.where(image > 0.5, 1, 0)
#
#     return binary_mask
#
#
# def process_images(input_folder, output_folder):
#     """
#     Process all binary images in a folder, generate edge masks, and save them in the output folder.
#
#     Args:
#     - input_folder (str): Path to the folder containing binary images.
#     - output_folder (str): Path to the folder where edge masks will be saved.
#     """
#     # Create the output folder if it does not exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # Process each image in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
#             # Full path to the input image
#             image_path = os.path.join(input_folder, filename)
#
#             # Load the binary mask
#             binary_mask = load_binary_image(image_path)
#
#             # Generate the edge mask
#             edge_mask = mask_edge_extraction(binary_mask)
#
#             # Save the edge mask to the output folder with the same name
#             output_path = os.path.join(output_folder, filename)
#             io.imsave(output_path, img_as_ubyte(edge_mask))
#             print(f"Processed and saved: {filename}")
#
#
# # Example usage:
# input_folder = r'new_glacier/label_A'  # Replace with the path to your folder of binary images
# output_folder = r'new_glacier/A_mask'  # Replace with the path to the folder where you want to save edge masks
#
# process_images(input_folder, output_folder)
import os
import numpy as np
from skimage import io, color, exposure
from skimage.feature import canny
from skimage.morphology import binary_closing, remove_small_objects
from skimage.util import img_as_ubyte


def enhance_contrast(image):
    """
    Enhance the contrast of the image to improve edge detection.

    Args:
    - image (np.array): Grayscale or binary image.

    Returns:
    - contrast_image (np.array): Contrast enhanced image.
    """
    # Rescale intensity to improve contrast
    contrast_image = exposure.rescale_intensity(image, in_range=(0, 1), out_range=(0, 1))
    return contrast_image


def mask_edge_extraction(mask):
    """
    Extract the edges from the binary mask using a combination of canny edge detection and morphological operations.

    Args:
    - mask (np.array): Binary mask where 0 represents the background and 1 represents the object.

    Returns:
    - edge_mask (np.array): The extracted edge mask.
    """
    # Use Canny edge detection to improve edge extraction
    edges = canny(mask, sigma=1.0)

    # Morphological closing to remove small gaps and clean edges
    cleaned_edges = binary_closing(edges)

    # Remove small objects (noise) in the edge mask
    cleaned_edges = remove_small_objects(cleaned_edges, min_size=10)

    return cleaned_edges


def load_binary_image(image_path):
    """
    Load a binary image from the file system and convert it to a binary mask (0 and 1).

    Args:
    - image_path (str): Path to the binary image file.

    Returns:
    - mask (np.array): The binary mask.
    """
    # Load the image
    image = io.imread(image_path)

    # Convert the image to grayscale if it is not
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Enhance contrast before thresholding
    image = enhance_contrast(image)

    # Convert the grayscale image to binary (thresholding)
    binary_mask = np.where(image > 0.5, 1, 0)

    return binary_mask


def process_images(input_folder, output_folder):
    """
    Process all binary images in a folder, generate edge masks, and save them in the output folder.

    Args:
    - input_folder (str): Path to the folder containing binary images.
    - output_folder (str): Path to the folder where edge masks will be saved.
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Full path to the input image
            image_path = os.path.join(input_folder, filename)

            # Load the binary mask
            binary_mask = load_binary_image(image_path)

            # Generate the edge mask
            edge_mask = mask_edge_extraction(binary_mask)

            # Save the edge mask to the output folder with the same name
            output_path = os.path.join(output_folder, filename)
            io.imsave(output_path, img_as_ubyte(edge_mask))
            print(f"Processed and saved: {filename}")


# Example usage:
input_folder = r'new_glacier/label_A'  # Replace with the path to your folder of binary images
output_folder = r'new_glacier/A_mask'  # Replace with the path to the folder where you want to save edge masks
process_images(input_folder, output_folder)
