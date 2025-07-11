import cv2
import glob
import concurrent.futures
import time  # For measuring execution time

# Function to load and resize an image
def load_and_resize_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load {image_path}")
            return None
        img = cv2.resize(img, (640, 480))  # Resize to smaller size to reduce memory usage
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to stitch a subset of images
def stitch_images(image_subset):
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    error, stitched_image = stitcher.stitch(image_subset)
    return error, stitched_image

# Function to create overlapping subsets
def create_overlapping_subsets(images, subset_size, overlap):
    subsets = []
    for i in range(0, len(images), subset_size - overlap):
        subset = images[i:i + subset_size]
        if len(subset) > 1:  # Ensure subsets have at least two images
            subsets.append(subset)
    return subsets

# Start timer
start_time = time.time()

# Load image paths
image_paths = glob.glob('unstitched images/*.jpg')
if not image_paths:
    print("Error: No images found in the specified directory.")
    exit()

# Parallel loading and resizing of images
images = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(load_and_resize_image, path): path for path in image_paths}
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result is not None:
            images.append(result)

# Time taken for loading and resizing images
load_resize_time = time.time() - start_time
print(f"Time taken for loading and resizing images: {load_resize_time:.2f} seconds")

# Check if images were loaded
if len(images) < 2:
    print("Error: Not enough images to stitch.")
    exit()

# Create sequential overlapping subsets
subset_size = 4 # Number of images in each subset
overlap = 1       # Number of overlapping images between subsets
image_subsets = create_overlapping_subsets(images, subset_size, overlap)

# Parallel stitching of subsets
stitched_results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(stitch_images, subset): subset for subset in image_subsets}
    for future in concurrent.futures.as_completed(futures):
        error, stitched = future.result()
        if error == cv2.Stitcher_OK:
            stitched_results.append(stitched)
        else:
            print(f"Stitching failed for a subset with error code: {error}")

# Check if any subsets were successfully stitched
if not stitched_results:
    print("Error: All subsets failed to stitch.")
    exit()

# Stitch the resulting subsets into a final panorama
final_error, final_stitched = stitch_images(stitched_results)
if final_error == cv2.Stitcher_OK:
    cv2.imwrite("finalStitchedOutput.png", final_stitched)
    print("Final panorama created successfully!")
    cv2.imshow("Stitched Image", final_stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Final stitching failed with error code: {final_error}")
    if final_error == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("Error: Need more images with better overlap.")
    elif final_error == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("Error: Homography estimation failed.")
    elif final_error == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("Error: Camera parameters adjustment failed.")

# Clear images list to release memory after stitching
images.clear()

# Total time
total_time = time.time() - start_time
print(f"Total time taken: {total_time:.2f} seconds")