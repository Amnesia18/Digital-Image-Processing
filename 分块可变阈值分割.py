import cv2
import numpy as np

def my_Otsu(im):
    # Apply Gaussian blur to the subregion
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    # Use OpenCV's Otsu's thresholding
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def divide_image(im, h_divisions, w_divisions):
    h, w = im.shape[:2]
    h_sub = h // h_divisions
    w_sub = w // w_divisions
    
    # Adjust the last subregion size if the image isn't evenly divisible
    subregions = []
    for i in range(h_divisions):
        for j in range(w_divisions):
            i_start = i * h_sub
            j_start = j * w_sub
            i_end = h if i == h_divisions - 1 else (i + 1) * h_sub
            j_end = w if j == w_divisions - 1 else (j + 1) * w_sub
            subregions.append(im[i_start:i_end, j_start:j_end])
    return subregions

def reconstruct_image(subregions, h_divisions, w_divisions):
    # Concatenate subregions back into an image
    rows = [np.hstack(subregions[i*w_divisions:(i+1)*w_divisions]) for i in range(h_divisions)]
    imd = np.vstack(rows)
    return imd

# Replace 'path_to_image' with the actual image path
im = cv2.imread('digital signal processing/bgztp.png', 0)

# Parameters for dividing the image into subregions
h_divisions = 2
w_divisions = 3

# Divide the image and apply Otsu's thresholding to each subregion
subregions = divide_image(im, h_divisions, w_divisions)
thresh_subregions = [my_Otsu(sr) for sr in subregions]

# Reconstruct the divided image from the thresholded subregions
im_segmented = reconstruct_image(thresh_subregions, h_divisions, w_divisions)

# Save and display the result
cv2.imwrite('result_segmented.jpg', im_segmented)
cv2.imshow('Segmented Image', im_segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
