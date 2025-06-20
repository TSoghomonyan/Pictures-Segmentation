import os
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# defining the canny detector function
 
# here weak_th and strong_th are thresholds for
# double thresholding step
def Canny_detector(img, weak_th=None, strong_th=None):

    # conversion of image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction step
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Calculating the gradients
    gx = cv2.Sobel(np.float32(blurred), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(blurred), cv2.CV_64F, 0, 1, 3)

    # Conversion of Cartesian coordinates to polar
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # setting the minimum and maximum thresholds 
    # for double thresholding
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5

    # getting the dimensions of the input image
    height, width = gray.shape

    # Looping through every pixel of the grayscale 
    # images
    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # top right (diagonal-1) direction
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            # In y-axis direction
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

            # top left (diagonal-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1
            
            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x]= 0
                    continue
 
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x]= 0

    # Don't needed
    # weak_ids = np.zeros_like(img)
    # strong_ids = np.zeros_like(img)

    ids = np.zeros_like(gray)

    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):

            grad_mag = mag[i_y, i_x]

            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2

    return mag.astype(np.uint8)

folder_path = r'C:\Users\User\Desktop\TigranSegmentetion\DataSpecialForTigran'
output_folder = r'C:\Users\User\Desktop\TigranSegmentetion\ProcessedCombined2'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.bmp'):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)

        if image is None:
            print(f"error {filename}")
            continue

        original = image.copy()
        canny_edges = Canny_detector(image)

        distance = ndi.distance_transform_edt(canny_edges)

        # Cast Error solved
        labels_mask = canny_edges.astype(np.uint8)

        local_max = peak_local_max(
            distance,
            min_distance=20,
            labels=labels_mask,
            footprint=np.ones((3, 3))
        )

        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(local_max.T)] = True
        markers, _ = ndi.label(mask)

        labels = watershed(-distance, markers, mask=labels_mask)

        for label in np.unique(labels):
            if label == 0:
                continue
            mask = np.zeros(canny_edges.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(original, cnts, -1, (0, 0, 255), 1)

        name, ext = os.path.splitext(filename)
        out_path = os.path.join(output_folder, f"{name}_combined.bmp")
        cv2.imwrite(out_path, original)
        print(f"Saved: {out_path}")
        print(f"Done {filename}")

cv2.destroyAllWindows()
