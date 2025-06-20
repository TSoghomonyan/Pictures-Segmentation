import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi 
from skimage.segmentation import watershed 
from skimage.feature import peak_local_max 
from skimage import morphology 
from skimage.measure import label as sk_label

# --- ՕԳՆԱԿԱՆ ՖՈՒՆԿՑԻԱՆԵՐ ---
def erode(img, k_s=3, it=1): 
    kernel = np.ones((k_s, k_s), 'uint8') 
    return cv2.erode(img, kernel, iterations=it)

def dilate(img, k_s=3, it=1): 
    kernel = np.ones((k_s, k_s), 'uint8') 
    return cv2.dilate(img, kernel, iterations=it)

def del_small_areas(thresh, area_black=20, area_white=20): 
    result = morphology.remove_small_objects(sk_label(thresh), area_white)
    result[result > 0] = 255 
    result = morphology.remove_small_objects(sk_label(255 - result), area_black)
    result[result > 0] = 255 
    return 255 - result

# --- ՏԵՂԵԿՈՒԹՅՈՒՆ ՆԿԱՐՆԵՐԻ ՖԱՅԼԵՐԻ ՄԱՍԻՆ ---
input_folder = r'C:\Users\User\Desktop\TigranSegmentetion\DataSpecialForTigran'
output_folder = r'C:\Users\User\Desktop\TigranSegmentetion\ProcessedWaterShedNew'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.bmp'):
        file_path = os.path.join(input_folder, filename)

        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        assert gray is not None, f"Failed to load {filename}"
        print(f"Processing {filename}")

        # Adaptive threshold -> erosion -> dilation -> small area removal
        block_size = 21
        thresh_local_1 = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size, 10
        )
        thresh_local = del_small_areas(erode(dilate(thresh_local_1)))

        # Distance transform
        distance = ndi.distance_transform_edt(thresh_local)

        # Peak local maxima
        local_max = peak_local_max(
            distance,
            min_distance=20,
            footprint=np.ones((3, 3)),
            labels=thresh_local.astype(np.int32)
        )

        # Marker labelling
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(local_max.T)] = True
        markers, _ = ndi.label(mask)

        # Watershed segmentation
        labels = watershed(-distance, markers, mask=thresh_local)

        # Draw contours
        image_contours = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        for label_val in np.unique(labels):
            if label_val == 0:
                continue
            region_mask = np.zeros(thresh_local.shape, dtype="uint8")
            region_mask[labels == label_val] = 255
            cnts = cv2.findContours(region_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cv2.drawContours(image_contours, cnts, -1, (0, 0, 255), 1)

        # -------- MATPLOTLIB Visualization --------
        # fig, axes = plt.subplots(ncols=4, figsize=(12, 6), sharex=True, sharey=True)
        # ax = axes.ravel()

        # ax[0].imshow(gray, cmap=plt.cm.gray)
        # ax[0].set_title('Original\nimage')

        # ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
        # ax[1].set_title('Segmentation\nwatershed')

        # ax[2].imshow(image_contours)
        # ax[2].set_title('Found\nborders')

        # # Dummy mask placeholder (change if needed)
        # mask_image = np.zeros_like(gray)
        # ax[3].imshow(mask_image, cmap=plt.cm.gray)
        # ax[3].set_title('C')

        # for a in ax:
        #     a.set_axis_off()

        # fig.tight_layout()
        # plt.show()
        # ------------------------------------------

        # Save result with contours
        name, ext = os.path.splitext(filename)
        out_path = os.path.join(output_folder, f"{name}_watershed.bmp")
        cv2.imwrite(out_path, image_contours)
        print(f"Saved to {out_path}")

cv2.destroyAllWindows()
