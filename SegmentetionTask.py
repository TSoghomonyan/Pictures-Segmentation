import os
import cv2
import numpy as np

input_folder = r'C:\Users\User\Desktop\TigranSegmentetion\DataSpecialForTigran'
output_folder = r'C:\Users\User\Desktop\TigranSegmentetion\Processed3'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.bmp'):
        file_path = os.path.join(input_folder, filename)

        image = cv2.imread(file_path)
        assert image is not None, f"Failed {filename}"
        
        # Saving copy cuz Watershed work on the spot
        original = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        print(f"Processing {filename}")
        
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # markers
        ret, markers = cv2.connectedComponents(sure_fg)

        # markers should not be a 0, cuz we also mark 0 as unknown sapces
        #markers = markers + 1

        #markers[unknown == 255] = 0
        
        # using wahteshed method
        markers = cv2.watershed(original, markers)

        # Where are separation paint red
        image[markers == -1] = [0, 0, 255]

        print(f"Processed {filename}")

        # Saving results
        name, ext = os.path.splitext(filename)

        # output_path = os.path.join(output_folder, filename)
        # cv2.imwrite(output_path, thresh)
        # cv2.imwrite(os.path.join(output_folder, f"{name}_thresh.bmp"), thresh)
        # cv2.imwrite(os.path.join(output_folder, f"{name}_open.bmp"), opening)
        # cv2.imwrite(os.path.join(output_folder, f"{name}_surefg.bmp"), sure_fg)
        # cv2.imwrite(os.path.join(output_folder, f"{name}_unknown.bmp"), unknown)
        
        out_path = os.path.join(output_folder, f"{name}_watershed.bmp")
        cv2.imwrite(out_path, image)
cv2.destroyAllWindows()