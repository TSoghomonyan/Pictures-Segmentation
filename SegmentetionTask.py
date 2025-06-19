import os
import cv2
import numpy as np

input_folder = r'C:\Users\User\Desktop\TigranSegmentetion\DataSpecialForTigran'
output_folder = r'C:\Users\User\Desktop\TigranSegmentetion\Processed4'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.bmp'):
        file_path = os.path.join(input_folder, filename)

        image = cv2.imread(file_path)
        assert image is not None, f"Failed {filename}"
        print(f"Processing {filename}")
        # Saving copy cuz Watershed work on the spot
        #original = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # draw conturs
        cv2.drawContours(image, contours, -1, (0, 0, 255), 1)  

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