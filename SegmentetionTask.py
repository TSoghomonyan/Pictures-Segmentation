import os
import cv2
import numpy as np

input_folder = r'C:\Users\User\Desktop\TigranSegmentetion\DataSpecialForTigran'
output_folder = r'C:\Users\User\Desktop\TigranSegmentetion\Processed9' 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.bmp'):
        file_path = os.path.join(input_folder, filename)

        image = cv2.imread(file_path)
        assert image is not None, f"Failed to load {filename}"
        original = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 2)
        # thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 6)
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        # threshold_value = np.mean(gray) + 30
        # all this are worse than hardcoded
        
        mask = gray > 180

        image[mask] = [0, 0, 255]  

        
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_red.bmp")
        cv2.imwrite(output_path, image)

        print(f"Processed and saved: {output_path}")

cv2.destroyAllWindows()