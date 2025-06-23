import os
import cv2
import numpy as np

input_folder = r'C:\Users\User\Desktop\TigranSegmentetion\DataSpecialForTigran'
output_folder = r'C:\Users\User\Desktop\TigranSegmentetion\Processed7'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.bmp'):
        file_path = os.path.join(input_folder, filename)

        image = cv2.imread(file_path)
        assert image is not None, f"Failed to load {filename}"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mask = gray > 150

        image[mask] = [0, 0, 255]  

        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_red.bmp")
        cv2.imwrite(output_path, image)

        print(f"Processed and saved: {output_path}")

cv2.destroyAllWindows()