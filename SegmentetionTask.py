import os
import cv2

input_folder = r'C:\Users\User\Desktop\TigranSegmentetion\DataSpecialForTigran'
output_folder = r'C:\Users\User\Desktop\TigranSegmentetion\Processed'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.bmp'):
        file_path = os.path.join(input_folder, filename)

        image = cv2.imread(file_path)
        assert image is not None, f"Failed {filename}"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        print(f"Processing {filename}")

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, thresh)

cv2.destroyAllWindows()