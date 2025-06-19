import os
import cv2


folder_path = r'C:\Users\User\Desktop\TigranSegmentetion\DataSpecialForTigran'

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.bmp'):
        file_path = os.path.join(folder_path, filename)

        image = cv2.imread(file_path)
        
        if image is None:
            print(f"error {filename}")
            continue

        cv2.imshow('Image', image)
        print(f"Showing {filename}")

        cv2.waitKey(0)

cv2.destroyAllWindows()