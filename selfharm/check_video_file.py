import cv2
import os

path = "./rgb"
file_list = os.listdir(path)
error_list = []

for file in file_list:
    file_path = os.path.join(path, file)
    video = cv2.VideoCapture(file_path)
    print(file_path)
    if not video.isOpened():
        print("Could not Open :", file)
        error_list.append(file)
        
print(error_list)
print(len(error_list))