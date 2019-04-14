import cv2
import os
user=input("Enter the name of the person \n")
while os.path.exists("images/"+user):
    print('User with this name already exist. Please try again with new name or press ctrl+c to quit')
    user=input("Enter the name of the person \n")
os.makedirs("images/"+user)
video_capture = cv2.VideoCapture(0)
# Check success
if not video_capture.isOpened():
    raise Exception("Could not open video device")
# Read picture. ret === True on success
count=0
while count<100:
    ret, frame = video_capture.read()
    cv2.imwrite("images/"+user+"/image"+str(count)+".jpg",frame)
    count=count+1
    # Close device
video_capture.release()
