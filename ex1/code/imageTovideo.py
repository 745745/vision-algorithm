import os

import cv2


def makevideo(imagePath):
    fps=24
    size = (752, 480)
    video = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    imageList=os.listdir(imagePath)
    imageList.sort()
    for item in imageList:
        if item.endswith('.jpg'):
            item = imagePath+'/' + item
            img = cv2.imread(item)
            video.write(img)
    video.release()

def playvideo():
    cap = cv2.VideoCapture('test.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()