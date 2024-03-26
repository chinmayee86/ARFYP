
import cv2

def FrameCapture(path):
    
    vidObj = cv2.VideoCapture(path)

    count = 0
    success = 1

    while success:

        success, image = vidObj.read()
                
        image = cv2.resize(image, (160, 120))
        cv2.imwrite("vid55/frame%d.jpg" % count, image)
        count += 1
        
if __name__ == '__main__':
    FrameCapture("vid6.mov")
