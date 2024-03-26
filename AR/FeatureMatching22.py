import cv2
import numpy as np
import time



MIN_MATCHES = 20
detector = cv2.ORB_create(nfeatures=5000)


index_params = dict(algorithm = 1, trees=3)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params,search_params)




def load_input(input_image):
    input_image = cv2.resize(input_image, (400,550),interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    return gray_image,keypoints, descriptors




def compute_matches(descriptors_input, descriptors_output):
    
    if(len(descriptors_output)!=0 and len(descriptors_input)!=0):
        matches = flann.knnMatch(np.asarray(descriptors_input,np.float32),np.asarray(descriptors_output,np.float32),k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.68*n.distance:
                good.append([m])
        return good
    else:
        return None




if __name__=='__main__':

    

    input_image1 = cv2.imread('img11.jpg')
    input_image2 = cv2.imread('img33.jpg')
    input_image4 = cv2.imread('img55.png')
    input_image5 = cv2.imread('img66.jpg')

    input_image1, input_keypoints1, input_descriptors1 = load_input(input_image1)
    input_image2, input_keypoints2, input_descriptors2 = load_input(input_image2)
    input_image4, input_keypoints4, input_descriptors4 = load_input(input_image4)
    input_image5, input_keypoints5, input_descriptors5 = load_input(input_image5)

    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    outputFlag = 0

    while(ret):
        ret, frame = cap.read()

        
        if(len(input_keypoints1)<MIN_MATCHES and len(input_keypoints2)<MIN_MATCHES and len(input_keypoints3)<MIN_MATCHES and len(input_keypoints4)<MIN_MATCHES):
            continue
        
        frame = cv2.resize(frame, (700,600)) 
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        output_keypoints, output_descriptors = detector.detectAndCompute(frame_bw, None)
        matches1 = compute_matches(input_descriptors1, output_descriptors)
        matches2 = compute_matches(input_descriptors2, output_descriptors)
        matches4 = compute_matches(input_descriptors4, output_descriptors)
        matches5 = compute_matches(input_descriptors5, output_descriptors)
        
        if(len(matches1)>20):
            outputFlag = 1
            id = 1
            print('Match found Image 1')
        elif(len(matches2)>20):
            outputFlag = 2
            id2 = 1
            print('Match found Image 2')
        elif(len(matches4)>20):
            outputFlag = 4
            id4 = 1
            print('Match found Image 4')
        elif(len(matches5)>20):
            outputFlag = 5
            id5 = 1
            print('Match found Image 5')
        else:
            sjdfks=1
            print('No Match found')

        if(outputFlag==1):
            mask = cv2.imread('vid11/frame%d.jpg' % id)
            id = id + 1
            if(id>=300):
                outputFlag=0
            pos = (100, 100)
            frame[200:(200+mask.shape[0]), 200:(200+mask.shape[1])] = mask
            cv2.imshow('Final Output', frame)

        if(outputFlag==2):
            mask = cv2.imread('vid33/frame%d.jpg' % id2)
            id2 = id2 + 1
            if(id2>=300):
                outputFlag=0
            pos = (200, 200)
            frame[200:(200+mask.shape[0]), 200:(200+mask.shape[1])] = mask
            cv2.imshow('Final Output', frame)

       
        if(outputFlag==4):
            mask = cv2.imread('vid55/frame%d.jpg' % id4)
            id4 = id4 + 1
            if(id4>=300):
                outputFlag=0
            pos = (200, 200)
            frame[200:(200+mask.shape[0]), 200:(200+mask.shape[1])] = mask
            cv2.imshow('Final Output', frame)
            
        if(outputFlag==5):
            mask = cv2.imread('vid66/frame%d.jpg' % id5)
            id5 = id5 + 1
            if(id5>=300):
                outputFlag=0
            pos = (200, 200)
            frame[200:(200+mask.shape[0]), 200:(200+mask.shape[1])] = mask
            cv2.imshow('Final Output', frame)

        if(outputFlag==0):
            cv2.imshow('Final Output', frame)
                            
        key = cv2.waitKey(5)
        if(key==27):
            break
            
        time.sleep(.05)
