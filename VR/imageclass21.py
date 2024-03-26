import cv2
import numpy as np
import os
from ffpyplayer.player import MediaPlayer


index_params = dict(algorithm = 1, trees=3)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params,search_params)


def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    print(video_path)
    player = MediaPlayer(video_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()

orb=cv2.ORB_create(nfeatures=5000)

path="imagesquery"
images=[]
classnames=[]
mylists=os.listdir(path)
print("Total classes detected", len(mylists))
for cl in mylists:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classnames.append(os.path.splitext(cl)[0])


def findDes(images):
    desList=[]
    for i in images:
        kp,des=orb.detectAndCompute(i,None)
        desList.append(des)
    return desList

def findId(img,desList,thresh=40):
    kp2,des2=orb.detectAndCompute(img,None)
    bf =cv2.BFMatcher()
    matchList=[]
    finalVal=-1
    try:
        for des in desList:
            matches = flann.knnMatch(np.asarray(descriptors_input,np.float32),np.asarray(descriptors_output,np.float32),k=2)
            good=[]
            for m,n in matches:
                if m.distance< 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    if len(matchList)!=0 :
        if max(matchList)>thresh :
            finalVal=matchList.index(max(matchList))
    return finalVal    

desList=findDes(images)

cap= cv2.VideoCapture(0)

while True:
    success,img2=cap.read()
    imgOriginal=img2.copy()
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    cv2.imshow('imgOriginal',imgOriginal)

    id = findId(img2, desList)

    if id!=-1:
        cv2.putText(imgOriginal,classnames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
        video_path='videoss\\'+classnames[id]+'.mp4'
        PlayVideo(video_path)
    cv2.waitKey(1)



'''

cv2.imshow('kp1',imkp1)
cv2.imshow('kp2',imkp2)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)

cv2.waitKey(0)
'''
