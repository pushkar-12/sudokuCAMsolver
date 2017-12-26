import cv2
import numpy as np
img=cv2.imread('mnistExamples.png',0)

img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,11,2)



im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)





cont=[]
maxarea=-1
for c in contours:
    area=cv2.contourArea(c)
    if(area>maxarea):
        maxarea=area
        cont=c

contours.remove(cont)

#cv2.drawContours(img,contours,-1,(0,255,0),1)


samples =  np.empty((0,100))
responses = []

cnt=0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if(w*h>150):
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        temp=img[y-2:y+h+2,x-2:x+w+2]
        temp = cv2.resize(temp, (10, 10), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('image',temp)

        key = cv2.waitKey(0)

        if key <= 57 and key >= 48:
            responses.append((key))
            sample = temp.reshape((1, 100))
            samples = np.append(samples, sample, 0)
        cnt += 1
        print(cnt)
        if (key == ord('q')):
            break
        cv2.destroyAllWindows()

responses = np.array(responses, np.uint8)
responses = responses.reshape((responses.size, 1))
print("training complete")
np.savetxt('generalsamples4.data', samples)
np.savetxt('generalresponses4.data', responses)
