#import libraries
import cv2
import numpy as np
print('Tasks are as follows:\n1) Edge Detection\n2) Motion Imaging')
n = int(input(('Enter the task to perform: ')))
# Video Capture
cap = cv2.VideoCapture(0)

# Read the capture and get the first frame.

ret, frame1 = cap.read()

# CONVERT FRAME TO GRAY SCALE.
prev_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# CREATE AN IMAGE WITH SAME DIMENSIONS AS THE FRAME FOR THE LATER DRAWING PURPOSE.
mask = np.zeros_like(frame1)

# SATURATION TO MAXIMUM.
mask[...,1] = 255

if n==1:
    cv2.namedWindow("Threshold")
    def nothing(x):
        pass

    cv2.createTrackbar('Threshold','Threshold',0,500,nothing)
    kernel = np.ones((5,5),np.float32)/25

def edge(frame2):
    img = frame2 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(img,cv2.CV_64F,ksize=3)
    boundaries = np.uint8(np.absolute(lap))
    sobelX = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = cv2.Sobel(img,cv2.CV_64F,0,1)
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv2.bitwise_or(sobelX,sobelY)
    sobel = np.uint8(np.absolute(sobel))
    if ret ==True:
        thrs = cv2.getTrackbarPos('Threshold','Threshold')
        canny = cv2.Canny(img,thrs,500)
        cv2.imshow('Edge',canny)
        cv2.imshow('oroginal',img)    

def motion_img(frame2,prev_gray):
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    mask[...,0] = ang*180/np.pi/2
    
    mask[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(mask,cv2.COLOR_HSV2BGR)
    cv2.imshow('Original',frame2)
    cv2.imshow('Motion Imaging',rgb)
    prev_gray = next
    

while(cap.isOpened()):
    
    # READ THE CAPTURE AND GET THE FIRST FRAME.

    ret, frame2 = cap.read()

############ Edge Detection ##############
    if n == 1:
        edge(frame2)
###########################################  

############ Motion Imaging ####################
    if n == 2:
        motion_img(frame2,prev_gray)
###############################################        
    
    if cv2.waitKey(300) & 0xff == ord("q"):
        break
    

cap.release()
cv2.destroyAllWindows()
