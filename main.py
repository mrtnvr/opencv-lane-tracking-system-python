import cv2
import numpy as np
import time
import pandas as pd
import math


xa=0
ca=0
xb=0
ya=550
yb=750
ua=170
uacon=170
ub = 1180
ubcon = 1180
solort = 363
sagort = 968
thres = 0.45



def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

	
            
def drow_the_lines(img, lines ,
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    global xa
    global ca
    global xb
    global ya
    global yb
    
    global ua
    global uacon
    
    global ub 
    global ubcon

    global sagort
    global solort
    
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
   
    


    for line in lines:
        for x1, y1, x2, y2 in line:
            #cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=2)
            if x1 > 450 and x1 < 530 and y1 > 510 and y1 < 525: #sol üst
                ya = x1               
            if x2 > 740 and x2 < 870 and y2 > 510 and y2 < 525 : #sağ üst
                yb = x2        
            if y1 > 690 and y1 < 720 and x1 > 100 and x1 < 600: #sol alt
                ua = x1
            if y1 > 700 and y2 < 740 and x1 > 601 and x1 < 1120:  #sağ alt
                ub = x1
            if x1 > 220 and x1 < 470 and y1 > 590 and y1 < 630:  #sol ort
                solort = x1
            if x1 > 720 and x1 < 805 and y1 > 590 and y1 < 630:  #sol ort
                sagort = x2   
    ret, frames = cap.read()   
     
    cv2.line(blank_image,(ya,520),(solort,620), (0, 0, 255), thickness=3) #sol 
    cv2.line(blank_image,(solort,620),(ua,720), (0, 0, 255), thickness=3) #sol ort baplama
    
    cv2.line(blank_image,(yb,520),(sagort,620), (0, 0, 255), thickness=3) #sağ
    cv2.line(blank_image,(sagort,620),(ub,720), (0, 0, 255), thickness=3) #sag ort
    
    cv2.fillConvexPoly(blank_image, np.array([(yb,520),(ya,520),(solort,620),(sagort,620)], 'int32'), (255,0,0))       
    cv2.fillConvexPoly(blank_image, np.array([(solort,620),(sagort,620),(ub,720),(ua,720)], 'int32'), (255,0,0)) 
    
    lc = (ya+ua)/2
    lc = int(lc)
    rc = (yb+ub)/2
    rc = int(rc)
    on = (rc+lc)/2
    on = int(on)
    fark_sag = (on-lc)*(10/57)
    fark_sol = (rc-on)*(10/57)
    durum = ''
    if fark_sag < 50 : 
        durum = 'too left'
    elif fark_sag == 50 : 
        durum = 'fine'
    elif fark_sag > 51:
        durum = 'too right'

    fark_sag = int(fark_sag)
    cv2.line(blank_image,(lc,640),(lc,600), (255, 255, 255), thickness=2) #sol cıbık
    cv2.line(blank_image,(rc,640),(rc,600), (255, 255, 255), thickness=2) #sag cıbık
    cv2.line(blank_image,(on,640),(on,600), (255, 255, 255), thickness=2) #orta sabit nokta
    
  
    cv2.putText(blank_image,' %d' % (fark_sag), (525, 550), cv2.FONT_HERSHEY_COMPLEX , 1.2, (0, 255, 0), 2,cv2.LINE_AA)
    cv2.putText(blank_image,' %d' % (100-fark_sag), (775, 550), cv2.FONT_HERSHEY_COMPLEX , 1.2, (0, 255, 0), 2,cv2.LINE_AA)
    cv2.putText(blank_image,'%s' % (durum), (525, 650), cv2.FONT_HERSHEY_COMPLEX , 1.2, (0, 255, 0), 2,cv2.LINE_AA)
    
    
    img = cv2.addWeighted(img, 0.2,blank_image, 1, 0.0)
    img = cv2.addWeighted(img,1,frames,1, 0.0)
    
    return img

def process(image):
   

    height = image.shape[0]
    width = image.shape[1]
    shape = np.array([[int(0), int(height)], [int(width), int(height)], [int(0.55*width), int(0.67*height)], [int(0.45*width), int(0.67*height)]])
  
    '''
    region_of_interest_vertices = [
        (120, height),
        (width/2,height-280),
        (width, height)
    ]
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 380, 270)
    
    cropped_image = region_of_interest(canny_image,
                    np.array([shape], np.int32),)
   
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    # = getObjects(blank_image,0.45,0.2,objects=[])
    image_with_lines = drow_the_lines(image, lines)
   # cv2.imshow('frasme',cropped_image)
    return image_with_lines
   


cap = cv2.VideoCapture("Test_Video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
