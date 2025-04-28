import cv2
import face_recognition

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

#face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
img= cv2.imread("akshitha.jpg")
face_locations = face_recognition.face_locations(img, model="cnn")
#cv2.imshow("Loaded Image", img) 
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104, 177, 123))
net.setInput(blob)
detections = net.forward()
#if len(faces)>0:
 ##  face_roi=img[y:y+h, x:x+h]
#zoomed_face=cv2.resize(face_roi, (w*2,h*2)) #cropping the detected face and hten zooming but this sisnt working 


zoom_factor=1.5
height,width=img.shape[:2]
zoomed_in=cv2.resize(img,(int(width*zoom_factor), int(height*zoom_factor)))
cv2.imshow("Zoomed IN", zoom_factor)

blurred=cv2.GaussianBlur(img,(25,25),0)
cv2.imshow("Gaussian Blurred", blurred)


for (x,y,w,h)in faces:
    cv2.rectangle(img,(x,y), (x+w,y+h), (255,0, 0), 4)
   # cv2.putText(img, "Face Detected", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5(0,255,0),2, cv2.LINE_AA) not excecuting


cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


