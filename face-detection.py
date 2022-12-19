import cv2

cap = cv2.VideoCapture(0)

haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    values = haar.detectMultiScale(gray)
    return values

while True:
    
    ret, frame = cap.read()
    
    values = detect_face(frame)
    
    for x,y,h,w in values:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    
    cv2.imshow("window", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    
cap.release()
cv2.destroyAllWindows()