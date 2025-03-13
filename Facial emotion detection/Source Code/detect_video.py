import os
import tensorflow as tf
import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

fromaddr="tdinesh986@gmail.com" 


toaddr= "tps.pythont23@gmail.com"




class Video:
    def __init__(self,camera_id=0):
        self.model = tf.keras.models.load_model("face_emotion_rec_v2.h5")
        self.path = "haarcascade_frontalface_default.xml"
        self.camera_id = camera_id
        self.font_scale = 1.5 
        self.font = cv2.FONT_HERSHEY_PLAIN
    
    

    def video_stream(self):
        classNames = []
        classFile = 'label.names'
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        face_roi = None    
        cap = cv2.VideoCapture(self.camera_id)
        #Check if the webcam is open correctly
        if not cap.isOpened():
            raise IOError("Can't open Webcam")
        
        def mail(text):
            print(text)
            msg=MIMEMultipart()
            msg['From']=fromaddr
            msg['To']=toaddr
            msg['Subject']="EMOTION DETECTION"
            body=text
            msg.attach(MIMEText(body,'plain'))
            filename="output/img.jpg"
            attachment=open("output/img.jpg","rb")
            p=MIMEBase('application','octet-stream')
            p.set_payload((attachment).read())
            encoders.encode_base64(p)
            p.add_header('Content-Disposition',"attachment; filename=%s"%filename)
            msg.attach(p)
            s=smtplib.SMTP('smtp.gmail.com',587)
            s.starttls()
            s.login(fromaddr,"zfyvrsnrvlyuuruz") 
            text=msg.as_string()
            s.sendmail(fromaddr,toaddr,text)
            s.quit()

        while True:
            ret , frame = cap.read()
            face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detect.detectMultiScale(gray_img, 1.1, 4)
            
            for x, y, w, h in faces:
                roi_gray_img = gray_img[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                facess = face_detect.detectMultiScale(roi_gray_img)
                if len(facess) == 0:
                    print("Face not detected")
                else:
                    for (ex,ey,ew,eh) in facess:
                        face_roi = roi_color[ey: ey+eh, ex:ex +ew]       

            if face_roi is not None:
                final_img = cv2.resize(face_roi, (224, 224))
                final_img = np.expand_dims(final_img, axis=0) # need 4th dimension
                final_img = final_img/255 # normalizing             

                prediction = self.model.predict(final_img)
                pred = np.argmax(prediction[0])

                x1, y1, w1, h1 = 0, 0, 175, 75

                cv2.rectangle(frame, (x1,x1), (x1+w1,y1+h1), (0,0,0), -1)

                emotion = classNames[pred]

                cv2.putText(frame, emotion, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame, emotion, (100, 150), self.font, 3, (0,0,255), cv2.LINE_4)

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))

                # Send email if emotion is "sad"
                if emotion == "Sad":
                   print("Sad detected")
                   cv2.imwrite('output/img.jpg', frame)
                   mail("SAD DETECTED")

            cv2.imshow("Face emotion recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

if __name__ == "__main__":
    fer = Video()
    fer.video_stream()     
