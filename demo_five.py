import cv2
import mediapipe as mp
import numpy as np
from Fatigue import Fatigue
import dlib
from imutils import face_utils
from tensorflow.keras.models import load_model
import pandas as pd
import time
#import openpyxl

ptime_new= 0
#### mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


#### dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('tools/shape_predictor_68_face_landmarks.dat')

## cascade
face_cascade = cv2.CascadeClassifier("tools/haarcascade_frontalface_default.xml")

## emtion model
label = ['angry','disgust','fear','happy','neutral','sad','surprise']
model = load_model('tools/emotiondetector.h5')


face_mesh_draw = False
#cap = cv2.VideoCapture('C:/Users/Hadi/Desktop/Fatigue_project/tools/8-3/segment_8.mp4')
cap = cv2.VideoCapture(0)

result_text = ""
predicted_emotion_label = ""
## thresholds
FNT = 0
T = 60
mar_threshold = 0.63
ear_avg_threshold = 0.21

## Nod parametes
N_nod = 0
F_nod = 0
F_prime_nod = 0
## Yawn parameters
N_yawn = 0
F_yawn = 0
F_prime_yawn = 0

## blink paraemtes
N_blink = 0
N_blink_perclose = 0
F_blink = 0
F_prime_blink = 0
perclose_value = 0
perclose_prime_value = 0

## emotion parameter
S_T = 0
N_emotion = 0
emotion_score_dict = {
    "angry" : 0.002,
    "disgust":0.001,
    "fear":0.003,
    "happy":-0.001,
    "neutral":0.000,
    "sad":0.002,
    "surprise":0.001
}

F = 0
S = 0


start_time = time.time()
start_time_ear = time.time()
start_time_mar = time.time()
start_time_nod = time.time()
start_time_emotion=time.time()

while True:
    
    success, image = cap.read()
    image_copy = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    FNT += 1

    current_time = time.time()
    elapsed_time = current_time - start_time

    



    report_img = np.zeros((550, 740, 3), dtype=np.uint8)

    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (xf,yf,wf,hf) in faces:
        cv2.rectangle(image,(xf,yf),(xf+wf,yf+hf),(0,255,0),2)
        roi_face = image[yf:yf+hf,xf:xf+wf]
        roi_face_copy = roi_face.copy()
        roi_face_copy = Fatigue.process_img(roi_face_copy)

        predictions = model.predict(roi_face_copy)
        predicted_emotion_class = np.argmax(predictions)
        predicted_emotion_label = label[predicted_emotion_class]

        current_time_emotion = time.time()
        elapsed_time_emotion = current_time_emotion - start_time_emotion
        if elapsed_time_emotion >= 1:
            N_emotion += 1
            S_T = (N_emotion * emotion_score_dict[predicted_emotion_label]) / T
            start_time_emotion = time.time()
            #print(S_T)


    if predicted_emotion_label in emotion_score_dict:
        emotion_score_value = N_emotion * emotion_score_dict[predicted_emotion_label]
        cv2.putText(report_img,f"EmotionScore: {emotion_score_value}",(20,500),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)

    else:
        print(f"Key '{predicted_emotion_label}' not found in emotion_score_dict.")


    image = cv2.flip(image,1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rects = detector(gray, 0)


    image.flags.writeable = False
    
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])       
            
            face_2d = np.array(face_2d, dtype=np.float64)

            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])


            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          


            if x < -10:
                current_time_nod = time.time()
                elapsed_time_nod = current_time_nod - start_time_nod
                if elapsed_time_nod >= 1:
                    text = "Looking Down"
                    N_nod += 1
                    F_nod = N_nod / T
                    F_prime_nod = Fatigue.normalize_fucntion(F_nod)
                start_time_nod = time.time()
                
            elif x > 10:
                current_time_nod = time.time()
                elapsed_time_nod = current_time_nod - start_time_nod
                if elapsed_time_nod >= 1:
                    text = "Looking Up"
                    N_nod += 1
                    F_nod = N_nod / T
                    F_prime_nod = Fatigue.normalize_fucntion(F_nod)
                start_time_nod = time.time()
            else:
                text = "Forward"

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        
        if face_mesh_draw:
            mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        #connections=mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
    


    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)


        x_p3_mouth,y_p3_mouth = shape[52]
        x_p5_mouth,y_p5_mouth = shape[56]
    
        x_p2_mouth,y_p2_mouth = shape[50]
        x_p6_mouth,y_p6_mouth = shape[58]
        
        x_p1_mouth,y_p1_mouth = shape[48]
        x_p4_mouth,y_p4_mouth = shape[54]

        
        x_p3_r, y_p3_r = shape[38]
        x_p5_r, y_p5_r = shape[40]
  
        x_p2_r,y_p2_r = shape[37]
        x_p6_r,y_p6_r = shape[41]
        
        x_p4_r,y_p4_r = shape[39]
        x_p1_r,y_p1_r = shape[36]
        
        x_p3_l,y_p3_l = shape[44]
        x_p5_l,y_p5_l = shape[46]
         
        x_p2_l,y_p2_l = shape[43]
        x_p6_l,y_p6_l = shape[47]
        
        x_p1_l,y_p1_l = shape[42]
        x_p4_l,y_p4_l = shape[45]


    p3_p5_mouth = Fatigue.distance((x_p3_mouth,y_p3_mouth),(x_p5_mouth,y_p5_mouth))
    p2_p6_mouth = Fatigue.distance((x_p3_mouth,y_p3_mouth),(x_p5_mouth,y_p5_mouth))
    p1_p4_mouth = Fatigue.distance((x_p1_mouth,y_p1_mouth),(x_p4_mouth,y_p4_mouth))

    p3_p5_right = Fatigue.distance((x_p3_r,y_p3_r),(x_p5_r,y_p5_r))
    p2_p6_right = Fatigue.distance((x_p2_r,y_p2_r),(x_p6_r,y_p6_r))
    p1_p4_right = Fatigue.distance((x_p1_r, y_p1_r),(x_p4_r, y_p4_r))
    EAR_RIGHT = Fatigue.ear_calcualtion(p3_p5_right,p2_p6_right,p1_p4_right)
    p3_p5_left = Fatigue.distance((x_p3_l,y_p3_l),(x_p5_l,y_p5_l))
    p2_p6_left = Fatigue.distance((x_p2_l,y_p2_l),(x_p6_l,y_p6_l))
    p1_p4_left = Fatigue.distance((x_p1_l,y_p1_l),(x_p4_l,y_p4_l))
    EAR_LEFT = Fatigue.ear_calcualtion(p3_p5_left,p2_p6_left,p1_p4_left)
        

    EAR_AVG = (EAR_LEFT + EAR_RIGHT) / 2
    if EAR_AVG <= ear_avg_threshold:
        N_blink_perclose+=1
        perclose_value = N_blink_perclose / FNT
        perclose_prime_value = Fatigue.normalize_fucntion(perclose_value)
        #print(perclose_value)
        current_time_ear = time.time()
        elapsed_time_ear = current_time_ear - start_time_ear
        if elapsed_time_ear >= 1:
            
            N_blink =N_blink+ 1
            F_blink = N_blink /  T
            F_prime_blink = Fatigue.normalize_fucntion(F_blink)
        #perclose_value = N_blink / FNT
        #perclose_prime_value = Fatigue.normalize_fucntion(perclose_value)
        start_time_ear = time.time()
    MAR = Fatigue.mar_calcualtion(p3_p5_mouth,p2_p6_mouth,p1_p4_mouth)

    if MAR >= mar_threshold:
        current_time_mar = time.time()
        elapsed_time_mar = current_time_mar- start_time_mar
        if elapsed_time_mar >= 1:
            N_yawn =N_yawn+ 1
            F_yawn = N_yawn / T
            print(F_yawn)
            F_prime_yawn = Fatigue.normalize_fucntion(F_yawn)
        start_time_mar = time.time()
    


    F = 0.5 * ((0.5 * F_prime_blink) + (0.3 * F_prime_yawn) + (0.2 * F_prime_nod) + (perclose_prime_value))

    S = (0.5 * S_T) + (0.5 * F)


    if S < 0.04 :
        # print("Normal")
        result_text = "Normal"
    
    elif S > 0.04 and S <= 0.05:
        # print("Mild Fatigue")
        result_text = "Mild Fatigue"
    
    elif S > 0.05 and S < 0.06:
        # print("Moderate Fatigue")
        result_text = "Moderate Fatigue"
    else:
        # print("Severe Fatigue")
        result_text = "Severe Fatigue"


    cv2.putText(report_img,f"Emotion: {predicted_emotion_label}",(20,50),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    cv2.putText(report_img,f"State: {result_text}",(20,100),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    cv2.putText(report_img,f"System time: {int(elapsed_time)}",(20,150),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    cv2.putText(report_img,f"Nb_blink: {N_blink}",(20,200),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    cv2.putText(report_img,f"Nb_Yawn: {N_yawn}",(20,250),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    cv2.putText(report_img,f"Nb_Nod: {N_nod}",(20,300),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    cv2.putText(report_img,f"Perclose: {(int(perclose_prime_value *100))}%",(20,350),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    cv2.putText(report_img,f"Indicators: {F}",(20,400),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    cv2.putText(report_img,f"Status Indicators: {S}",(20,450),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)


    ctime_new= time.time()
    fps= 1/(ctime_new- ptime_new)
    ptime_new = ctime_new

    cv2.putText(image, "FPS: {}".format(int(fps)), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    # file= open("report_fatigue.txt",'a+')
    # file_data = file.write(
    #                          f"NbFrame: {FNT} \n"
    #                          f"Emotion: {predicted_emotion_label} \n"
    #                          f"State: {result_text} \n"
    #                          f"Nb_blink: {N_blink} \n"
    #                          f"Nb_Yawn: {N_yawn} \n"
    #                          f"Nb_Nod: {N_nod} \n"
    #                          f"Perclose: {(int(perclose_value *100))}% \n"
    #                          f"Indicators: {F} \n"
    #                          f"Status Indicators: {S} \n"
    #                          f"EmotionScore: {N_emotion*emotion_score_dict[predicted_emotion_label]} \n"
    #                          "----------------------------------------------------------------- \n"
    #                         )



    
    if elapsed_time >= 500:
        FNT = 0
        N_nod = 0
        F_nod = 0
        F_prime_nod = 0
        N_yawn = 0
        F_yawn = 0
        F_prime_yawn = 0
        N_blink = 0
        F_blink = 0
        F_prime_blink = 0
        perclose_value = 0
        perclose_prime_value = 0
        N_blink_perclose=0
        S_T = 0
        N_emotion = 0
        F = 0
        S = 0
        print("System Reset")
        start_time = time.time() 

    cv2.imshow("Fatigue Report", report_img)
    cv2.imshow('Fatigue Detection', image)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

