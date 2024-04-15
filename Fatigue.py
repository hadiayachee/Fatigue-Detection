import math
import cv2
import numpy as np


class Fatigue():
    def normalize_fucntion(F):
        F_prime = math.atan(F)
        pi_value = math.pi
        F_prime = F_prime * (2 / pi_value)
        return F_prime
    

    def distance(point_1,point_2):
        dist = sum([(i-j) ** 2 for i,j in zip(point_1,point_2)]) ** 0.5
        return dist


    def mar_calcualtion(vertical_first_line,vertical_second_line,horizantal_line):
        mar_value = (vertical_first_line + vertical_second_line) / (2 * horizantal_line)
        mar_value = round(mar_value,2)
        return mar_value
    

    def ear_calcualtion(vertical_first_line,vertical_second_line,horizantal_line):
        ear_value = (vertical_first_line + vertical_second_line) / (2 * horizantal_line)
        ear_value = round(ear_value,2)
        return ear_value
    


    def process_img(img):
        img = cv2.resize(img, (48, 48))  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = np.reshape(img, (1, 48, 48, 1))
        return img

