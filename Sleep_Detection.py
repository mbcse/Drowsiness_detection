from scipy.spatial import distance as dist

from imutils.video import FileVideoStream

from imutils.video import VideoStream

from imutils import face_utils

import numpy as np

from imutils import face_utils

import argparse

import imutils

import time

import dlib

import cv2

import pyttsx3

import sys
#sys.argv = ['Sleed_detection.py','--shape_predictor_68_face_landmarks.dat']
#exec(open('Sleed_detection.py').read())

engine = pyttsx3.init()

rate = engine.getProperty('rate')

engine.setProperty('rate',rate - 50)



def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])

    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])



    #Compute eye aspect ratio

    ear = (A+B)/(2*C)



    return ear



def mouth_aspect_ratio(mouth):

    A = dist.euclidean(mouth[2],mouth[10])

    B = dist.euclidean(mouth[3],mouth[9])

    C = dist.euclidean(mouth[4],mouth[8])



    mar = (A+B+C)/3



    return mar

def jaw_aspect_ratio(jaw,left_eyebrow,right_eyebrow):

    A=dist.euclidean(jaw[7],left_eyebrow[3])

    B=dist.euclidean(jaw[9],right_eyebrow[3])

    jar=(A+B)/2

    return jar



blink_time=0
blinks_time=0
blinke_time=0



ap = argparse.ArgumentParser()

ap.add_argument("-p", "--shape-predictor", required=True,

	help="path to facial landmark predictor")

ap.add_argument("-v", "--video", type=str, default="",

	help="path to input video file")

args = vars(ap.parse_args())



EYE_AR_THRESH = 0.23  #threshold for blink

EYE_AR_CONSEC_FRAMES = 25 #consecutive considered true

sleep_flag = 0

yawn_flag = 0

count_mouth = 0



counter = 0

total = 0

total_yawn = 0



print("[INFO] loading facial landmark predictor...")

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(args["shape_predictor"])



(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

(bStart, bEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

(cStart, cEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]






print("[INFO] starting video stream thread...")

vs = FileVideoStream(args["video"]).start()

fileStream = True

vs = VideoStream(src=0).start()

#vs = VideoStream('http://10.12.2.195:8008/video').start()

time.sleep(1.0)

start_time = time.time()

elapsed_time = start_time





while True:

    # if fileStream and not vs.more():

    #     break



    frame = vs.read()
    #cv2.imshow('frame',frame)
    frame = imutils.resize(frame, width = 640)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    rects = detector(gray,0)    #dlib’s built-in face detector.



    for rect in rects:

        shape = predictor(gray, rect)

        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]

        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)

        rightEAR = eye_aspect_ratio(rightEye)

        mouth = shape[mStart: mEnd]

        jaw=shape[jStart:jEnd]

        left_eyebrow=shape[bStart:bEnd]
        right_eyebrow=shape[cStart:cEnd]

        if(len(jaw)!=1):

            mouthEAR = mouth_aspect_ratio(mouth)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)



            left_eyebrow=shape[bStart:bEnd]
            right_eyebrow=shape[cStart:cEnd]

            jawEAR= jaw_aspect_ratio(jaw,left_eyebrow,right_eyebrow)





            jawHull=cv2.convexHull(jaw)

            cv2.drawContours(frame, [jawHull], -1   , (0, 255, 0), 1)



        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)

        rightEyeHull = cv2.convexHull(rightEye)

        #mouthHull = cv2.convexHull(mouth)

    #    jawHull=cv2.convexHull(jaw)

        left_eyebrowHull=cv2.convexHull(left_eyebrow)
        right_eyebrowHull=cv2.convexHull(right_eyebrow)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)

        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

#        cv2.drawContours(frame, [jawHull], -1   , (0, 255, 0), 1)

        cv2.drawContours(frame, [right_eyebrowHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [left_eyebrowHull], -1, (0, 255, 0), 1)


        if ((mouthEAR > 30) or (jawEAR > 140)):

            count_mouth += 1

            if count_mouth >= 10:

                if yawn_flag < 0:

                    print("You are yawning")

                    #time.sleep(4.0)

                    yawn_flag = 1

                    total_yawn += 1

                    cv2.putText(frame, "Yawn Detected", (150, 150),

            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                else:

                    yawn_flag = 1

            else:

                yawn_flag = -1

        else:

            count_mouth = 0

            yawn_flag = -1

        # else:

        #     print("You are working")

        if ear <  EYE_AR_THRESH:

            blinks_time=time.time()

            counter += 1
            if(ear> EYE_AR_THRESH):

                blinke_time=time.time()
                blink_time=blinks_time-blinke_time




            if (counter >= EYE_AR_CONSEC_FRAMES)   :
                #blinks_time=0

                if (sleep_flag < 0)  :

                    print("You are sleeping.")
                    #time.sleep(2.0)

                    cv2.putText(frame, "Sleep Detected", (150, 150),

            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    sleep_flag = 1

                    total += 1

            else:

                sleep_flag = -1

            # if total > 0:

            #     elapsed_time  = time.time() - start_time

            #     if elapsed_time > 10:

            #         print("You are sleeping dear")

            #         start_time = time.time()

            #     else:

            #         print("You are working")



        else:

            counter = 0

            sleep_flag = -1



        cv2.putText(frame, "Total Sleep: {}".format(total), (10, 30),

            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, "Total Yawns: {}".format(total_yawn), (10, 70),

            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),

			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, "MAR: {:.2f}".format(mouthEAR), (540, 30),

			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, "JAR: {:.2f}".format(jawEAR), (200, 30),

    		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if (total+total_yawn > 4)  :

            print("playing sound")

            engine.say("You are sleeping. I recommend you going for a walk or listen to music")

            engine.runAndWait()

            total = 0

            total_yawn = 0



    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break



# do a bit of cleanup

cv2.destroyAllWindows()

vs.stop()
