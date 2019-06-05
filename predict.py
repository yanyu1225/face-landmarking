# -*- coding: UTF-8 -*-

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
def getCoordinate(path,width):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(path)
    image = imutils.resize(image, width=width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    return shape,image
def draw(img,f1,f2,f3,f4,i):
    cv2.putText(img,"F1:" +str(f1), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 0)
    cv2.putText(img, "F2:"+str(f2), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 0)
    cv2.putText(img, "F3:"+str(f3), (20, 60), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255), 0)
    cv2.putText(img, "F4:"+str(f4), (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 0)
    if not os.path.exists("result"):
        os.mkdir("result")
    cv2.imwrite("./result/" + str(i) + '.jpg', img)  # 存储为图像
    cv2.imshow("img",img)
    cv2.waitKey(1)


def distance(coord1,coord2):
    point1_x = coord1[0]
    point1_y = coord1[1]
    point2_x = coord2[0]
    point2_y = coord2[1]
    distance = ((point1_x - point2_x) ** 2 + (point1_y - point2_y) ** 2) ** 0.5
    return distance
def judge(feature1,feature2,feature3,feature4):
    print(feature1)
    print(feature2)
    print(feature3)
    print(feature4)
def main(path,width=500):
    File = os.listdir(path)
    i = 0
    for file in File:
        i +=1
        imgPath = os.path.join(path,file)

        coordinate, image = getCoordinate(imgPath, width)


        Feature11 = coordinate[33]
        Feature12 = coordinate[51]

        Feature21 = coordinate[36]
        Feature22 = coordinate[39]

        Feature31 = coordinate[51]
        Feature32 = coordinate[62]

        Feature41 = coordinate[37]
        Feature42 = coordinate[51]

        feature1 = distance(Feature11, Feature12)
        feature2 = distance(Feature21, Feature22)
        feature3 = distance(Feature31, Feature32)
        point1 = Feature41[1]
        point2 = Feature42[1]
        feature4 = np.abs(point1 - point2)
        # judge(feature1, feature2, feature3, feature4)
        draw(image, feature1, feature2, feature3, feature4,i)

    # judge(feature1,feature2,feature3,feature4)



if __name__ == '__main__':
    main("data")



