#https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

from function.function import *
from function.funcAP import *

printToTxt()
printToAP()
true_pos = 0
false_pos = 0
false_neg = 0
threshold = 0.5


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#na vypocet false_negatives
for imagePath in paths.list_images(args["images"]):
    variableFind = 0 # pomocna s ktorou preskocim do elif
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    image = cv2.imread(imagePath)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0,
         (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()


    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(startX, startY, endX, endY)

            variableFind += 1
        #TODO zistovanie false_negative
        elif (i == 199 and variableFind == 0):
            false_neg += 1

print("FALSE NEGATIVE ",false_neg)

for imagePath in paths.list_images(args["images"]):
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image = cv2.imread(imagePath)

    (h, w) = image.shape[:2]
    # blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    #    (300, 300), (104.0, 177.0, 123.0))
    blob = cv2.dnn.blobFromImage(image, 1.0,
         (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()


    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #hodnoty Ground-truth BB
            photo_name = os.path.basename(imagePath)
            line = readFromTxt(photo_name)
            x_r = int(line[1])
            y_r = int(line[2])
            w_r = int(line[3])
            h_r = int(line[4])
            x2_r = x_r + w_r
            y2_r = y_r + h_r

            # draw the bounding box of the face along with the associated
            # probability, TODO prediction BB, red box
            # text = "{:.2f}%".format(confidence * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(image, (startX, startY), (endX, endY),
            #               (0, 0, 255), 2)
            # cv2.putText(image, text, (startX, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # #TODO read rectangle from file Ground-truth BB, green box
            # cv2.rectangle(image, (x_r, y_r), (x2_r, y2_r),
            #               (0, 255, 0), 2)


            boxA = [startX,startY,endX,endY]
            boxB = [x_r, y_r, x2_r, y2_r]
            # compute the intersection over union and display it
            iou = bb_intersection_over_union(boxA, boxB)
            # cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # vypocet Precision a Recall
            if (iou >= threshold):
                true_pos += 1
            elif (iou < threshold):
                false_pos += 1

            calcPrecisionRecall(photo_name,true_pos,false_pos,false_neg,1)

            # show the output image
            width = endX - startX
            height = endY - startY

            #vyprintovanie do konzoly
            print(photo_name)
            print("Prediction BB ", startX,startY,endX,endY)
            print("Ground truth BB ", x_r,y_r,x2_r,y2_r)


            #vyprintovanie do priecinku image.txt
            appendToTxt(photo_name,startX,startY,width,height)

            # show the output image
            # cv2.imshow("Output", image)
            # cv2.waitKey(0)
            break

print("FALSE NEGATIVEs ",false_neg)
print("TRUE POSITIVES ", true_pos)
print("FALSE POSITIVES ",false_pos)
plot_model(precisionArray,recallArray)

#python detect_faces.py -i C:\Users\Lenovo\Desktop\bakalarka\blurPhotos\foto7.jpg -p C:\Users\Lenovo\PycharmProjects\faceDetection\opencv\sa
#mples\dnn\face_detector\deploy.prototxt -m C:\Users\Lenovo\PycharmProjects\faceDetection\res10_300x300_ssd_iter_140000.caffemodel

#pre viac fotiek na nacitanie zo suboru
#python face_position.py -i C:\Users\Lenovo\Desktop\bakalarka\faces -p C:\Users\Lenovo\PycharmProjects\faceDetection\opencv\samples\dnn\
#face_detector\deploy.prototxt -m C:\Users\Lenovo\PycharmProjects\faceDetection\res10_300x300_ssd_iter_140000.caffemodel

# python main.py -i C:\Users\Lenovo\Desktop\bakalarka\Celeb -p C:\Users\Lenovo\PycharmProjects\avaragePrediction_2\deploy.prototxt
# -m C:\Users\Lenovo\PycharmProjects\avaragePrediction_2\res10_300x300_ssd_iter_140000.caffemodel