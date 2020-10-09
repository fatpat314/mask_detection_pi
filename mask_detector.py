from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import imutils
import cv2
import os
import time


def mask_detector(frame,faceNet,maskNet):
    #grab the dimensions of the frame and then construct a blob from it

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape, "Yo mang")

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locations = []
    predictions = []

    # loop over detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5: #0.5
            # compute the (x, y)-coordinates of the bounding box for the objects
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding box fall within the dimensions of the frame
            (startX, startY) = (max(0,startX), max(0, startY))
            (endX, endY) = (min(width - 1, endX), min(height - 1, endY))


            # extract the face ROI, convert it to BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to thir respective lists
            faces.append(face)
            locations.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather then one-by-one predictions
        # in the above 'for' loop
        faces = np.array(faces, dtype="float32")
        predictions = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding locations
    return (locations, predictions)

# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


def camera_stream():
    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    # count = 0
    # mask_count = 0
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixles
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a mask or not
        (locations, predictions) = mask_detector(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding locations
        for (box, predictions) in zip(locations, predictions):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = predictions

            label = "Mask"
            if mask > withoutMask:
                color = (0, 255, 0)
            elif withoutMask > 0.99:
                label = "No Mask"

            if label == "Mask":
                color = (0, 255, 0)
                # mask_count += 1
                # if mask_count == 10:
                if mask > 0.999
                    playsound("mask.wav")
                    # os.system("vcgencmd display_power 0")
                # elif mask_count > 10:
                    # pass

            elif withoutMask > 0.99:
                color = (0, 0, 255)
                # count += 1
                # if count == 10:
                playsound("no_mask.wav")
                # os.system("vcgencmd display_power 1")
                # elif count > 10:
                # count = 0
                # mask_count = 0

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if key == ord("q"):
            break

camera_stream()
# clean up
cv2.destroyAllWindows()
vs.stop()
