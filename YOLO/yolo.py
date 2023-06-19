# -*- coding: utf-8 -*-
"""
Created on Thu May 25 23:41:46 2023

@author: Houdini69
"""

import cv2
import numpy as np
# define the minimum confidence (to filter weak detections),
# Non-Maximum Suppression (NMS) threshold, and the green color
ct=[0.5,0.3] #confidence_thresh
nt=[0.3,0.4] #NMS_thresh
ycyw=[["yolov3-config/yolov3.cfg","yolov3-config/yolov3.weights"],["yolov3-spp-config/yolov3-spp.cfg","yolov3-spp-config/yolov3-spp.weights"]] #yolo_config and yolo_weights
for c in ct:
    for n in nt:
        for yy in ycyw:
            confidence_thresh=c
            NMS_thresh=n
            red = (0, 0, 255)
            # Load the image and get its dimensions
            image = cv2.imread("examples/images/30.jpg")
            # resize the image to 25% of its original size
            image = cv2.resize(image,(int(image.shape[0] * 0.5),int(image.shape[1] * 0.5)))
            # get the image dimensions
            h = image.shape[0]
            w = image.shape[1]
            # load the class labels the model was trained on
            classes_path= "yolov3-config/coco.names"
            with open(classes_path, "r") as f:
                classes = f.read().strip().split("\n")
            # load the configuration and weights from disk
            yolo_config=yy[0]
            yolo_weights=yy[1]
            # load YOLOv3 network pre-trained on the COCO dataset
            net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            # Get the name of all the layers in the network
            layer_names= net.getLayerNames()
            # Get the names of the output layers
            output_layers= [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
            # create a blob from the image
            blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
            # pass the blob through the network and get the output predictions
            net.setInput(blob)
            outputs = net.forward(output_layers)
            # create empty lists for storing the bounding boxes, confidences, and class IDs
            boxes,confidences,class_ids= [],[],[]
            for output in outputs: # loop over the output predictions
                for detection in output: # loop over the detections
            # get the class ID and confidence of the detected object
                    scores = detection[5:]
                    class_id= np.argmax(scores)
                    confidence = scores[class_id]
                    # we keep the bounding boxes if the confidence (i.e. class probability) is greater than the minimum confidence
                    if confidence > confidence_thresh:
                        # perform element-wise multiplication to get the coordinates of the bounding box
                        box = [int(a * b) for a, b in zip(detection[0:4], [w, h, w, h])]
                        center_x, center_y, width, height = box
                        # get the top-left corner of the bounding box
                        x = int(center_x-(width / 2))
                        y = int(center_y-(height / 2))
                        # append the bounding box, confidence, and class ID to their respective lists
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, width, height])
                        # draw the bounding boxes on a copy of the original image before applying non-maxima suppression
            image_copy= image.copy()
            for box in boxes:
                x, y, width, height = box
                cv2.rectangle(image_copy, (x, y), (x + width, y + height), red, 2)
            # show the output image
            cv2.imshow("Before NMS", image_copy)
            cv2.waitKey(0)
            # apply non-maximum suppression to remove weak bounding boxes that overlap with others.
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, NMS_thresh)
            indices = indices.flatten()
            for i in indices:
                (x, y, w, h) = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                cv2.rectangle(image, (x, y), (x + w, y + h), red, 2)
                text = f"{classes[class_ids[i]]}: {confidences[i] * 100:.2f}%"
                cv2.putText(image, text, (x, y -15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
            # show the output image
            print("confidence_thresh="+str(c)+",NMS_thresh="+str(n)+",yolo_config="+yy[0]+",yolo_weights="+yy[1])
            cv2.imshow("After NMS", image)
            cv2.waitKey(0)


