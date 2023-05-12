# Basic ROS 2 program to publish real-time streaming
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
import time

import numpy as np
# Import the necessary libraries
import rclpy  # Python Client Library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library
from PIL import Image as Img
from PIL import ImageTk

class ImagePublisher(Node):
    """
    Create an ImagePublisher class, which is a subclass of the Node class.
    """

    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('image_publisher')

        # Create the publisher. This publisher will publish an Image
        # to the video_frames topic. The queue size is 10 messages.
        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)

        self.net_ = cv2.dnn.readNet('/home/fhcampus01/Documents/ros2_opencv/best.onnx')
        #self.net_.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)



        # We will publish a message every 0.1 seconds
        timer_period = 0.1  # seconds

        # Create the timer
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Create a VideoCapture object
        # The argument '0' gets the default webcam.
        self.cap = cv2.VideoCapture(0)

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

    def timer_callback(self):
        """
        Callback function.
        This function gets called every 0.1 seconds.
        """
        INPUT_WIDTH = 640
        INPUT_HEIGHT = 640
        SCORE_THRESHOLD = 0.2
        NMS_THRESHOLD = 0.4
        CONFIDENCE_THRESHOLD = 0.4

        # Define yolov8 classes
        CLASESS_YOLO = ["poison",
                        "oxygen", "flammable", "flammable-solid", "corrosive", "dangerous", "non-flammable-gas",
                        "organic-peroxide", "explosive", "radioactive", "inhalation-hazard",
                        "spontaneously-combustible", "infectious-substance"
                        ]
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        ret, frame = self.cap.read()

        # Load names of classes and get random colors
        #classes = open('coco.names').read().strip().split('\n')
        classes = CLASESS_YOLO
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

        # Give the configuration and weight files for the model and load the network.
        net = self.net_
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # determine the output layer
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        r = blob[0, 0, :, :]

        #cv.imshow('blob', r)
        text = f'Blob shape={blob.shape}'
        #cv2.displayOverlay('blob', text)
        #cv.waitKey(1)

        net.setInput(blob)
        t0 = time.time()
        outputs = net.forward(ln)
        t = time.time()
        print('time=', t - t0)

        print(len(outputs))
        for out in outputs:
            print(out.shape)

        def trackbar2(x):
            confidence = x / 100
            r = r0.copy()
            for output in np.vstack(outputs):
                if output[4] > confidence:
                    x, y, w, h = output[:4]
                    p0 = int((x - w / 2) * 416), int((y - h / 2) * 416)
                    p1 = int((x + w / 2) * 416), int((y + h / 2) * 416)
                    cv2.rectangle(r, p0, p1, 1, 1)
            #cv.imshow('blob', r)
            text = f'Bbox confidence={confidence}'
            cv2.displayOverlay('blob', text)

        r0 = blob[0, 0, :, :]
        r = r0.copy()
        #cv2.imshow('blob', r)
        #cv2.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
        #trackbar2(50)

        boxes = []
        confidences = []
        classIDs = []
        h, w = frame.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                #print(np.argmax(scores))
                #classID = np.argmax(scores)
                _, _, _, max_idx = cv2.minMaxLoc(scores)
                classID = max_idx[1]
                confidence = scores[classID]
                print(confidence)
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if ret == True:
            # Publish the image.
            # The 'cv2_to_imgmsg' method converts an OpenCV
            # image to a ROS 2 image message
            self.publisher_.publish(self.br.cv2_to_imgmsg(frame))

        # Display the message on the console
        self.get_logger().info('Publishing video frame')


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_publisher = ImagePublisher()

    # Spin the node so the callback function is called.
    rclpy.spin(image_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_publisher.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == '__main__':
    main()