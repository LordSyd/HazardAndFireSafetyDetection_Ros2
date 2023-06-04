# Basic ROS 2 program to publish real-time streaming
# video from your built-in webcam combined with YOLOv8 object detection
# Author: Daniel Kleissl
# Adopted from Code by:
# - Addison Sears-Collins
# - https://automaticaddison.com
#
# - Ultralytics YOLOv8 Docs
# - https://docs.ultralytics.com/modes/predict/#streaming-source-for-loop

import time

import numpy as np
# Import the necessary libraries
import rclpy  # Python Client Library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library

# Define yolov8 classes
CLASSES = ['Alarm_Activator', 'Fire_Blanket', 'Fire_Exit', 'Fire_Extinguisher', 'Fire_Suppression_Signage', 'corrosive',
           'dangerous', 'explosive', 'flammable', 'flammable-solid', 'infectious-substance', 'inhalation-hazard',
           'non-flammable-gas', 'organic-peroxide', 'oxygen', 'poison', 'radioactive', 'spontaneously-combustible']

# Get a color for the bounding box
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

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
        # to the video_frames and video_frames_annotated topic. The queue size is 10 messages.
        # This is done to allow viewing the original webcam stream if inference is slow
        self.publisher_ = self.create_publisher(Image, 'video_frames_annotated', 10)
        self.publisher_original_ = self.create_publisher(Image, 'video_frames', 10)

        # load trained model into OpenCV for inference
        self.net_ = cv2.dnn.readNet(
            '/home/fhcampus01/Documents/GitHub/HazardAndFireSafetyDetection_Ros2/src/opencv_tools/opencv_tools/models/yoloV8nCustom.onnx')
        #self.net_.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

        # We will try to publish a message every 0.05 seconds
        timer_period = 0.05  # seconds

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

        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        ret, frame = self.cap.read()


        def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
            label = f'{CLASSES[class_id]} ({confidence:.2f})'
            color = colors[class_id]
            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
            cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if ret == True:

            #Publish original video stream
            self.publisher_original_.publish(self.br.cv2_to_imgmsg(frame))

            input_image = frame
            model: cv2.dnn.Net = self.net_
            original_image: np.ndarray = input_image
            [height, width, _] = original_image.shape
            length = max((height, width))
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = original_image
            scale = length / 640

            blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
            model.setInput(blob)

            #start inference
            outputs = model.forward()

            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []

            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                if maxScore >= 0.66:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2], outputs[0][i][3]]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

            detections = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    'class_id': class_ids[index],
                    'class_name': CLASSES[class_ids[index]],
                    'confidence': scores[index],
                    'box': box,
                    'scale': scale}
                detections.append(detection)
                draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale),
                                  round(box[1] * scale),
                                  round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

            # Publish the image.
            # The 'cv2_to_imgmsg' method converts an OpenCV
            # image to a ROS 2 image message
            self.publisher_.publish(self.br.cv2_to_imgmsg(original_image))

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