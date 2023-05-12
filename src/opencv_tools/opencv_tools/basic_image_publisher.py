# Basic ROS 2 program to publish real-time streaming
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
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

        self.net_ = cv2.dnn.readNetFromONNX('/home/fhcampus01/Documents/ros2_opencv/best.onnx')
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

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        self.net_.setInput(blob)
        preds = self.net_.forward()
        preds = preds.transpose((0, 2, 1))

        # Extract output detection
        class_ids, confs, boxes = list(), list(), list()

        image_height, image_width, _ = frame.shape
        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT

        rows = preds[0].shape[0]

        for i in range(rows):
            row = preds[0][i]
            conf = row[4]

            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            print(classes_score[8])
            if (classes_score[class_id] > .25):
                confs.append(conf)
                label = CLASESS_YOLO[int(class_id)]
                class_ids.append(label)

                # extract boxes
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = list(), list(), list()

        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45)
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i])
            r_boxes.append(boxes[i])

        for i in indexes:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 3)


        if ret == True:
            # Publish the image.
            # The 'cv2_to_imgmsg' method converts an OpenCV
            # image to a ROS 2 image message
            self.publisher_.publish(self.br.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

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