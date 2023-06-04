# ROS2 Object Detection with YOLO

This repository contains a ROS2 package that implements object detection using YOLO (You Only Look Once). This package is designed to detect various objects in a scene. The object detection model included in this package are derived from:

1. [A deep learning based hazardous materials (HAZMAT) sign detection robot with restricted computational resources](https://www.sciencedirect.com/science/article/pii/S2666827021000529) by Amir Sharifi, Ahmadreza Zibaei, and Mahdi Rezaei.

2. [FireNet](https://rdr.ucl.ac.uk/articles/dataset/FireNet/9137798) by Jan Boehm, Fabio Panella, and Victor Melatti.

## Installation

Please follow these steps to install the ROS2 Object Detection package.

1. Clone the repository into your workspace:
   ```bash
   git clone https://github.com/YourUsername/ros2_object_detection.git
   ```
   
Note: After cloning change line 49 in the publisher script to point to your desired local model file!

2. Navigate to the cloned directory:
   ```bash
   cd ros2_object_detection
   ```
3. Build the package:
   ```bash
   colcon build
   ```
4. Source the setup file:
   ```bash
   source install/setup.bash
   ```
   
## Usage

Once you have installed the package, you can start detecting objects by launching the object_detection node:

```bash
ros2 run ros2_object_detection detection_publisher
```

The publisher publishes two topics: /video_frames and /video_frames_annotated. If you want to change the topic in the subscriber, please change line 31 in the source file.

If you do not have Rviz running to detect the published topic, you can use the provided subscriber to test the package

```bash
ros2 run ros2_object_detection detection_subscriber
```

## Citation

If you use this package in your research, please cite it as follows:

```bibtex
@misc{ hazmatfirerescue_dataset,
    title = { HazmatFireRescue Dataset },
    type = { Open Source Dataset },
    author = { Daniel Kleissl},
    howpublished = { \url{ https://universe.roboflow.com/fhcampuswienkleissl/hazmatfirerescue } },
    url = { https://universe.roboflow.com/fhcampuswienkleissl/hazmatfirerescue },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2023 },
    month = { jun },
    note = { visited on 2023-06-04 },
}
```

## Contributing

I welcome contributions to the ROS2 Object Detection package. Please submit pull requests and/or open issues on the GitHub repository.

## License

This package is released under the [MIT License](LICENSE).
 
