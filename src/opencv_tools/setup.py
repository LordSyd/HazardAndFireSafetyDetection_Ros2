from setuptools import setup

package_name = 'ros2_object_detection_yolo'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daniel Kleissl',
    maintainer_email='daniel.kleissl@stud.fh-campuswien.ac.at',
    description='Proof of Concept ROS2 object detection publisher and subscriber node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_publisher = ros2_object_detection.basic_image_publisher:main',
            'detection_subscriber = ros2_object_detection.basic_image_subscriber:main'
        ],
    },
)
