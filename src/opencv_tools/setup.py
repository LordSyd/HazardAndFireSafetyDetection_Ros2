from setuptools import setup

package_name = 'opencv_tools_yolo_obj_detection'

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
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'img_publisher = opencv_tools.basic_image_publisher:main',
            'img_subscriber = opencv_tools.basic_image_subscriber:main'
        ],
    },
)
