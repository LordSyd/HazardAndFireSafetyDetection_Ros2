from setuptools import setup

package_name = 'opencv_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fhcampus01',
    maintainer_email='fhcampus01@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            #'img_publisher = opencv_tools.detect:main',
            'img_publisher = opencv_tools.basic_image_publisher3:main',
            'img_subscriber = opencv_tools.basic_image_subscriber:main'
        ],
    },
)
