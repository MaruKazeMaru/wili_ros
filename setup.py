from glob import glob
from setuptools import find_packages, setup

package_name = 'wili_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.launch.py')),
        ('share/' + package_name, glob('config/*.param.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shinagawa Kazemaru',
    maintainer_email='marukazemaru0@gmail.com',
    description='ROS2 package for WiLI',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'db_agent=wili_ros.db_agent:main',
            'hmm=wili_ros.hmm_node:main',
            'suggester=wili_ros.suggester_node:main',
            'obs_stacker=wili_ros.obs_stacker:main',
            'apriltag=wili_ros.apriltag_node:main',
        ],
    },
)
