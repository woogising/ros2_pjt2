from setuptools import find_packages, setup
import glob
import os

package_name = 'a4_cobot2'

setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(include=[
        'voice',
        'voice.*',
        'task_manager',
        'task_manager.*',
        'workspace',
        'workspace.*',
        'robot_arm',
        'robot_arm.*',
        'safety',
        'safety.*',
        'notification',
        'notification.*',
        'object_detection',
        'object_detection.*',
        'database',
        'database.*',        
        'hmi',
        'hmi.*',
    ]),

    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # resource 폴더의 일반 파일 설치
        (
            os.path.join('share', package_name, 'resource'),
            glob.glob('resource/*')
        ),

        # .env는 숨김 파일이라 glob('resource/*')에 포함되지 않으므로 직접 설치
        (
            os.path.join('share', package_name, 'resource'),
            ['resource/.env']
        ),

        # launch 파일 설치
        (
            os.path.join('share', package_name, 'launch'),
            glob.glob('launch/*.launch.py')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey4090',
    maintainer_email='rokey4090@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',

    entry_points={
        'console_scripts': [
            'command_input_node = voice.command_input_node:main',
            'task_manager_node = task_manager.task_manager_node:main',
            'workspace_judge_node = workspace.workspace_judge_node:main',
            'robot_arm_node = robot_arm.robot_arm_node:main',
            'safety_node = safety.safety_node:main',
            'status_notifier_node = notification.status_notifier_node:main',
            'vlm_report_node = notification.vlm_report_node:main',
            'object_detection_node = object_detection.detection:main',
        ],
    },
)




