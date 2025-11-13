#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 启动参数定义
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='inference',
        description='运行模式: serialize(序列化模型) 或 inference(推理)'
    )
    
    wts_path_arg = DeclareLaunchArgument(
        'wts_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('trt_yolo'),
            'yolo11',
            'yolo11n-seg.wts'
        ]),
        description='WTS权重文件路径(序列化模式需要)'
    )
    
    engine_path_arg = DeclareLaunchArgument(
        'engine_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('trt_yolo'),
            'yolo11',
            'yolo11n-seg.engine'
        ]),
        description='TensorRT引擎文件路径'
    )
    
    # model_type_arg = DeclareLaunchArgument(
    #     'model_type',
    #     default_value='n',
    #     description='模型类型: n/s/m/l/x (nano/small/medium/large/xlarge)'
    # )

    labels_file_arg = DeclareLaunchArgument(
        'labels_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('trt_yolo'),
            'yolo11',
            'coco.txt'
        ]),
        description='coco标签文件'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='输入图像话题'
    )
    
    cuda_post_process_arg = DeclareLaunchArgument(
        'cuda_post_process',
        default_value='c',
        description='后处理方式: c(CPU) 或 g(GPU)'
    )
    
    publish_image_arg = DeclareLaunchArgument(
        'publish_image',
        default_value='true',
        description='是否发布带检测框的图像'
    )
    
    yolo11_node = Node(
        package='trt_yolo',
        executable='yolo11_seg',
        name='yolo11_detector',
        output='screen',
        parameters=[{
            'mode': LaunchConfiguration('mode'),
            'wts_path': LaunchConfiguration('wts_path'),
            'engine_path': LaunchConfiguration('engine_path'),
            # 'model_type': LaunchConfiguration('model_type'),
            'labels_file': LaunchConfiguration('labels_file'),
            'cuda_post_process': LaunchConfiguration('cuda_post_process'),
            'image_topic': LaunchConfiguration('image_topic'),
            'output_topic': '/yolo11/detections',
            'publish_image': LaunchConfiguration('publish_image'),
        }],
        # 显式设置参数类型
        arguments=['--ros-args', '--log-level', 'info']
    )

    return LaunchDescription([
        mode_arg,
        wts_path_arg,
        engine_path_arg,
        # model_type_arg,
        labels_file_arg,
        image_topic_arg,
        cuda_post_process_arg,
        publish_image_arg,
        yolo11_node,
    ])