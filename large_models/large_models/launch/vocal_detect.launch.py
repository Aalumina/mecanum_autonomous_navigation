import os
from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.actions import OpaqueFunction, DeclareLaunchArgument

def launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration('mode', default=1)
    enable_wakeup = LaunchConfiguration('enable_wakeup', default='true')
    awake_method = LaunchConfiguration('awake_method', default=os.environ['MIC_TYPE'])
    chinese_awake_words = LaunchConfiguration('chinese_awake_words', default='xiao3 huan4 xiao3 huan4')
    enable_setting = LaunchConfiguration('enable_setting', default='false')
    
    mode_arg = DeclareLaunchArgument('mode', default_value=mode)
    enable_wakeup_arg = DeclareLaunchArgument('enable_wakeup', default_value=enable_wakeup) 
    awake_method_arg = DeclareLaunchArgument('awake_method', default_value=awake_method)
    awake_words_arg = DeclareLaunchArgument('chinese_awake_words', default_value=chinese_awake_words)
    enable_setting_arg = DeclareLaunchArgument('enable_setting', default_value=enable_setting)

    vocal_detect_node = Node(
        package='large_models',
        executable='vocal_detect',
        output='screen',
        parameters=[{"port": "/dev/ring_mic",
                     "mic_type": "mic6_circle",
                     "awake_method": awake_method,
                     "awake_word": chinese_awake_words,
                     "enable_setting": enable_setting,
                     "enable_wakeup": enable_wakeup,
                     "mode": mode}]
    )

    return [
            mode_arg,
            enable_wakeup_arg,
            awake_method_arg,
            awake_words_arg,
            enable_setting_arg,
            vocal_detect_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # Create a LaunchDescription object. (创建一个LaunchDescription对象)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
