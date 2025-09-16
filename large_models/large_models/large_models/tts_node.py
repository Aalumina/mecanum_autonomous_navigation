#!/usr/bin/env python3
# encoding: utf-8
# @Author: Lumina
# @Date: 2025/4/18
import os
import time
import rclpy
import threading
from rclpy.node import Node
from std_srvs.srv import Trigger, Empty
from std_msgs.msg import String, Bool

from speech import speech
from large_models.config import *

class TTSNode(Node):
    """

    Text-to-Speech node, responsible for converting text to speech and playing it(文本转语音节点，负责将文本转换为语音并播放)
    """
    def __init__(self, name):
        """
        Initialize the TTS node(初始化TTS节点)
        
        Args:
            name:  Node name(节点名称)
        """
        rclpy.init()
        super().__init__(name)
        
        self.text = None  #  Text to be converted(待转换的文本)
        speech.set_volume(80)  # Set volume to 80% (设置音量为80%)
        self.language = os.environ["ASR_LANGUAGE"]  # Get language setting from environment variable (从环境变量获取语言设置)
        if self.language == 'Chinese':
            self.tts = speech.RealTimeTTS(log=self.get_logger())  # Chinese TTS engine (中文TTS引擎)
        else:
            self.tts = speech.RealTimeOpenAITTS(log=self.get_logger())  # English TTS engine (英文TTS引擎)
       
        # Create publisher and subscriber (创建发布者和订阅者)
        self.play_finish_pub = self.create_publisher(Bool, '~/play_finish', 1)  # Play finish publisher (播放完成发布者)
        self.create_subscription(String, '~/tts_text', self.tts_callback, 1)  # Subscribe to text message(订阅文本消息)

        # Start TTS processing thread (启动TTS处理线程)
        threading.Thread(target=self.tts_process, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)  # Create initialization finish service (创建初始化完成服务)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')  # Print start information (打印启动信息)

    def get_node_state(self, request, response):
        """
        Node state service callback(获取节点状态服务回调)
        
        Args:
            request: Request (请求)
            response: Response (响应)
        
        Returns:
            response: Response object (响应对象)
        """
        return response

    def tts_callback(self, msg):
        """

        Text message callback function(文本消息回调函数)
        
        Args:
            msg: Message containing text to be converted to speech (包含要转换为语音的文本的消息)
        """
        # self.get_logger().info(msg.data)
        self.text = msg.data  # Update text to be processed (更新待处理的文本)

    def tts_process(self):
        """

        TTS processing thread function, continuously monitors and processes text-to-speech requests(TTS处理线程函数，持续监听并处理文本转语音请求)
        """
        while True:
            if self.text is not None:  # If there is text to be processed (如果有待处理的文本)
                if self.text == '':
                    speech.play_audio(no_voice_audio_path)  # Play no voice prompt sound (播放无语音提示音)
                else:
                    self.tts.tts(self.text, model=tts_model, voice=voice_model)  # Perform text-to-speech (执行文本转语音)
                self.text = None  # Clear processed text (清空已处理的文本)
                msg = Bool()
                msg.data = True
                self.play_finish_pub.publish(msg)  # Publish play finish message (发布播放完成消息)
            else:
                time.sleep(0.01)  # Short sleep to reduce CPU usage (短暂休眠以减少CPU占用)

def main():
    node = TTSNode('tts_node')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('shutdown')
    finally:
        rclpy.shutdown() 

if __name__ == "__main__":
    main()
