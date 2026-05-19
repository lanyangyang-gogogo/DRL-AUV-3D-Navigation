#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# 声呐图像话题名称
sonar_image_topic = '/rexrov/blueview_p900/sonar_image'  # 根据你的实际话题名修改

def callback(data):
    # 创建 CvBridge 实例
    bridge = CvBridge()
    
    try:
        # 将 ROS 图像消息转换为 OpenCV 图像
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        # 显示图像
        cv2.imshow("Sonar Image", cv_image)
        cv2.waitKey(1)
        
        # 将图像保存为文件
        cv2.imwrite('sonar_image.png', cv_image)
        rospy.loginfo("Image saved as sonar_image.png")

    except Exception as e:
        rospy.logerr("Error converting image: %s", str(e))

def sonar_image_listener():
    # 初始化 ROS 节点
    rospy.init_node('sonar_image_listener', anonymous=True)
    
    # 订阅声呐图像话题
    rospy.Subscriber(sonar_image_topic, Image, callback)
    
    # 保持节点运行
    rospy.spin()

if __name__ == '__main__':
    try:
        sonar_image_listener()
    except rospy.ROSInterruptException:
        pass
