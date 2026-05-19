#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import geometry_msgs.msg

def velocity_callback(msg):
    # 打印接收到的消息
    rospy.loginfo("Received velocity: linear(x: %f, y: %f, z: %f), angular(x: %f, y: %f, z: %f)",
                  msg.linear.x, msg.linear.y, msg.linear.z,
                  msg.angular.x, msg.angular.y, msg.angular.z)

    # 修改水流方向，例如，我们反转线速度的方向：
    # 你可以根据需求来调整此逻辑，控制水流的方向。
    msg.linear.x = -msg.linear.x  # 反转 x 方向
    msg.linear.y = -msg.linear.y  # 反转 y 方向
    msg.linear.z = -msg.linear.z  # 反转 z 方向

    # 发布修改后的消息
    pub.publish(msg)

def main():
    # 初始化 ROS 节点
    rospy.init_node('water_flow_direction_modifier', anonymous=True)

    # 创建一个订阅者，订阅 /rexrov/current_velocity 话题
    rospy.Subscriber('current_velocity', geometry_msgs.msg.Vector3, velocity_callback)

    # 创建一个发布者，用于发布修改后的 velocity 到同一个话题
    global pub
    pub = rospy.Publisher('current_velocity', geometry_msgs.msg.Vector3, queue_size=10)

    # 保持节点运行，直到被停止
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
