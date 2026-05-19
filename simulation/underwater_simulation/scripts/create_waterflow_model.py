#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from random import uniform
import time

def create_waterflow_model():
    # 初始化 ROS 节点
    rospy.init_node('underwater_current_simulator', anonymous=True)
    
    # 创建一个发布者用于发布水流速度
    waterflow_pub = rospy.Publisher('/waterflow_velocity', Vector3, queue_size=10)

    # 启动 Gazebo 设置模型状态服务
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
    
    # 设置水流模型状态
    def set_waterflow_state():
        # 创建一个水流模型的状态
        model_state = ModelState()
        model_state.model_name = 'water_current'  # 水流模型名称
        model_state.pose.position.x = 0  # 水流起始位置（可以调整）
        model_state.pose.position.y = 0
        model_state.pose.position.z = -10  # 水下环境

        # 给模型添加随机速度来模拟水流
        model_state.twist.linear.x = uniform(-1.0, 1.0)  # 随机水流速度
        model_state.twist.linear.y = uniform(-1.0, 1.0)
        model_state.twist.linear.z = uniform(-0.5, 0.5)
        
        # 发布模型状态到 Gazebo
        try:
            response = set_model_state(model_state)
            rospy.loginfo("Waterflow model state updated: %s" % response)
        except rospy.ServiceException as e:
            rospy.logerr("Set model state failed: %s" % e)

    # 发布水流速度数据
    def publish_waterflow_velocity():
        while not rospy.is_shutdown():
            water_velocity = Vector3()
            water_velocity.x = uniform(-1.0, 1.0)  # 水流X轴速度
            water_velocity.y = uniform(-1.0, 1.0)  # 水流Y轴速度
            water_velocity.z = uniform(-0.5, 0.5)  # 水流Z轴速度

            waterflow_pub.publish(water_velocity)
            rospy.loginfo("Published waterflow velocity: %s" % str(water_velocity))
            time.sleep(1)  # 每秒发布一次水流速度

    # 设置水流模型状态
    set_waterflow_state()

    # 启动发布水流速度
    publish_waterflow_velocity()

if __name__ == '__main__':
    try:
        create_waterflow_model()
    except rospy.ROSInterruptException:
        pass
