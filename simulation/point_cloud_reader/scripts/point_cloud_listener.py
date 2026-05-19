#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import random
import numpy as np




def point_cloud_callback(msg):
    # 获取点云数据中的 x, y, z 信息
    try:
        # 提取点云数据
        data = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False))
                # 定义 RPY 角度（以弧度表示）
        r = -1.57  # roll
        p = 0      # pitch
        y = -1.57  # yaw

      # 构建旋转矩阵
        R_x = np.array([
            [1, 0, 0],
           [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)]
        ])

        R_y = np.array([
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)]
        ])
        
        R_z = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵
        R = np.dot(R_z, np.dot(R_y, R_x))

        # 应用旋转矩阵到数据
        rotated_data = []
        for (x, y, z) in data:
            point = np.array([x, y, z])
            rotated_point = np.dot(R, point)
            rotated_point[2] -= 50
            rotated_data.append(tuple(rotated_point))
        data = rotated_data
        
                # 打印点云中的总点数
        total_points = len(data)
        print(f"Total number of points in the point cloud: {total_points}")
        
        # 打印前几个点的坐标（仅为示例，你可以根据需要处理所有点）
       #  for i, point in enumerate(points[:10]):  # 这里只打印前10个点
           #  print(f"Point {i+1}: x = {point[0]}, y = {point[1]}, z = {point[2]}")
           
                   # 随机选择10个点进行打印
        selected_points = random.sample(data, 10)
        
        # 打印选中的10个点的坐标
        for i, point in enumerate(selected_points):
            print(f"Point {i+1}: x = {point[0]}, y = {point[1]}, z = {point[2]}")
    
    except Exception as e:
        rospy.logerr("Error reading PointCloud2 data: %s", e)

def listener():
    # 初始化 ROS 节点
    rospy.init_node('point_cloud_listener', anonymous=True)

    # 订阅点云话题，订阅消息类型为 PointCloud2
    rospy.Subscriber("/rexrov/blueview_p900/blueview_p900/point_cloud", PointCloud2, point_cloud_callback)

    # 保持节点运行
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass








    #     # -0.03 是一个小的偏移量，这个偏移量可能用于避免一些特殊情况（如接近极限的边界问题），从而确保能够包含一些必要的间隔
    #     self.gaps = [[-np.pi / 4 - 0.03, -np.pi / 4 + np.pi / self.environment_dim]]   # 改了
    #     for m in range(self.environment_dim - 1):
    #         self.gaps.append(
    #             [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
    #         )
    #     self.gaps[-1][-1] += 0.03
        
        
        
    #     # ---velodyne_data，用来存储每个角度范围内的最小距离数据---
    #     self.velodyne_data = np.ones(self.environment_dim) * 100
    #     # filtered_data = []
    #     for i in range(len(data)):
    #         # 过滤掉地面以下或不需要的数据点，可改-0.2
    #         if -60 < data[i][2] < -40:
    #         #     # filtered_data.append(data[i])

    #         #     # # 打印符合条件的数据点个数
    #         #     # print(f"Filtered data points: {len(filtered_data)}")

    #         #     dot = (data[i][0] - self.odom_x) * 1 + (data[i][1] - self.odom_y) * 0
    #         #     mag1 = math.sqrt((data[i][0] - self.odom_x)** 2 + (data[i][1] - self.odom_y)** 2)
    #         #     mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))

    #         # # 检查 mag1 是否为零
    #         #     if mag1 == 0:
    #         #         print(f"Warning: mag1 is zero at point {data[i]}. Setting default values.")
    #         #         continue  # 跳过当前点或设置默认值
    #         #     beta = math.acos(dot / (mag1 * mag2)) * np.sign((data[i][1] - self.odom_y))
    #         #     dist = math.sqrt((data[i][0] - self.odom_x) ** 2 + (data[i][1] - self.odom_y) ** 2 + (data[i][2] + 50) ** 2)

    #         #     for j in range(len(self.gaps)):
    #         #         if self.gaps[j][0] <= beta < self.gaps[j][1]:
    #         #             self.velodyne_data[j] = min(self.velodyne_data[j], dist)
    #         #             break
    #                     # 计算当前位置（小车位置）与当前点之间的距离
    #             _x = data[i][0] - (self.odom_x + 1.32)  # 当前点与小车在x方向的差距
    #             _y = data[i][1] - self.odom_y  # 当前点与小车在y方向的差距
    #             _z = data[i][2] - (-50.75)        # 当前点与小车在z方向的差距，z坐标固定为-50
                 
    # # 计算从小车到当前点的距离
    #             dist = math.sqrt(_x ** 2 +_y ** 2 +_z ** 2)

    # # 计算当前点与参考向量(1, 0)的角度
    #             dot = _x * 1 + _y * 0
    #             mag1 = math.sqrt(_x ** 2 + _y ** 2)
    #             mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
    
    # # 检查 mag1 是否为零
    #             if mag1 == 0:
    #                 print(f"Warning: mag1 is zero at point {data[i]}. Setting default values.")
    #                 continue  # 跳过当前点或设置默认值

    #             beta = math.acos(dot / (mag1 * mag2)) * np.sign(_y)

    # # 根据角度区间更新距离
    #             for j in range(len(self.gaps)):
    #                 if self.gaps[j][0] <= beta < self.gaps[j][1]:
    #                     self.velodyne_data[j] = min(self.velodyne_data[j], dist)
    #                     break
        
        
        
        



