import math
import os
import random
import time
import threading
from os import path
from collections import deque

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 5.0     # Goal reached threshold (3D spherical check)
COLLISION_DIST = 2.5        # Collision distance threshold

# High-frequency micro-stepping: reduce step duration from 0.1s to 0.05s
TIME_DELTA = 0.05

# === Environment rules and spatial constraints ===

# Unified pillar obstacle detection (center +-10m safety margin)
def is_in_pillar(x, y):
    pillar_centers = [(20, 20), (20, -20), (-20, 20), (-20, -20)]
    safety_margin = 10.0  
    
    for cx, cy in pillar_centers:
        if (cx - safety_margin < x < cx + safety_margin) and \
           (cy - safety_margin < y < cy + safety_margin):
            return True
    return False

def check_pos(x, y, z):
    goal_ok = True
    if x > 42 or x < -42 or y > 42 or y < -42 or z > -10 or z < -90:
        goal_ok = False
    if is_in_pillar(x, y):
        goal_ok = False
    return goal_ok

def check_pos_st(x, y, z):
    if not (-38 <= x <= 38 and -38 <= y <= 38 and -60 <= z <= -15):
        return False
    if is_in_pillar(x, y):
        return False
    return True

def check_pos_go(x, y, z):
    if not (-38 <= x <= 38 and -38 <= y <= 38 and -85 <= z <= -15):
        return False
    if is_in_pillar(x, y):
        return False
    return True
# ==========================================

class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.episode_count = 0

        # Thread lock to prevent data races between ROS callbacks and main loop
        self.lock = threading.Lock()

        self.odom_x = 0
        self.odom_y = 0
        self.odom_z = -50

        self.goal_x = 5
        self.goal_y = 0.0
        self.goal_z = -50

        # Store initial pose at episode start
        self.initial_robot_x = 0.0
        self.initial_robot_y = 0.0
        self.initial_goal_x = 0.0
        self.initial_goal_y = 0.0

        # Curriculum learning parameters
        self.curriculum_dist = 5.0
        self.max_goal_dist = 40.0
        self.dist_increment = 0.01
        self.last_distance = 0.0

        self.upper = 60
        self.lower = -60
        self.velodyne_data = np.ones(self.environment_dim) * 50
        self.velodyne_data_down = np.ones(self.environment_dim) * 50
        self.last_odom = None

        # Store previous action for smoothness penalty computation
        self.last_action = [0.0, 0.0, 0.0, 0.0]

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "rexrov"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = -50.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
 
        # FOV-based LiDAR sector partitioning
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        self.action_phase = 0
        self.angular_z_duration = 0.08

        # ROS node and pub/sub initialization
        rospy.init_node("gym", anonymous=True)

        self.vel_pub = rospy.Publisher("/rexrov/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.publisher4 = rospy.Publisher("pitch_velocity", MarkerArray, queue_size=1)    
        self.publisher5 = rospy.Publisher("Heave_velocity", MarkerArray, queue_size=1)    

        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.velodyne_down = rospy.Subscriber(
            "/velodyne_points_2", PointCloud2, self.velodyne_callback_down, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/rexrov/pose_gt", Odometry, self.odom_callback, queue_size=1
        )

        print("Waiting for ROS sensor data...")
        rospy.wait_for_message("/rexrov/pose_gt", Odometry)
        rospy.wait_for_message("/velodyne_points", PointCloud2)
        print("All sensor data received, environment initialized.")

    def velodyne_callback(self, v):
            try:
                # Lock to prevent concurrent modification of velodyne_data
                with self.lock:
                    self.velodyne_data = np.ones(self.environment_dim) * 50

                    # Iterate over point cloud generator directly to save memory
                    for p in pc2.read_points(v, skip_nans=True, field_names=("x", "y", "z")):
                        if -2 < p[2] < 2:
                            dot = p[0] * 1 + p[1] * 0
                            mag1 = math.sqrt(p[0] ** 2 + p[1] ** 2)
                            mag2 = 1.0  # sqrt(1^2 + 0^2) is always 1.0
                
                            if mag1 == 0:
                                continue 
                                                                                                            
                            beta = math.acos(dot / (mag1 * mag2)) * np.sign(p[1])
                            dist = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
                
                            for j in range(len(self.gaps)):
                                if j < len(self.gaps) and j < len(self.velodyne_data):
                                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                                        break
            except Exception as e:
                rospy.logwarn(f"Error in velodyne_callback: {e}")

    def velodyne_callback_down(self, v):
            try:
                # Lock to prevent concurrent modification of velodyne_data_down
                with self.lock:
                    self.velodyne_data_down = np.ones(self.environment_dim) * 50

                    for p in pc2.read_points(v, skip_nans=True, field_names=("x", "y", "z")):
                        if -2 < p[2] < 2: 
                            dot = p[0] * 1 + p[1] * 0
                            mag1 = math.sqrt(p[0]**2 + p[1]**2)
                            mag2 = 1.0
                
                            if mag1 == 0:
                                continue
                        
                            beta = math.acos(dot / (mag1 * mag2)) * np.sign(p[1])
                            dist = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                
                            for j in range(len(self.gaps)):
                                if j < len(self.gaps) and j < len(self.velodyne_data_down):
                                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                                        self.velodyne_data_down[j] = min(self.velodyne_data_down[j], dist)
                                        break
            except Exception as e:
                rospy.logwarn(f"Error in velodyne_callback_down: {e}")

    def odom_callback(self, od_data):
        # Thread-safe odometry data update
        with self.lock:
            self.last_odom = od_data

    def step(self, action):
        """Execute one action and return new state, reward, done, target, collision."""
        target = False

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        vel_cmd.angular.y = action[2]
        vel_cmd.linear.z = action[3]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # Thread-safe snapshot of sensor data
        with self.lock:
            current_velodyne = np.copy(self.velodyne_data)
            current_velodyne_down = np.copy(self.velodyne_data_down)
            current_odom = self.last_odom

        done, collision, min_laser, min_laser_down = self.observe_collision(
            current_velodyne, current_velodyne_down
        )
        
        self.odom_x = current_odom.pose.pose.position.x
        self.odom_y = current_odom.pose.pose.position.y
        self.odom_z = current_odom.pose.pose.position.z
        
        quaternion = Quaternion(
            current_odom.pose.pose.orientation.w,
            current_odom.pose.pose.orientation.x,
            current_odom.pose.pose.orientation.y,
            current_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)   
        angle = round(euler[2], 4)   
        pitch = round(euler[1], 4)

        current_dist_3d = np.linalg.norm([
            self.odom_x - self.goal_x,
            self.odom_y - self.goal_y,
            self.odom_z - self.goal_z
        ])

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        
        distance_z = self.odom_z - self.goal_z 

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0    
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))  
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        
        if mag1 * mag2 == 0:
            beta = 0
        else:
            val = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
            beta = math.acos(val)

        if skew_y < 0:   
            if skew_x < 0:
                beta = -beta  
            else:
                beta = 0 - beta  
        theta = beta - angle   
        
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        if current_dist_3d < GOAL_REACHED_DIST:
            target = True
            done = True
            print(f"Target reached! 3D distance: {current_dist_3d:.2f}")

        norm_laser = current_velodyne / 50.0
        norm_laser_down = current_velodyne_down / 50.0
        
        norm_current_dist = current_dist_3d / 110.0  
        norm_planar_dist = distance / 110.0
        norm_z_dist = distance_z / 110.0
        norm_theta = theta / np.pi              

        robot_state = [
            norm_current_dist, 
            norm_planar_dist, 
            norm_z_dist, 
            norm_theta, 
            action[0], action[1], action[2], action[3]
        ]

        v_state = []
        v_state[:] = norm_laser[:]
        laser_state = [v_state]

        v_state_down = []
        v_state_down[:] = norm_laser_down[:]
        laser_state_down = [v_state_down]

        state = np.append(np.append(laser_state, laser_state_down), robot_state)
        
        reward = self.get_reward(target, collision, action, min_laser, min_laser_down, theta, current_dist_3d)
        
        self.last_distance = current_dist_3d

        # Store action for smoothness penalty in next step
        self.last_action = [action[0], action[1], action[2], action[3]]

        return state, reward, done, target, collision

    def reset(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        pitch = 0.0

        quaternion = Quaternion.from_euler(0.0, pitch, angle) 
        object_state = self.set_self_state

        x = 0
        y = 0
        z = -50
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-25, 25)  
            y = np.random.uniform(-25, 25)  
            z = np.random.uniform(-60, -15) 
            position_ok = check_pos_st(x, y, z)     
            
        object_state.pose.position.x = x  
        object_state.pose.position.y = y
        object_state.pose.position.z = z   
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w

        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x   
        self.odom_y = object_state.pose.position.y
        self.odom_z = object_state.pose.position.z

        self.initial_robot_x = self.odom_x
        self.initial_robot_y = self.odom_y
        self.initial_robot_z = self.odom_z

        self.episode_count += 1
        self.action_phase = 0 
        
        # Reset action cache on new episode
        self.last_action = [0.0, 0.0, 0.0, 0.0]

        # Goal generation with curriculum: close-range -> wide-range 3D spherical sampling
        goal_ok = False
        max_attempts = 100  
        attempts = 0

        while not goal_ok:
            attempts += 1
            if attempts > max_attempts:
                self.goal_x = self.initial_robot_x
                self.goal_y = self.initial_robot_y
                self.goal_z = max(-85.0, self.initial_robot_z - 10.0)
                break

            if self.episode_count < 500:
                radius = random.uniform(5, 15) 
                angle_goal = random.uniform(-np.pi, np.pi)
                
                self.goal_x = self.initial_robot_x + radius * math.cos(angle_goal)
                self.goal_y = self.initial_robot_y + radius * math.sin(angle_goal)
                
                z_drop = random.uniform(2, 10)
                self.goal_z = self.initial_robot_z - z_drop 
                
            else:
                progress = min(1.0, (self.episode_count - 500) / 4500.0)
                current_max_dist = 18.0 + progress * (70.0 - 18.0)
                
                r = random.uniform(5.0, current_max_dist)
                
                theta_angle = random.uniform(-np.pi, np.pi)
                phi = random.uniform(0.1 * np.pi, 0.45 * np.pi) 

                self.goal_x = self.initial_robot_x + r * math.sin(phi) * math.cos(theta_angle)
                self.goal_y = self.initial_robot_y + r * math.sin(phi) * math.sin(theta_angle)
                self.goal_z = self.initial_robot_z - r * math.cos(phi) 

            self.goal_x = max(-38.0, min(38.0, self.goal_x))
            self.goal_y = max(-38.0, min(38.0, self.goal_y))
            self.goal_z = max(-85.0, min(-15.0, self.goal_z))

            goal_ok = check_pos_go(self.goal_x, self.goal_y, self.goal_z)

        self.initial_goal_x = self.goal_x
        self.initial_goal_y = self.goal_y
        self.initial_goal_z = self.goal_z
        
        self.last_distance = np.linalg.norm([
            self.initial_robot_x - self.initial_goal_x,
            self.initial_robot_y - self.initial_goal_y,
            self.initial_robot_z - self.initial_goal_z
        ])

        self.dist_history = deque(maxlen=5)
        for _ in range(5):
            self.dist_history.append(self.last_distance)

        self.publish_markers([0.0, 0.0, 0.0, 0.0])  

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
            
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        
        distance_z = self.odom_z - self.goal_z 
        
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        
        if mag1 * mag2 == 0:
            beta = 0
        else:
            val = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
            beta = math.acos(val)
        
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        current_dist_3d = np.linalg.norm([
            self.odom_x - self.goal_x,
            self.odom_y - self.goal_y,
            self.odom_z - self.goal_z
        ])

        # Thread-safe sensor data snapshot for initial state construction
        with self.lock:
            current_velodyne = np.copy(self.velodyne_data)
            current_velodyne_down = np.copy(self.velodyne_data_down)

        norm_laser = current_velodyne / 50.0
        norm_laser_down = current_velodyne_down / 50.0
        
        norm_current_dist = current_dist_3d / 110.0
        norm_planar_dist = distance / 110.0
        norm_z_dist = distance_z / 110.0
        norm_theta = theta / np.pi

        robot_state = [
            norm_current_dist, 
            norm_planar_dist, 
            norm_z_dist, 
            norm_theta, 
            0.0, 0.0, 0.0, 0.0
        ] 

        v_state = []
        v_state[:] = norm_laser[:]
        laser_state = [v_state]

        v_state_down = []
        v_state_down[:] = norm_laser_down[:]
        laser_state_down = [v_state_down]

        state = np.append(np.append(laser_state, laser_state_down), robot_state)
        return state   

    def calculate_combined_angle_diff(self, robot_x, robot_y, robot_z, robot_yaw, robot_pitch, goal_x, goal_y, goal_z):
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        dz = goal_z - robot_z

        if dx == 0 and dy == 0 and dz == 0:
            return 0.0

        target_vec = np.array([dx, dy, dz])
        target_vec_norm = np.linalg.norm(target_vec)  
        target_unit_vec = target_vec / target_vec_norm  

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        cos_pitch = math.cos(robot_pitch)
        sin_pitch = math.sin(robot_pitch)

        robot_unit_vec = np.array([
            cos_yaw * cos_pitch,  
            sin_yaw * cos_pitch,  
            sin_pitch              
        ])

        dot_product = np.dot(robot_unit_vec, target_unit_vec)
        dot_product = np.clip(dot_product, -1.0, 1.0)  

        combined_angle = math.acos(dot_product)

        return combined_angle

    def random_box(self):
        for i in range(3):    
            name = "random_box_" + str(i)    

            x = 0
            y = 0
            z = -50
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-25, 25)   
                y = np.random.uniform(-25, 25)
                y = np.random.uniform(-75, -25)
                box_ok = check_pos(x, y, z)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y, z - self.odom_z])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y, z - self.goal_z])
                if distance_to_robot < 10 or distance_to_goal < 10:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = z  
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0  
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = self.goal_z

        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "world"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "world"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

        markerArray4 = MarkerArray()
        marker4 = Marker()
        marker4.header.frame_id = "world"
        marker4.type = marker.CUBE
        marker4.action = marker.ADD
        marker4.scale.x = abs(action[2])
        marker4.scale.y = 0.1
        marker4.scale.z = 0.01
        marker4.color.a = 1.0
        marker4.color.r = 1.0
        marker4.color.g = 0.0
        marker4.color.b = 0.0
        marker4.pose.orientation.w = 1.0
        marker4.pose.position.x = 5
        marker4.pose.position.y = 0.2
        marker4.pose.position.z = 0.2

        markerArray4.markers.append(marker4)
        self.publisher4.publish(markerArray4)

        markerArray5 = MarkerArray()    
        marker5 = Marker()
        marker5.header.frame_id = "world"
        marker5.type = marker.CUBE
        marker5.action = marker.ADD
        marker5.scale.x = abs(action[3])
        marker5.scale.y = 0.1
        marker5.scale.z = 0.01
        marker5.color.a = 1.0
        marker5.color.r = 1.0
        marker5.color.g = 0.0
        marker5.color.b = 0.0
        marker5.pose.orientation.w = 1.0
        marker5.pose.position.x = 5
        marker5.pose.position.y = 0
        marker5.pose.position.z = 0.2

        markerArray5.markers.append(marker5)
        self.publisher5.publish(markerArray5)

    def observe_collision(self, velodyne_data, velodyne_data_down):
        min_laser = np.min(velodyne_data) if len(velodyne_data) > 0 else float('inf')
        min_laser_down = np.min(velodyne_data_down) if len(velodyne_data_down) > 0 else float('inf')
        
        cube_center = (0, 0, -50)
        cube_half_side = 105  
        x_min, x_max = cube_center[0] - cube_half_side, cube_center[0] + cube_half_side
        y_min, y_max = cube_center[1] - cube_half_side, cube_center[1] + cube_half_side
        z_min, z_max = cube_center[2] - cube_half_side, cube_center[2] + cube_half_side
        
        out_of_bounds = (
            self.odom_x < x_min or self.odom_x > x_max or
            self.odom_y < y_min or self.odom_y > y_max or
            self.odom_z < z_min or self.odom_z > z_max
        )
        
        if min_laser < COLLISION_DIST or min_laser_down < COLLISION_DIST or out_of_bounds:
            if out_of_bounds:
                print(f"Boundary collision! Position: (x:{self.odom_x:.2f}, y:{self.odom_y:.2f}, z:{self.odom_z:.2f})")
            else:
                print(f"LiDAR collision! Front min distance:{min_laser:.2f}, Down min distance:{min_laser_down:.2f}, Threshold:{COLLISION_DIST}")
            return True, True, min_laser, min_laser_down
        return False, False, min_laser, min_laser_down

    def get_reward(self, target, collision, action, min_laser, min_laser_down, theta, distance_3d):
        """High-frequency responsive reward function."""
        # Terminal rewards
        if target:
            return 100.0
        elif collision:
            return -100.0

        reward = 0.0

        linear_x = action[0]
        angular_z = action[1]
        heave_z = action[3]

        # Potential-based reward: reward progress toward goal
        diff = self.last_distance - distance_3d
        reward += diff * 8.0

        # Heading discipline: penalize forward motion when not facing target
        is_facing_target = abs(theta) < 0.5

        if not is_facing_target:
            if linear_x > 0.1:
                reward -= linear_x * 0.5

        # Depth alignment: encourage correct vertical direction
        z_error = self.odom_z - self.goal_z

        if z_error > 0.5:
            if heave_z < -0.1:
                reward += 0.2
        elif z_error < -0.5:
            if heave_z < 0:
                reward -= abs(heave_z) * 3.0

        # Physical constraints: penalize backward motion and ascent
        if linear_x < 0:
            reward -= abs(linear_x) * 2.0
        if heave_z > 0:
            reward -= heave_z * 2.0

        # Dynamic obstacle avoidance: repulsive field
        warning_dist = 4.5

        # Front LiDAR avoidance
        if min_laser < warning_dist:
            danger_ratio = (warning_dist - min_laser) / warning_dist
            reward -= (danger_ratio ** 2) * 2.5

            if linear_x > 0.3:
                reward -= linear_x * 1.5

        # Downward LiDAR avoidance
        if min_laser_down < warning_dist:
            bottom_danger = (warning_dist - min_laser_down) / warning_dist
            reward -= (bottom_danger ** 2) * 2.5

            if heave_z < -0.1:
                reward -= abs(heave_z) * 2.0

        # Exploration stability: penalize excessive yaw rate
        if abs(angular_z) > 0.6:
            reward -= (abs(angular_z) - 0.6) * 0.1

        # Action smoothness penalty: discourage high-frequency jitter at 20Hz control
        action_diff = sum(abs(a - la) for a, la in zip(action, self.last_action))
        reward -= action_diff * 0.05

        # Small per-step survival penalty (halved due to doubled control frequency)
        reward -= 0.005

        return reward