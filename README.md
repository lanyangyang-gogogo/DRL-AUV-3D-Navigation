# DRL-AUV-3D-navigation

本研究介绍了一种无先验地图情况下基于深度强化学习TD3算法的三维AUV避障导航方法，该方法使用ROS Gazebo仿真器。AUV在仿真环境中导航至随机目标点并避开障碍物。该模型在ROS Gazebo仿真器中使用PyTorch进行训练，并在Ubuntu 20.04系统上使用ROS Noetic进行全部的训练和测试。
我写了一个帖子放在这里：
(https://blog.csdn.net/gggggg123_/article/details/161395580?spm=1001.2014.3001.5502)

Gazebo：

<img width="600" alt="Gazebo simulation" src="https://github.com/lanyangyang-gogogo/DRL-AUV-3D-Navigation/blob/master/gazebo.gif?raw=true" />


Rviz：

<img width="600" alt="Rviz visualization" src="https://github.com/lanyangyang-gogogo/DRL-AUV-3D-Navigation/blob/master/rviz.gif?raw=true" />




# 本研究主要依据以下项目进行：
* [uuv_simulator](https://github.com/uuvsimulator/uuv_simulator)
* [DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation)
* [DAVE](https://field-robotics-lab.github.io/dave.doc/contents/installation/Clone-Dave-Repositories/)

  
## 安装
主要依赖项: 
* [ROS Noetic](http://wiki.ros.org/noetic/Installation)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [Tensorboard](https://github.com/tensorflow/tensorboard)

项目的安装和环境的配置需要参考uuv_simulator的官方文档，但是需要在创建工作空间之后把uuv_simulator原项目文件换成本项目的simulation部分再进行编译和和环境向量的配置。
可以参考(https://zhuanlan.zhihu.com/p/689227681)
algorithm部分无需进行编译，直接复制粘贴放在Ubuntu的主文件夹中可以运行。

## 运行
建议使用Anaconda创建独立的纯净虚拟环境以隔离ROS的默认Python版本（环境名字用your_env代替），需要提前下载好pyrorch相关的库，运行本项目需要打开两个独立的终端，
运行仿真环境和机器人模型（我使用的仿真模型的工作空间名为uuv_ws,这个需要根据自己的情况修改）：
```shell
$ conda deactivate
$ source /opt/ros/noetic/setup.bash
$ source ~/uuv_ws/devel/setup.bash   
$ roslaunch uuv_gazebo start_pid_demo_with_teleop.launch
```
等机器人模型加载完成，再打开另一个终端进行训练，algorithm这个名字也可以自己进行修改，但是需要前后保持一致（-X faulthandle可以捕捉段错误）：
```shell
$ cd ~/algorithm/TD3
$ conda activate your_env
$ python3 -X faulthandler train_velodyne_td3.py
```
查看训练过程，激活虚拟环境后再打开一个终端：
```shell
$  cd ~/algorithm/TD3
$ tensorboard --logdir runs
```
杀死进程：
```shell
$ killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
```
训练结束后测试模型：
```shell
$ cd ~/algorithm/TD3
$ conda activate your_env
$ python3 -X faulthandler test_velodyne_td3.py
```
## License

See `algorithm/LICENSE` for details.
