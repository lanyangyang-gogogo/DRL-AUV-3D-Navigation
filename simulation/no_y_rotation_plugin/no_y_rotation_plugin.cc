#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Plugin.hh>
#include <ignition/math/Quaternion.hh>

namespace gazebo
{
  class NoYRotationPlugin : public ModelPlugin
  {
  public:
    // 构造函数
    NoYRotationPlugin() : ModelPlugin()
    {
    }

    // 初始化插件
    void Load(physics::ModelPtr model, sdf::ElementPtr sdf)
    {
      // 存储模型指针
      this->model = model;

      // 创建一个事件侦听器，定时更新模型的状态
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&NoYRotationPlugin::OnUpdate, this));

      // 输出加载信息
      std::cout << "NoYRotationPlugin Loaded!" << std::endl;
    }

    // 在每个模拟步骤中运行的更新方法
    void OnUpdate()
    {
      // 获取模型的当前姿态（位置和旋转）
      ignition::math::Pose3d pose = this->model->WorldPose();

      // 强制绕Y轴的旋转角度为零（即只保持X轴和Z轴的旋转）
      ignition::math::Quaterniond currentOrientation = pose.Rot();
      currentOrientation.Euler(0.0, 0.0, currentOrientation.Yaw());

      // 将修改后的旋转设置回模型
      this->model->SetWorldPose(pose);
    }

  private:
    physics::ModelPtr model;  // 机器人模型指针
    event::ConnectionPtr updateConnection;  // 更新连接事件
  };

  // 注册插件
  GZ_REGISTER_MODEL_PLUGIN(NoYRotationPlugin)
}


