import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Limit PyTorch CPU threads to prevent deadlock with ROS multi-threading callbacks
torch.set_num_threads(1)

# Ensure velodyne_env is in the same directory or in PYTHONPATH
from velodyne_env import GazeboEnv

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # First 20 dims: LiDAR data, last 8 dims: robot state
        self.laser_dim = 20
        self.robot_dim = 8

        # Perception stream: processes LiDAR data
        self.laser_net = nn.Sequential(
            nn.Linear(self.laser_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        # State stream: processes robot ego-state
        self.robot_net = nn.Sequential(
            nn.Linear(self.robot_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        # Fusion head: combines features and outputs action
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Output bounded in [-1, 1]
        )

        # Weight initialization for training stability
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, s):
        # State layout: first 20 dims = LiDAR, last 8 dims = robot ego-state
        laser = s[:, :self.laser_dim]
        robot = s[:, self.laser_dim:]

        # Extract features from each stream
        l_feat = self.laser_net(laser)
        r_feat = self.robot_net(robot)

        # Concatenate and produce action
        combined = torch.cat([l_feat, r_feat], dim=1)
        a = self.head(combined)
        return a

# TD3 policy wrapper (actor-only for inference)
class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Switch to eval mode — critical because of LayerNorm
        self.actor.eval()

        state = torch.Tensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
max_ep = 3000
file_name = "TD3_velodyne"

environment_dim = 10
robot_dim = 8
# State = front LiDAR (10) + down LiDAR (10) + robot ego-state (8) = 28
state_dim = environment_dim + environment_dim + robot_dim
action_dim = 4

print("Initializing environment...")
env = GazeboEnv("start_pid_demo_with_teleop.launch", environment_dim)
env.episode_count = 500  # Skip curriculum phase (first 500 eps), use wide-range goal sampling
time.sleep(5)

torch.manual_seed(seed)
np.random.seed(seed)

print(f"State Dim: {state_dim}, Action Dim: {action_dim}")

network = TD3(state_dim, action_dim)

model_dir = "/home/lzde/DRL-robot-navigation/pytorch_models"
try:
    network.load(file_name, model_dir)
    print(f"Model loaded: {model_dir}/{file_name}_actor.pth")
except Exception as e:
    raise ValueError(f"Failed to load model. Check path. Error: {e}")

done = False
episode_timesteps = 0
state = env.reset()

print("Starting test ...")
test_episodes = 0
success_count = 0
collision_count = 0

while True:
    action = network.get_action(np.array(state))
    action = np.clip(action, -1, 1)

    # Action mapping: asymmetric vertical thrust + safety ceiling
    val_z = action[3]
    real_z = 0.0

    # Asymmetric thrust: weak ascent (0.2x), strong descent (0.5x)
    if val_z > 0:
        real_z = val_z * 0.2
    else:
        real_z = val_z * 0.5

    # Safety ceiling: block ascent when near surface to avoid blind zone
    if hasattr(env, 'odom_z'):
        if env.odom_z > -10.0 and real_z > 0:
            real_z = 0.0

    # Compose final action command
    a_in = [
        (action[0] + 1) / 2,     # Linear X: 0 ~ 1
        action[1],               # Angular Z: -1 ~ 1
        action[2] / 10,          # Angular Y: -0.1 ~ 0.1
        real_z                   # Linear Z: -0.5 ~ 0.2
    ]

    # step() returns 5 values: next_state, reward, done, target, collision
    response = env.step(a_in)
    if len(response) == 5:
        next_state, reward, done, target, collision = response
    else:
        next_state, reward, done, target = response
        collision = False

    if episode_timesteps + 1 == max_ep:
        print("Max episode timesteps reached")
        done = True

    if done:
        test_episodes += 1
        if target:
            success_count += 1
        if collision:
            collision_count += 1

        print(f"Episode {test_episodes} ended. Steps: {episode_timesteps}. Target Reached: {target}. Collision: {collision}")
        print(f"Stats -> Success Rate: {success_count/test_episodes*100:.2f}% | Collision Rate: {collision_count/test_episodes*100:.2f}%")
        print("-" * 50)

        state = env.reset()
        done = False
        episode_timesteps = 0
    else:
        state = next_state
        episode_timesteps += 1
