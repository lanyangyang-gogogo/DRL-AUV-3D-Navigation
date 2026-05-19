import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Limit PyTorch CPU threads to prevent deadlock with ROS multi-threading callbacks
torch.set_num_threads(1)
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import math

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv


def evaluate(network, epoch, eval_episodes=2):
    avg_reward = 0.0
    col = 0
    suc = 0
    print(f"Evaluating at epoch {epoch}...")
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        print(f"Evaluation episode {_}:")
        done = False
        while not done and count < 1501:
            action = network.get_action(np.array(state))
            action = np.clip(action, -1, 1)

            # Asymmetric vertical thrust mapping + safety ceiling
            val_z = action[3]
            real_z = 0.0

            # Asymmetric thrust: weak ascent (0.2x), strong descent (0.5x)
            if val_z > 0:
                real_z = val_z * 0.2
            else:
                real_z = val_z * 0.5

            # Safety ceiling: block ascent when depth > -10m to avoid blind zone
            if env.odom_z > -10.0 and real_z > 0:
                real_z = 0.0

            a_in = [
                (action[0] + 1) / 2,     # Linear X
                action[1],               # Angular Z
                action[2] / 10,          # Angular Y
                real_z                   # Linear Z
            ]
            state, reward, done, target, collision = env.step(a_in)
            avg_reward += reward
            count += 1
            if collision:
                col += 1
            if target:
                suc += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_suc = suc / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i, Avg Reward:%f, Collisions: %f, Successes: %f"
        % (eval_episodes, epoch, avg_reward, avg_col, avg_suc)
    )
    print("..............................................")
    return avg_reward

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

        # Fusion head: 256 (laser) + 256 (robot) = 512 -> action
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Output bounded in [-1, 1]
        )

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

        l_feat = self.laser_net(laser)
        r_feat = self.robot_net(robot)

        combined = torch.cat([l_feat, r_feat], dim=1)
        a = self.head(combined)
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.laser_dim = 20
        self.robot_dim = 8

        # Q1 architecture (dual-stream)
        self.l1_laser = nn.Sequential(nn.Linear(self.laser_dim, 512), nn.GELU())
        self.l1_robot = nn.Sequential(nn.Linear(self.robot_dim + action_dim, 256), nn.GELU())

        self.q1_head = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        # Q2 architecture (dual-stream)
        self.l2_laser = nn.Sequential(nn.Linear(self.laser_dim, 512), nn.GELU())
        self.l2_robot = nn.Sequential(nn.Linear(self.robot_dim + action_dim, 256), nn.GELU())

        self.q2_head = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, s, a):
        laser = s[:, :self.laser_dim]
        robot = s[:, self.laser_dim:]

        # Concatenate action with robot state (action directly affects robot, not environment)
        ra = torch.cat([robot, a], dim=1)

        # Q1 forward
        l1 = self.l1_laser(laser)
        r1 = self.l1_robot(ra)
        q1 = self.q1_head(torch.cat([l1, r1], dim=1))

        # Q2 forward
        l2 = self.l2_laser(laser)
        r2 = self.l2_robot(ra)
        q2 = self.q2_head(torch.cat([l2, r2], dim=1))

        return q1, q2
    
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_lr = 0.0001
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Critic networks (twin Q-networks for TD3)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_lr = 0.0001
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Switch to eval mode — critical because LayerNorm behaves differently in train/eval
        self.actor.eval()

        state = torch.Tensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()

        self.actor.train()
        return action

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=256,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        current_episode=0,
        episode_reward_metric=0.0
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        avg_reward = 0.0
        self.episode_num = current_episode

        self.actor.train()
        self.critic.train()

        for it in range(iterations):
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)

            reward = torch.Tensor(batch_rewards).to(device).view(-1, 1)
            done = torch.Tensor(batch_dones).to(device).view(-1, 1)

            # Target action with clipped noise (TD3 smoothing)
            next_action = self.actor_target(next_state)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Twin Q-targets: take the minimum to reduce overestimation
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q).item()
            max_Q = max(max_Q, torch.max(target_Q).item())

            # Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Critic loss
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # Delayed policy updates (TD3: update actor less frequently than critic)
            if self.iter_count % policy_freq == 0:
                # Gradient ascent on Q-values w.r.t. actor parameters
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()

                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                # Soft update actor target network
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                # Soft update critic target network
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss.item()
            avg_reward += reward.sum().item()

        avg_reward /= iterations

        self.iter_count += 1
        if self.iter_count % 100 == 0:
            self.writer.add_scalar("loss", av_loss / iterations, self.episode_num)
            self.writer.add_scalar("Av. Q", av_Q / iterations, self.episode_num)
            self.writer.add_scalar("Max. Q", max_Q, self.episode_num)
            self.writer.add_scalar("Batch_Reward", avg_reward, self.episode_num)
            self.writer.add_scalar("actor_learning_rate", self.actor_optimizer.param_groups[0]['lr'], self.episode_num)
            self.writer.add_scalar("critic_learning_rate", self.critic_optimizer.param_groups[0]['lr'], self.episode_num)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
eval_freq = 20000
max_ep = 3000
eval_ep = 2
max_timesteps = 2.4e6
max_episodes = 5000
expl_noise = 0.2
expl_decay_steps = 500000
expl_min = 0.05
batch_size = 256
discount = 0.995
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
buffer_size = 1.8e6
file_name = "TD3_velodyne"
save_model = True
load_model = False
random_near_obstacle = False

# Warm-up steps: collect random transitions before training to avoid early overfitting
start_timesteps = 10000

# Create storage directories
if not os.path.exists("/home/lzde/DRL-robot-navigation/results"):
    os.makedirs("/home/lzde/DRL-robot-navigation/results")

if save_model and not os.path.exists("/home/lzde/DRL-robot-navigation/pytorch_models"):
    os.makedirs("/home/lzde/DRL-robot-navigation/pytorch_models")

# Create the training environment
environment_dim = 10
robot_dim = 8
env = GazeboEnv("start_pid_demo_with_teleop.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + environment_dim + robot_dim
action_dim = 4
max_action = 1

# Create the network and replay buffer
network = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "/home/lzde/DRL-robot-navigation/pytorch_models")
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# Create evaluation data store
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

episode_reward = 0
episode_timesteps = 0

# Action-hold mechanism: prevents high-frequency random actions from canceling each other
hold_action_counter = 0
held_action = None

# Begin the training loop
while episode_num < max_episodes:

    # On termination of episode
    if done:
        # Log episode total reward to TensorBoard
        network.writer.add_scalar("Episode_Total_Reward", episode_reward, episode_num)

        # Evaluation and model saving
        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="/home/lzde/DRL-robot-navigation/pytorch_models")
            np.save("/home/lzde/DRL-robot-navigation/results/%s" % (file_name), evaluations)
            epoch += 1

        print(f"Episode {episode_num} has ended. Reward: {episode_reward:.2f}. Episode Timesteps: {episode_timesteps}. Total Timesteps: {timestep}. Resetting environment...")
        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        # Reset action-hold counter at episode end
        hold_action_counter = 0

    # Decay exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    # Warm-up phase with action-hold (5-step hold to avoid thrust cancellation)
    if timestep < start_timesteps:
        if hold_action_counter == 0:
            held_action = np.random.uniform(-max_action, max_action, action_dim)
            hold_action_counter = 5
        action = held_action
        hold_action_counter -= 1
    else:
        # Network decision with exploration noise
        action = network.get_action(np.array(state))
        action = np.clip(action, -1, 1)
        action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
            -max_action, max_action
        )

    # Random action near obstacles to encourage exploration
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[0:-8]) < 5
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.concatenate((
                np.random.uniform(-1, 1, 2),
                np.random.uniform(-0.1, 0.1, 1),
                np.random.uniform(-1, 1, 1)
            ))

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action

    # Action mapping: asymmetric vertical thrust + safety ceiling
    val_z = action[3]
    real_z = 0.0

    # Asymmetric thrust: weak ascent (0.2x), strong descent (0.5x)
    if val_z > 0:
        real_z = val_z * 0.2
    else:
        real_z = val_z * 0.5

    # Safety ceiling: block ascent when depth > -10m to avoid blind zone
    if env.odom_z > -10.0 and real_z > 0:
        real_z = 0.0

    a_in = [
        (action[0] + 1) / 2,     # Linear X: 0 ~ 1
        action[1],               # Angular Z: -1 ~ 1
        action[2] / 10,          # Angular Y: -0.1 ~ 0.1 (Pitch)
        real_z                   # Linear Z: -0.5 ~ 0.2 (Heave)
    ]

    next_state, reward, done, target, collision = env.step(a_in)

    if episode_timesteps + 1 == max_ep:
        print("Max episode timesteps reached")
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # Save transition in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Step-based update: train the network once per environment step
    if timestep > start_timesteps:
        network.train(
            replay_buffer,
            iterations=1,
            batch_size=batch_size,
            discount=discount,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            current_episode=episode_num,
            episode_reward_metric=episode_reward
        )

    # Update counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After training is done, evaluate and save
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="/home/lzde/DRL-robot-navigation/pytorch_models")
np.save("/home/lzde/DRL-robot-navigation/results/%s" % file_name, evaluations)