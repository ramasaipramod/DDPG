from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import threading
import gymnasium as gym
import torch
import random
import torch.nn as nn 
import torch.optim as optim
import numpy as np
from itertools import count
from networks import *
from buffer import *

app = Flask(__name__)
socketio = SocketIO(app, logger=True, engineio_logger=True)

batch_size = 32
steps_done = 0
num_episodes = 20000
eps_start = 1.0
eps_end = 0.1
eps_decay = 10000 
lr = 1e-3
tau = 0.005
gamma = 0.99
warm_up = 50000
target_update = 1000  
policy_update = 4

env = gym.make('HumanoidStandup-v4', render_mode='rgb_array')
nstate = env.observation_space.shape[0]
naction = env.action_space.shape[0]

memory = ReplayMemory(100000)
observation, info = env.reset()

critic_network = critic(nstate, naction)
actor_network = actor(nstate)

critic_target_network = critic(nstate, naction)
actor_target_network = actor(nstate)

critic_target_network.load_state_dict(critic_network.state_dict())
actor_target_network.load_state_dict(actor_network.state_dict())

critic_optimizer = optim.Adam(critic_network.parameters(), lr=lr)
actor_optimizer = optim.Adam(actor_network.parameters(), lr=lr)

@app.route('/')
def index():
    return render_template('index.html')

def select_action(state):
    global steps_done
    select = random.random()
    tanh = nn.Tanh()
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if select > eps_threshold:
        with torch.no_grad():
            return actor_network.forward(state)
    else:
        return 0.4 * tanh(torch.randn(naction))

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
    
    state_action_values = critic_network.forward(torch.cat((state_batch, action_batch), dim=1))
    next_state_values = torch.zeros(batch_size)

    with torch.no_grad():
        next_state_values[non_final_mask] = critic_target_network.forward(
            torch.cat((non_final_next_states, actor_target_network.forward(non_final_next_states)), dim=1)).squeeze(1)

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    critic_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    critic_optimizer.zero_grad()
    critic_loss.backward(retain_graph=True)
    critic_optimizer.step()

    policy_actions = actor_network(state_batch)
    actor_loss = -critic_network(torch.cat((state_batch, policy_actions), dim=1)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

def train_model():
    cumm_reward = 0
    for i in range(num_episodes):
        observation, info = env.reset()
        state = torch.tensor(observation)
        episode_reward = 0
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, info = env.step(action.numpy())
            frame = env.render()
            encoded_frame = encode_image(frame)
            socketio.emit('frame_update', {'frame': encoded_frame})
            reward = torch.tensor(reward)
            episode_reward += reward.item()
            done = terminated or truncated
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation)

            memory.push(state, action, reward, next_state)
            state = next_state

            optimize_model()
            if steps_done % target_update == 0:
                critic_target_network.load_state_dict(critic_network.state_dict())
                actor_target_network.load_state_dict(actor_network.state_dict())

            if done:
                cumm_reward += episode_reward
                if i % 50 == 0:
                    print(f'Episode {i}, Steps {t}, Average Reward {cumm_reward / 50:.2f}')
                    cumm_reward = 0
                break
    socketio.emit('training_done')
    print('Training completed')

@socketio.on('start_training')
def handle_start_training():
    print('Training started')
    thread = threading.Thread(target=train_model)
    thread.start()

if __name__ == '__main__':
    socketio.run(app, debug=True)
