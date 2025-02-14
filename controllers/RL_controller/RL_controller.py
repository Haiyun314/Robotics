
import debugpy
debugpy.listen(("localhost", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import random
from controller import Robot, Motor, Gyro

def build_actor(state_dim, action_dim):
    inputs = Input(shape=(state_dim,))
    x = Dense(64, activation="relu")(inputs)
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(action_dim, activation="tanh")(x)  # Actions between -1 and 1
    return Model(inputs, outputs)

def build_critic(state_dim, action_dim):
    state_input = Input(shape=(state_dim,))
    action_input = Input(shape=(action_dim,))
    x = tf.keras.layers.Concatenate()([state_input, action_input])
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)  # Q-value estimation
    return Model([state_input, action_input], outputs)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.actor = build_actor(state_dim, action_dim)
        self.critic = build_critic(state_dim, action_dim)
        
        self.target_actor = build_actor(state_dim, action_dim)
        self.target_critic = build_critic(state_dim, action_dim)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        
        self.memory = []
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Soft update rate

    def select_action(self, state):
        state = np.reshape(state, (1, self.state_dim))
        action = self.actor.predict(state)[0]
        return action * self.action_bound  # Scale to action range

    def update(self):
        if len(self.memory) < 1000: return

        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        # Compute target Q-value
        next_actions = self.target_actor.predict(next_states)
        next_Q = self.target_critic.predict([next_states, next_actions])
        target_Q = rewards + self.gamma * next_Q.squeeze()

        # Train Critic
        with tf.GradientTape() as tape:
            Q_values = self.critic([states, actions])
            loss = tf.keras.losses.MSE(target_Q, Q_values)
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        # Train Actor
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_value = self.critic([states, new_actions])
            actor_loss = -tf.reduce_mean(critic_value)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        # Update Target Networks
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.trainable_variables, source.trainable_variables):
            target_param.assign(self.tau * source_param + (1 - self.tau) * target_param)
    
    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 10000:  
            self.memory.pop(0)
           


 
class Init_State:
    def __init__(self, motors):
        self.motors = motors
        self.initial_positions = [motor.getTargetPosition() for motor in motors]
    
    def reset(self):
        for motor, initial_position in zip(self.motors, self.initial_positions):
            motor.setPosition(initial_position)
            

robot = Robot()
time_step = int(robot.getBasicTimeStep())

motor_names = [
    "front left shoulder abduction motor", "front left shoulder rotation motor", "front left elbow motor",
    "front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor",
    "rear left shoulder abduction motor", "rear left shoulder rotation motor", "rear left elbow motor",
    "rear right shoulder abduction motor", "rear right shoulder rotation motor", "rear right elbow motor"
]

motors = [robot.getDevice(name) for name in motor_names]

# Camera names and initialization
camera_names = ["left head camera", "right head camera", "left flank camera", "right flank camera", "rear camera"]
cameras = [robot.getDevice(name) for name in camera_names]

# LED names and initialization
led_names = [
    "left top led", "left middle up led", "left middle down led", "left bottom led",
    "right top led", "right middle up led", "right middle down led", "right bottom led"
]

leds = [robot.getDevice(name) for name in led_names]

gyro = robot.getDevice("gyro")
gyro.enable(time_step)

init_state = Init_State(motors)

state_dim = 15  # 12 joint angles + 3 gyro readings
action_dim = 12  # 12 motors
action_bound = 1.0  # Motor range

agent = DDPGAgent(state_dim, action_dim, action_bound)

for episode in range(500):  # Train for 500 episodes
    state = np.zeros(state_dim)
    init_state.reset()
    
    total_reward = 0
    for step in range(200):  # Each episode runs for 200 steps
        joint_angles = [motor.getPositionSensor().getValue() for motor in motors]
        gyro_readings = gyro.getValues()
        state = np.concatenate([joint_angles, gyro_readings])

        action = agent.select_action(state)
        for i, motor in enumerate(motors):
            motor.setPosition(action[i])

        robot.step(time_step)

        # Reward function
        forward_velocity = gyro_readings[0]
        reward = forward_velocity - 0.1 * np.sum(np.abs(action))  # Encourage movement, penalize excessive action
        total_reward += reward

        next_joint_angles = [motor.getPositionSensor().getValue() for motor in motors]
        next_gyro_readings = gyro.getValues()
        next_state = np.concatenate([next_joint_angles, next_gyro_readings])

        agent.store_transition(state, action, reward, next_state)
        agent.update()

        state = next_state

    print(f"Episode {episode}, Total Reward: {total_reward}")

