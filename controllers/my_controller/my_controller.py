import debugpy
debugpy.listen(("localhost", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
import time
import random
from controller import Robot, Motor, Camera, LED, Emitter
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import os



def build_actor(state_dim, action_dim):
    inputs = Input(shape=(state_dim,))
    x = Dense(64, activation="relu")(inputs)
    x = Dense(64, activation="tanh")(x)
    x = Dense(64, activation="tanh")(x)
    outputs = Dense(action_dim, activation="tanh")(x)  # Actions between -1 and 1
    return Model(inputs, outputs)

def build_critic(state_dim, action_dim):
    state_input = Input(shape=(state_dim,))
    action_input = Input(shape=(action_dim,))
    x = tf.keras.layers.Concatenate()([state_input, action_input])
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="tanh")(x)
    x = Dense(64, activation="tanh")(x)
    outputs = Dense(1)(x)  # Q-value estimation
    return Model([state_input, action_input], outputs)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, memory_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.high_reward_memory_size, self.low_reward_memory_size = memory_size

        # try:
        #     self.actor = tf.keras.models.load_model(Root_path + '/model/actor.h5')
        # except Exception as e:  
        #     print(f"Actor Model loading failed: {e}")
        self.actor = build_actor(state_dim, action_dim)
        # try:
        #     self.critic = tf.keras.models.load_model(Root_path + '/model/critic.h5')
        # except Exception as e: 
        #     print(f"Critic Model loading failed: {e}")
        self.critic = build_critic(state_dim, action_dim)
        
        self.target_actor = build_actor(state_dim, action_dim)
        self.target_critic = build_critic(state_dim, action_dim)
        
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
        
        self.memory = Memory( self.low_reward_memory_size, self.high_reward_memory_size)
        self.gamma = 0.95  # Discount factor
        self.tau = 0.1  # Soft update rate

    def select_action(self, state):
        state = np.reshape(state, (1, self.state_dim))
        action = self.actor.predict(state)[0]
        return action * self.action_bound  # Scale to action range

    def update(self):
        if any([len(self.memory.high_reward_memory) < self.memory.high_reward_size-1,
        len(self.memory.low_reward_memory) < self.memory.low_reward_size-1]): return

        batch = self.memory.sample(64)
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
        self.memory.store(state, action, reward, next_state)

           
class Memory:
    def __init__(self, low_reward_size, high_reward_size):
        self.low_reward_memory = []
        self.high_reward_memory = []  # Separate buffer for high rewards
        self.low_reward_size = low_reward_size
        self.high_reward_size = high_reward_size
        self.init_size = 10
        self.reward_baseline = 100 # init reward baseline 

    def store(self, state, action, reward, next_state):
        transition = [state, action, reward, next_state]
        print(f"length of high_reward_memory **** { len(self.high_reward_memory)} \n"
                f'length_of_low_reward_memory **** {len(self.low_reward_memory)} \n'
                f'averaged high reward **** {self.reward_baseline} \n')
        self.reward_baseline = (self.reward_baseline + transition[2])/2
        while self.init_size >= 0:
            self.init_size -= 1
            self.high_reward_memory.append(transition)
            return None
        # Check if the transition is high reward
        if transition[2] > self.reward_baseline:  
            if len(self.high_reward_memory) < self.high_reward_size:
                self.high_reward_memory.append(transition)
            else:
                self.high_reward_memory.pop(0)  # Remove oldest high-reward transition
                self.high_reward_memory.append(transition)
        else:
            if len(self.low_reward_memory) < self.low_reward_size:
                self.low_reward_memory.append(transition)
            else:
                self.low_reward_memory.pop(0)
                self.low_reward_memory.append(transition)

    def sample(self, batch_size):
        # Sample 25% from high-reward buffer, 75% from regular buffer
        high_reward_batch_size = int(0.25 * batch_size)
        regular_batch_size = batch_size - high_reward_batch_size

        high_reward_sample = random.sample(self.high_reward_memory, min(len(self.high_reward_memory), high_reward_batch_size))
        regular_sample = random.sample(self.low_reward_memory, min(len(self.low_reward_memory), regular_batch_size))

        return high_reward_sample + regular_sample


class Robot_Control:
    NUMBER_OF_DEVICES = 13
    def __init__(self, robot, time_step):
        self.robot = robot
        self.time_step = time_step
        self.motor_names = ['imu_sensor', 'left_wheel_joint', 'left_wheel_joint_sensor', 
               'right_wheel_joint', 'right_wheel_joint_sensor', 'left_hip_joint', 
               'left_hip_joint_sensor', 'left_knee_joint', 'left_knee_joint_sensor', 
               'right_hip_joint', 'right_hip_joint_sensor', 'right_knee_joint', 'right_knee_joint_sensor'
                ]
        self.motors = [self.robot.getDevice(name) for name in self.motor_names]
        # Ensure motors are in velocity control mode
        self.motors[1].setPosition(float('inf'))  # Left wheel motor
        self.motors[3].setPosition(float('inf'))  # Right wheel motor

        self.emitter = self.robot.getDevice('robot_emitter') 
        self.emitter.setChannel(1) 


    def imu_enable(self):
        imu = self.robot.getDevice(self.motor_names[0])
        imu.enable(self.time_step)
        return imu

    def set_wheels_velocity(self, left_velocity:float, right_velocity: float):
        left_velocity = np.clip(left_velocity, -10, 10)
        right_velocity = np.clip(right_velocity, -10, 10)
        self.motors[1].setVelocity(left_velocity) # Make the movement more radical
        self.motors[3].setVelocity(right_velocity)

    def get_wheels_velocity(self):
        left_wheel_velocity = self.motors[1].getVelocity()
        right_wheel_velocity = self.motors[3].getVelocity()
        return left_wheel_velocity, right_wheel_velocity

    def send_message_to_supervisor(self, message):
        self.emitter.send(message.encode('utf-8'))

if __name__ == '__main__':
    Root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    rc = Robot()
    time_step = int(rc.getBasicTimeStep())
    rc = Robot_Control(rc, time_step)
    imu = rc.imu_enable()
    rc.set_wheels_velocity(0, 0)
    
    acc_time_interval = time_step/100

    state_dim = 3  # imu result
    action_dim = 2  # left and right wheels
    action_bound = 1.0  # Motor range

    agent = DDPGAgent(state_dim, action_dim, action_bound, (500, 500))
    episode  = 0

    while rc.robot.step(time_step) != -1:
        noise_scale = max(0.1, 10 * (0.99 ** episode)) 

        init_noise = np.random.uniform(-noise_scale, noise_scale, size= 2)
        rc.send_message_to_supervisor('reset')
        state = imu.getRollPitchYaw()
        print(state)
        rc.set_wheels_velocity(init_noise[0], init_noise[1])

        episode += 1
        total_reward = 0
        time_r = 1 # time reward
        start_timer = time.perf_counter()
        while True: # stop when it falls
            state = imu.getRollPitchYaw()
            noise = np.random.uniform(-0.2, 0.2, size=action_dim)
            action = agent.select_action(state) + noise

            left_velocity, right_velocity = rc.get_wheels_velocity()

            left_velocity += action[0]*10 * acc_time_interval
            right_velocity += action[1]*10 * acc_time_interval
            rc.set_wheels_velocity(left_velocity, right_velocity)

            rc.robot.step(time_step)

            # Reward function
            state_r = np.tanh(1 - np.sum(np.abs(state[0]))*2 ) * 10
            time_r += 0.3
            print(left_velocity, '*'*5 ,right_velocity, '\n')
            reward = state_r + time_r * time_r
            total_reward += reward

            next_state = imu.getRollPitchYaw()
            if state_r <= -4.0: # additional punishment
                reward -= 200
                agent.store_transition(state, action, reward, next_state)
                agent.update()
                state = next_state
                print(f"Episode {episode}, Total Reward: {total_reward}")
                break

            agent.store_transition(state, action, reward, next_state)
            agent.update()

            state = next_state
            print(f"Episode {episode}, Total Reward: {total_reward}")
        if time.perf_counter() - start_timer >= 60:
            tf.keras.models.save_model(agent.actor, Root_path + '/model/actor_v2.h5')
            tf.keras.models.save_model(agent.critic, Root_path + '/model/critic_v2.h5')
            break



