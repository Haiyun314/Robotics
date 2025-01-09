import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

# Custom balancing environment (use gym CartPole or Pendulum for testing)
class BalanceSystem:
    def __init__(self):
        self.state = np.array([0.0, 0.0])  # [angle, angular velocity]
        self.dt = 0.05  # Time step
        self.max_angle = 90.0
        self.max_velocity = 1.0

    def reset(self):
        self.state = np.random.uniform(low=[-10, -0.1], high=[10, 0.1])
        return self.state

    def step(self, action):
        angle, velocity = self.state
        velocity += action * self.dt
        angle += velocity * self.dt

        # Reward is higher when the pole is upright
        reward = -abs(angle)
        done = abs(angle) > self.max_angle

        self.state = np.clip([angle, velocity], [-self.max_angle, -self.max_velocity], [self.max_angle, self.max_velocity])
        return self.state, reward, done

# Neural networks for actor and critic
def build_actor(state_dim, action_dim, action_bound):
    inputs = tf.keras.Input(shape=(state_dim,))
    out = tf.keras.layers.Dense(256, activation="relu")(inputs)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    outputs = tf.keras.layers.Dense(action_dim, activation="tanh")(out)
    outputs = outputs * action_bound
    return tf.keras.Model(inputs, outputs)

def build_critic(state_dim, action_dim):
    state_input = tf.keras.Input(shape=(state_dim,))
    action_input = tf.keras.Input(shape=(action_dim,))
    concat = tf.keras.layers.Concatenate()([state_input, action_input])

    out = tf.keras.layers.Dense(256, activation="relu")(concat)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    outputs = tf.keras.layers.Dense(1)(out)
    return tf.keras.Model([state_input, action_input], outputs)

# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.size = 0

    def add(self, transition):
        if self.size < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.size % self.max_size] = transition
        self.size += 1

    def sample(self, batch_size):
        indices = np.random.choice(min(self.size, self.max_size), batch_size)
        return [self.buffer[i] for i in indices]

# Hyperparameters
state_dim = 2
action_dim = 1
action_bound = 1.0
lr_actor = 0.001
lr_critic = 0.002
gamma = 0.99
tau = 0.005  # For soft target updates
buffer_size = 100000
batch_size = 64

# Initialize environment and models
env = BalanceSystem()
actor = build_actor(state_dim, action_dim, action_bound)
critic = build_critic(state_dim, action_dim)
target_actor = build_actor(state_dim, action_dim, action_bound)
target_critic = build_critic(state_dim, action_dim)

# Copy weights to target networks
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

# Optimizers
actor_optimizer = tf.keras.optimizers.Adam(lr_actor)
critic_optimizer = tf.keras.optimizers.Adam(lr_critic)

# Replay buffer
replay_buffer = ReplayBuffer(buffer_size)

# Noise for exploration
class OrnsteinUhlenbeckNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=0.01):
        self.mean = mean
        self.std_dev = std_dev
        self.theta = theta
        self.dt = dt
        self.reset()

    def reset(self):
        self.x_prev = np.zeros_like(self.mean)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x

noise = OrnsteinUhlenbeckNoise(mean=np.zeros(action_dim), std_dev=0.2)

# Training loop
episodes = 100
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action = actor(state_tensor)[0].numpy() + noise()
        next_state, reward, done = env.step(action)
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        # Train when buffer has enough samples
        if replay_buffer.size >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            next_states = np.array(next_states)
            dones = np.array(dones).reshape(-1, 1)

            # Critic update
            next_actions = target_actor(next_states)
            target_q = rewards + gamma * target_critic([next_states, next_actions]) * (1 - dones)
            with tf.GradientTape() as tape:
                q = critic([states, actions])
                critic_loss = tf.reduce_mean(tf.square(q - target_q))
            critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

            # Actor update
            with tf.GradientTape() as tape:
                actions_pred = actor(states)
                actor_loss = -tf.reduce_mean(critic([states, actions_pred]))
            actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

            # Soft target updates
            for target_param, param in zip(target_actor.trainable_variables, actor.trainable_variables):
                target_param.assign(tau * param + (1 - tau) * target_param)
            for target_param, param in zip(target_critic.trainable_variables, critic.trainable_variables):
                target_param.assign(tau * param + (1 - tau) * target_param)

    print(f"Episode {episode + 1}: Reward: {episode_reward}")
