import numpy as np
from scipy.linalg import solve_continuous_are, inv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define system parameters
m = 0.1   # Mass of pendulum (kg)
M = 1.0   # Mass of cart (kg)
L = 0.5   # Length of pendulum (m)
g = 9.81  # Acceleration due to gravity (m/s^2)
d = 0.1   # Damping coefficient (friction)

# State-space representation
A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, -(m*g)/M, 0],
    [0, 0, 0, 1],
    [0, d/(M*L), (M+m)*g/(M*L), 0]
])

B = np.array([[0], [1/M], [0], [-1/(M*L)]])

# Cost matrices
Q = np.diag([10, 1, 100, 1])  # Penalize position and angle deviations
R = np.array([[1]])           # Penalize control effort

# Solve the Algebraic Riccati Equation to find K
P = solve_continuous_are(A, B, Q, R)
K = inv(R) @ (B.T @ P)

# Simulation parameters
dt = 0.01  # Time step
time = np.arange(0, 10, dt)

# Initial state: cart at 0, pendulum slightly tilted
x = np.array([[-0.1], [0.0], [-0.3], [0.0]])  # [position, velocity, angle, angular velocity]
states = [x.flatten()]
inputs = []

# Simulate the system
for t in time:
    # Compute control input
    u = -K @ x
    inputs.append(u.item())
    
    # Update state using Euler integration
    x_dot = A @ x + B @ u
    x = x + x_dot * dt
    states.append(x.flatten())

states = np.array(states)

# Animation
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')

# Cart and pendulum visualization
cart, = ax.plot([], [], 'blue', lw=6)  # Cart
pendulum, = ax.plot([], [], 'red', lw=2)  # Pendulum
trail, = ax.plot([], [], 'green', lw=1, alpha=0.6)  # Cart position trail
trail_x = []

# Initialize animation
def init():
    cart.set_data([], [])
    pendulum.set_data([], [])
    trail.set_data([], [])
    return cart, pendulum, trail

# Update animation
def update(frame):
    pos = states[frame, 0]  # Cart position
    angle = states[frame, 2]  # Pendulum angle
    
    # Cart coordinates
    cart_x = [pos - 0.2, pos + 0.2]
    cart_y = [0, 0]
    
    # Pendulum coordinates
    pend_x = [pos, pos + L * np.sin(angle)]
    pend_y = [0, -L * np.cos(angle)]
    
    # Update cart and pendulum
    cart.set_data(cart_x, cart_y)
    pendulum.set_data(pend_x, pend_y)
    
    # Update trail
    trail_x.append(pos)
    trail.set_data(trail_x, [0] * len(trail_x))
    
    return cart, pendulum, trail

ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=dt*1000)

# Display the animation
plt.title("LQR Control: Inverted Pendulum on a Cart")
plt.xlabel("Cart Position (m)")
plt.ylabel("Height (m)")
plt.grid(True)
plt.show()
