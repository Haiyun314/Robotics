import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # Gravity (m/s^2)
m1, m2 = 1.0, 1.0  # Masses (kg)
l1, l2 = 1.0, 1.0  # Lengths (m)

# Equations of motion
def derivatives(y, t):
    theta1, omega1, theta2, omega2 = y

    # Denominators
    delta = theta1 - theta2
    denom1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
    denom2 = (l2 / l1) * denom1

    # Equations
    dtheta1_dt = omega1
    domega1_dt = (-g * (m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2*theta2) -
                  2 * m2 * l2 * omega2**2 * np.sin(delta) - m2 * l1 * omega1**2 * np.sin(2*delta)) / denom1
    dtheta2_dt = omega2
    domega2_dt = (2 * np.sin(delta) * ((m1 + m2) * l1 * omega1**2 + g * (m1 + m2) * np.cos(theta1) +
                  m2 * l2 * omega2**2 * np.cos(delta))) / denom2

    return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])

# RK4 method
def rk4_step(f, y, t, h):
    k1 = f(y, t)
    k2 = f(y + h * k1 / 2, t + h / 2)
    k3 = f(y + h * k2 / 2, t + h / 2)
    k4 = f(y + h * k3, t + h)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Initial conditions
theta1_0, omega1_0 = np.pi / 2, 0.0  # Initial angle and velocity of first pendulum
theta2_0, omega2_0 = np.pi / 2, 0.0  # Initial angle and velocity of second pendulum
y0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0])

# Time settings
t_start, t_end, dt = 0, 10, 0.01
time = np.arange(t_start, t_end, dt)

# Solve the system
trajectory = np.zeros((len(time), len(y0)))
trajectory[0] = y0

for i in range(1, len(time)):
    trajectory[i] = rk4_step(derivatives, trajectory[i-1], time[i-1], dt)

# Extract results
theta1, theta2 = trajectory[:, 0], trajectory[:, 2]

# Visualization
plt.plot(time, theta1, label="Theta 1")
plt.plot(time, theta2, label="Theta 2")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.show()
