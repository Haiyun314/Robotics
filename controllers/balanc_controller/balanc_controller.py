from controller import Supervisor
import numpy as np
import tensorflow as tf

class Env:
    def __init__(self, Supervisor):
        # Create Supervisor instance
        self.supervisor = Supervisor()

        # Time step
        self.timestep = int(self.supervisor.getBasicTimeStep())

        # Access robots
        self.robot_A = self.supervisor.getFromDef('motor')

        self.robot_B = self.supervisor.getFromDef('object')

    def object(self):
        robot_B_translation = self.robot_B.getField('translation') # ball current position [x, y, z]
        return robot_B_translation

    def motor(self):
        # Access their translation fields
        robot_A_rotation = self.robot_A.getField('rotation') # motor current rotation [x=0, y=1, z=0, angle]
        return robot_A_rotation

    def done(self):
        # Check if the robot has fallen
        if self.robot_A_translation[1] < 0.1:
            return True
        return False

    def reset(self):
        self.supervisor.simulationReset()

    def get_state(self):
        return self.motor, self.object, self.done



def nn(input_shape, output_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(output_shape, activation='tanh')(x)
    return tf.keras.Model(inputs=input, outputs=output)

def policy_network(input, epochs=1000):
    model = nn(input_shape=2, output_shape=1)
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            action = model(input)
            loss = tf.reduce_mean(action)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model

def main():
    env_init = Env(Supervisor)

    # Main simulation loop
    while supervisor.step(timestep) != -1:
        # Reset robot A's position to initial state
        ve = robot_A_rotation.getSFFloat() # Reset to origin
        # Reset robot B's position to initial state
        translation = robot_B_translation.getSFFloat()  # Set to some initial position
        
        # robot_A_rotation.setSFFloat([0, 1, 0, position])
        # robot_B_translation.setSFFloat([0.6, 0, 1.5])
        print(ve, translation)

if __name__ == '__main__':
    main()


        
