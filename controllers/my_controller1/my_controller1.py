from controller import Robot, Motor, Camera

TIME_STEP = 64
MAX_SPEED = 5

# create the Robot instance.
robot = Robot()

# get a handler to the motors and set target position to infinity (velocity control)
leftMotor = robot.getDevice('w1m')
leftMotor.setPosition(float('inf'))  # infinity for velocity control

cam = robot.getDevice('camera')
cam.enable(TIME_STEP)

# get a handler to the motors and set target position to infinity (velocity control)
rightMotor = robot.getDevice('w2m')
rightMotor.setPosition(float('inf'))  # infinity for velocity control

# get a handler to the motors and set target position to infinity (velocity control)
leftfMotor = robot.getDevice('w3m')
leftfMotor.setPosition(float('inf'))  # infinity for velocity control

# get a handler to the motors and set target position to infinity (velocity control)
rightfMotor = robot.getDevice('w4m')
rightfMotor.setPosition(float('inf'))  # infinity for velocity control


# set up the motor speeds to 10% of the MAX_SPEED.
leftMotor.setVelocity(MAX_SPEED )
rightMotor.setVelocity(MAX_SPEED )
leftfMotor.setVelocity(MAX_SPEED )
rightfMotor.setVelocity(MAX_SPEED )

# main loop: continuously move in a circle
while robot.step(TIME_STEP) != -1:
   pass