# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

from controller import Supervisor, Receiver
import numpy as np

class Init_Sup:
    def __init__(self, name: str, super, time_step):
        self.time_step = time_step
        self.supervisor = super

        self.robot_node = self.supervisor.getFromDef(name)

        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.initial_translation = self.translation_field.getSFVec3f()  # [x, y, z] position
        self.initial_rotation = self.rotation_field.getSFRotation()    # [axis_x, axis_y, axis_z, angle]

        self.receiver = self.supervisor.getDevice('receiver')  # Assuming 'receiver' is available
        self.receiver.enable(self.time_step)

    def receive_message(self):
        if self.receiver.getQueueLength() > 0:
            message = self.receiver.getString()
            print("Message received:", message)
            self.receiver.nextPacket()  # Clear the packet
            return message
        return None

    def reset(self):
        self.translation_field.setSFVec3f(self.initial_translation)
        self.rotation_field.setSFRotation(self.initial_rotation)
        self.supervisor.simulationResetPhysics()
    

if __name__ == '__main__':
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())

    init_sup = Init_Sup('robot', super= supervisor, time_step= time_step)
    while init_sup.supervisor.step(init_sup.time_step) != -1:
        info = init_sup.receive_message()
        if info == 'reset':
            init_sup.reset()
