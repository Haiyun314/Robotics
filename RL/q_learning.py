import numpy as np

Nstates = 10
Nactions = 2
q_table = np.zeros((Nstates, Nactions))
class Env:
    def __init__(self, Nstates, state, target):
        assert 0<= target <= Nstates-1, f'target {target} not in the enviroment'
        self.state = state
        self.target = target
        self.Nstates = Nstates
        self.env = np.zeros(Nstates)

    def reset(self):
        self.env = np.zeros(self.Nstates)
        self.env[self.state] = 1

    def step(self,action):
        if action=='left':
            if self.state == 0:
                self.state = self.state
            else:
                self.state += -1
        else: #right
            if self.state == self.Nstates-1:
                self.state == self.Nstates -1
            else:
                self.state += 1
    
def q_next(state, action):
    q_table += 
