import numpy as np

class bandit_stationary:
    def __init__(self, k):
        self.reward = np.zeros(shape=(1,k))
        self.arm_no = 0

    def pull_arm(self, arm_no):
        self.arm_no = arm_no
        self.reward[0][self.arm_no] = [100 if np.random.rand() > 0.5 else 0][0]
        return self.reward[0]



class bandit_non_stationary:
    def __init__(self, k):
        self.reward = np.zeros(10)
        self.reward_mean = np.random.randn(10)
        print(self.reward_mean)
        self.arm_no = 0

    def show_rewards(self):
        return self.reward_mean

    def pull_arm(self, arm_no):
        self.arm_no = arm_no
        self.reward[self.arm_no] = 1*np.random.randn() + self.reward_mean[self.arm_no]                      ### N(mu, sigma^2)  =  sigma*np.random.randn() + mu
        return {'arm_number' : self.arm_no, 'reward': self.reward[self.arm_no]}


bandit_machine = bandit_non_stationary(10)
reward_acc = []
for i in range(100000):
    reward_acc.append(bandit_machine.pull_arm(8)['reward'])