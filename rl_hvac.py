import numpy as np
import pandas as pd
from datetime import timedelta
from simplesimple import Building

state_table = pd.read_csv(r'X:\Learning\RL\pycodes\data\state_table.csv')
action_table = pd.read_csv(r'X:\Learning\RL\pycodes\data\action_table.csv')

class building:
    def __init__(self, conditioned_floor_area, heat_transmission, maximum_cooling_power, maximum_heating_power,
                 initial_building_temperature, time_step_size):
        self.conditioned_floor_area = conditioned_floor_area
        self.heat_mass_capacity = 165000*self.conditioned_floor_area
        self.heat_transmission = heat_transmission
        self.maximum_cooling_power = maximum_cooling_power
        self.maximum_heating_power = maximum_heating_power
        self.initial_building_temperature = initial_building_temperature
        self.time_step_size = time_step_size
        self.b_obj = Building(self.heat_mass_capacity, self.heat_transmission, self.maximum_cooling_power,
                         self.maximum_heating_power, self.initial_building_temperature, self.time_step_size,
                         self.conditioned_floor_area)
        print('Initial building temperature : ', self.b_obj.current_temperature)

    def next_state(self, outside_temperature_t, action_t):
        self.b_obj.step(outside_temperature=outside_temperature_t, heating_setpoint=-10, cooling_setpoint=80, hc_power=action_t)
        return round(self.b_obj.current_temperature, 2)

class reward:
    def __init__(self, comfortable_temp_lower, comfortable_temp_upper):
        self.comfortable_temp_lower = comfortable_temp_lower
        self.comfortable_temp_upper = comfortable_temp_upper

    def calculate(self, outside_temperature_t, action_t):
        self.state_t_next = tech_m_building.next_state(outside_temperature_t, action_t)
        self.reward_t_next = - action_t - 1000*(np.abs(self.comfortable_temp_upper - self.state_t_next) + np.abs(self.state_t_next - self.comfortable_temp_lower))
        return self.reward_t_next, self.state_t_next




### Define Q-Table
q_table = np.zeros([state_table.shape[0], action_table.shape[0]])

### Generate outside temperature
ot = 30
ot_list = []
bt_list = []

for i in range(10000):
    if i < 5000:
        ot = ot + 0.01*np.random.randn()
    else:
        ot = ot - 0.01 * np.random.randn()
    ot_list.append(ot)


### Do Q-Learning
epsilon = 0.8
alpha = 0.1
gamma = 0.1
state_t = 28
rwd = reward(18,26)

### Define building
tech_m_building = building(conditioned_floor_area = 100, heat_transmission = 200, maximum_cooling_power = -6000, maximum_heating_power = 6000,
                 initial_building_temperature = state_t, time_step_size = timedelta(minutes=1))

for i in range(10000):
    ot = round(ot_list[i], 2)
    state_t_index = state_table[state_table['state'] == state_t].index[0]
    if np.random.rand() < epsilon:
        # Explore action space
        action_t = action_table.sample(1)
        action_t_index = action_t.index[0]
        action_t_value = action_t.values[0][0]
    else:
        # Explore action space

        action_t_index = np.argmax(q_table[state_t_index])
        action_t_value = np.max(q_table[state_t_index])

    reward_t_next, state_t_next = rwd.calculate(ot, action_t_value)

    q_value_old = q_table[state_t_index, action_t_index]
    state_t_next_index = state_table[state_table['state'] == state_t_next].index[0]
    q_value_state_t_next_max = np.max(q_table[state_t_next_index])

    q_value_new = (1 - alpha) * q_value_old + alpha * (reward_t_next + gamma * q_value_state_t_next_max)
    q_table[state_t_index, action_t_index] = q_value_new

    state_t = state_t_next
    bt_list.append(state_t_next)