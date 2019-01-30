from datetime import timedelta
from simplesimple import Building
import numpy as np
from matplotlib import pyplot as plt

conditioned_floor_area = 100
building = Building(
    heat_mass_capacity=165000 * conditioned_floor_area,
    heat_transmission=200,
    maximum_cooling_power=-10000,
    maximum_heating_power=10000,
    initial_building_temperature=30,
    time_step_size=timedelta(minutes=1),
    conditioned_floor_area=conditioned_floor_area
)

# simulate one time step
print(building.current_temperature) # returns 16
ot = 1
ot_list = []
bt_list = []
for i in range(3000):
    """
    if i < 1000:
        ot = ot + 0.2*np.random.randn()
    else:
        ot = ot - 0.3 * np.random.randn()
    
    """
    ot_list.append(ot)
    building.step(outside_temperature=ot, heating_setpoint=-10, cooling_setpoint=50, hc_power=-1000)
    bt_list.append(building.current_temperature)
    #print(building.current_temperature) # returns ~16.4