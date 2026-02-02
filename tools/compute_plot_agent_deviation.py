#%%
import os
import sys
CARLA_ROOT="/home/ishan/dev/carla/CARLA_0.9.10.1"
sys.path.append(f"{CARLA_ROOT}/PythonAPI")
sys.path.append(f"{CARLA_ROOT}/PythonAPI/carla")
sys.path.append(f"{CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg")
sys.path.append("/home/ishan/dev/repos/github/av-attack-detection")
sys.path.append("/home/ishan/dev/repos/github/av-attack-detection/scenario_runner")
import carla
import math
import numpy as np
from matplotlib import pyplot as plt

from srunner.metrics.tools.metrics_log import MetricsLog
from srunner.metrics.examples.distance_to_lane_center import DistanceToLaneCenter

# %%
client = carla.Client('localhost', 2000)
#client = carla.Client('192.168.0.201', 2000)
world = client.get_world()

# %%
models = ['transfuser', 'interfuser']
folders = ['esia_s1_d1',
           'esia_s1_d4',
           'esia_s1_d11',
           'esia_s2_d1',
           'esia_s2_d4',
           'esia_s2_d11',
           'esia_s3_d1',
           'esia_s3_d4',
           'esia_s3_d11']
filename = 'RouteScenario_0_rep0.log'

base_path = '/home/ishan/Downloads/recordings/'

#%%
k = 1
j = 0

for j in range(len(folders)):
    m = models[k]
    f = folders[j]
    recorder_filepath = os.path.join(base_path, m, f, filename)

    print(recorder_filepath)

    client.set_timeout(600.0)
    recorder_str = client.show_recorder_file_info(recorder_filepath, True)

    header = recorder_str.split("\n")
    sim_map = header[1][5:]
    print(sim_map)

    world = client.load_world(sim_map)
    town_map = world.get_map()

    log = MetricsLog(recorder_str)

    del(header)

    ego_id = log.get_ego_vehicle_id()
    print(ego_id)

    dist_list = []
    frames_list = []

    # Get the frames the ego actor was alive and its transforms
    start, end = log.get_actor_alive_frames(ego_id)

    # Get the projected distance vector to the center of the lane
    for i in range(start, end + 1):

        ego_location = log.get_actor_transform(ego_id, i).location
        ego_waypoint = town_map.get_waypoint(ego_location)

        # Get the distance vector and project it
        a = ego_location - ego_waypoint.transform.location      # Ego to waypoint vector
        b = ego_waypoint.transform.get_right_vector()           # Waypoint perpendicular vector
        b_norm = math.sqrt(b.x * b.x + b.y * b.y + b.z * b.z)

        ab_dot = a.x * b.x + a.y * b.y + a.z * b.z
        dist_v = ab_dot/(b_norm*b_norm)*b
        dist = math.sqrt(dist_v.x * dist_v.x + dist_v.y * dist_v.y + dist_v.z * dist_v.z)

        # Get the sign of the distance (left side is positive)
        c = ego_waypoint.transform.get_forward_vector()         # Waypoint forward vector
        ac_cross = c.x * a.y - c.y * a.x
        if ac_cross < 0:
            dist *= -1

        dist_list.append(dist)
        frames_list.append(i)

    # Save the results to a file
    results = {'frames': frames_list, 'distance': dist_list}

    # for report
    label_size = 22
    title = f'{m}_{f}'
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['frames'], results['distance'])
    ax.set_xlabel('Frame number', fontsize=label_size)
    ax.set_ylabel('Distance from lane centre (m)', fontsize=label_size)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    #ax.set_title(title)
    plt.savefig(f"report_{title.replace('/', '-')}.png", dpi=150, bbox='tight')
    plt.show()

    # compute avg distance from lane center.
    avg_c_dist = np.mean(np.abs(results['distance']))
    std_c_dist = np.std(np.abs(results['distance']))

    print(title)
    print(f'Avg distance from the centre: {avg_c_dist}')
    print(f'Std deviation: {std_c_dist}')

    # save distance results to pickle file
    import pickle as pkl
    file_path = f"{title.replace('/', '-')}.pkl"
    with open(file_path, 'wb') as file:
        pkl.dump(results, file)

    print('*'*10)

# %%
