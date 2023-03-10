"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!
Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""



import random
import os
import numpy as np
import torch
from PIL import Image
from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod

from rgb_policy import RGBPolicy
from rgb_policy_V2 import RGBPolicy_V2
from rgb_policy_SF import RGBPolicy_SlowFast

import time

if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        offscreen_render=True,
        manual_control=True,
        traffic_density=0.2,
        environment_num=100,
        random_agent_model=False,
        start_seed=random.randint(0, 1000),
        vehicle_config = dict(image_source="rgb_camera", 
                              rgb_camera= (256 , 256),
                              stack_size=1),
        block_dist_config=PGBlockDistConfig,
        random_lane_width=False,
        random_lane_num=False,
        map_config={
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            #BaseMap.GENERATE_CONFIG: "SCCSCCS",  # it can be a file path / block num / block ID sequence
            BaseMap.GENERATE_CONFIG: "SCCSCCS",
            BaseMap.LANE_WIDTH: 4,
            BaseMap.LANE_NUM: 1,
            "exit_length": 50,
        },
        agent_policy = RGBPolicy_SlowFast
    )
    env = MetaDriveEnv(config)

    imgs = []
    frames = []

    st = time.time()
    cost = 0
    try:
        o = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        assert isinstance(o, dict)
        print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        #for i in range(1, 31):
        for i in range(1, 1000000000):
            o, r, d, info = env.step([0, 0])

            cost += info["cost"]
            
            # Action space is of form (float, float) -> Tuple
            # It encodes the necessary info about vehicle movement
            action_space = (info['steering'], info['acceleration'], info['velocity'])
            #print(action_space)
            

            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if d and info["arrive_dest"]:
                print(cost)
                print('Execution time: ', time.time()-st, ' seconds')
                break
                env.reset()
                env.current_track_vehicle.expert_takeover = True

        #imgs[0].save("demo2.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)

    except Exception as e:
        raise e
    finally:
        env.close()
        print(cost)
        print('Execution time: ', time.time()-st, ' seconds')