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

import pygame

PRINT_IMG = False
SAMPLING_INTERVAL = 10


if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        offscreen_render=True,
        manual_control=True,
        traffic_density=0,
        environment_num=100,
        random_agent_model=False,
        start_seed=random.randint(0, 1000),
        vehicle_config = dict(image_source="rgb_camera", 
                              rgb_camera= (128 , 128),
                              stack_size=1),
        block_dist_config=PGBlockDistConfig,
        random_lane_width=True,
        random_lane_num=False,
        map_config={
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: "CCCSCCC",  # it can be a file path / block num / block ID sequence
            BaseMap.LANE_WIDTH: 4,
            BaseMap.LANE_NUM: 1,
            "exit_length": 50,
        },
        agent_policy=RGBPolicy
    )
    env = MetaDriveEnv(config)

    imgs = []
    frames = []
    try:
        o = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        assert isinstance(o, dict)
        print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        #for i in range(1, 31):
        for i in range(1, 400000):
            o, r, d, info = env.step([0, 0])
            
            # Action space is of form (float, float) -> Tuple
            # It encodes the necessary info about vehicle movement
            action_space = (info['steering'], info['acceleration'], info['velocity'])
            #print(action_space)
            
            # Change PRINT_IMG to True if recording FPV
            # Change step_size to set sampling rate
            # Image saved is named by the action
            if PRINT_IMG and i%SAMPLING_INTERVAL == 0:
                img = np.squeeze(o['image'][:,:,:,0]*255).astype(np.uint8)
                img = img[...,::-1].copy() # convert to rgb
                img = Image.fromarray(img)
                root_dir = os.path.join(os.getcwd(), 'dataset', 'val')
                img_path = os.path.join(root_dir, str(action_space) + ".png")
                img.save(str(img_path))
            

            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if d and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True

        #imgs[0].save("demo2.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)

    except Exception as e:
        raise e
    finally:
        env.close()