"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!
Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""



import random
import os
import numpy as np
from PIL import Image
from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE


PRINT_IMG = True
SAMPLING_INTERVAL = 10


if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        offscreen_render=True,
        manual_control=True,
        traffic_density=0.1,
        environment_num=100,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        map=4,  # seven block
        start_seed=random.randint(0, 1000),
        vehicle_config = dict(image_source="rgb_camera", 
                              rgb_camera= (256 , 128),
                              stack_size=1)
    )
    env = MetaDriveEnv(config)
    try:
        o = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        assert isinstance(o, dict)
        print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        for i in range(1, 20):
            o, r, d, info = env.step([0, 0])
            
            # Action space is of form (float, float) -> Tuple
            # It encodes the necessary info about vehicle movement
            action_space = (info['steering'], info['acceleration'])
            print(action_space)

            # Change PRINT_IMG to True if recording FPV
            # Change step_size to set sampling rate
            # Image saved is named by the action
            if PRINT_IMG and i%SAMPLING_INTERVAL == 0:
                img = np.squeeze(o['image'][:,:,:,0]*255).astype(np.uint8)
                img = img[...,::-1].copy() # convert to rgb
                print(img.shape)
                img = Image.fromarray(img)
                root_dir = os.path.join(os.getcwd(), 'dataset')
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
    except Exception as e:
        raise e
    finally:
        env.close()