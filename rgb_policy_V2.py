import numpy as np

from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math_utils import not_zero, wrap_to_pi
from model import ViT_V2
import torch
from torchvision import transforms
from torchvision.utils import save_image
from panda3d.core import Texture
from direct.gui.OnscreenImage import OnscreenImage


class RGBPolicy_V2(BasePolicy):

    #MAX_SPEED = 40
    PATH = "model_vit_V2_2k_grayscale.pt"
    
    def __init__(self, control_object, random_seed):
        super(RGBPolicy_V2, self).__init__(control_object=control_object, random_seed=random_seed)

        self.model = ViT_V2(image_size = 128,
                            patch_size = 16,
                            num_classes = 3,
                            dim = 192,
                            depth = 8,
                            heads = 4,
                            dim_head = 48,
                            mlp_dim = 768,
                            pool = 'cls',
                            dropout = 0.1,
                            emb_dropout = 0.1)
        self.model.load_state_dict(torch.load(RGBPolicy_V2.PATH, map_location=torch.device('cpu')))
        self.model.eval()

    def act(self, *args, **kwargs):

        img = self.control_object.image_sensors["rgb_camera"].get_image(self.control_object)

        myTextureObject = Texture()
        myTextureObject.load(img)
        img = self.__convert_img_to_tensor(myTextureObject)

        data_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img = data_transform(img)

        speed = (self.control_object.speed-37.906848060080954)/7.288086708484034
        speed = torch.unsqueeze(torch.tensor([speed]), 0)
        speed = speed.repeat(192, 1).t()

        action = self.model(img, speed).detach().numpy()


        steering = action[0] = action[0]*0.06458930098761928 + 0.0027872782659134248
        accel = action[1]= action[1]*0.3710675398605016 + 0.2993908500203321
        action[2] = action[2]*7.288086708484034 + 37.906848060080954
        
        if steering < 0:
            steering *= 0.85

        print("MODEL PREDICTION:")
        print(action)

        return [steering, accel]


    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance control
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)


    def acceleration(self, front_obj, dist_to_front) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))
        if front_obj and (not self.disable_idm_deceleration):
            d = dist_to_front
            speed_diff = self.desired_gap(ego_vehicle, front_obj) / not_zero(d)
            acceleration -= self.ACC_FACTOR * (speed_diff**2)
        return acceleration


    def reset(self):
        self.heading_pid.reset()
        self.lateral_pid.reset()
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)


    def __convert_img_to_tensor(self, myTextureObject):

        img = np.frombuffer(myTextureObject.getRamImageAs("RGBA").getData(), dtype=np.uint8)
        img = img.reshape((myTextureObject.getYSize(), myTextureObject.getXSize(), 4))
        img = img[::-1]
        img = img[...,:-1] - np.zeros_like((128,128,3))
        img = torch.from_numpy(img)
        img = img/255
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        return img