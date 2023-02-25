import numpy as np

from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math_utils import not_zero, wrap_to_pi
from model import Resnet
import torch
from torchvision import transforms
from torchvision.utils import save_image
from panda3d.core import Texture
from direct.gui.OnscreenImage import OnscreenImage

class RGBPolicy(BasePolicy):

    MAX_SPEED = 10
    PATH = "model.pt"
    
    def __init__(self, control_object, random_seed):
        super(RGBPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.model = Resnet(mode='linear',pretrained=True)
        checkpoint = torch.load(self.PATH, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.target_speed = self.NORMAL_SPEED

    def act(self, *args, **kwargs):
        # all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        img = self.control_object.image_sensors["rgb_camera"].get_image(self.control_object)

        myTextureObject = Texture()
        myTextureObject.load(img)
        #OnscreenImage(image = myTextureObject

        # PNMImage to tensor
        img = self.__convert_img_to_tensor(myTextureObject)

        data_transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img = data_transform(img)

        action = self.model(img)[0].detach().numpy()

        action[0] = action[0]*0.06545050570131895 + 0.005975310273351187
        action[1] = action[1]*0.37149717438120655 + 0.3121460530513671

        print("MODEL PREDICTION:")
        print(action)

        return action

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