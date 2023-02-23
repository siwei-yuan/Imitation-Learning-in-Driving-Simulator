import numpy as np

from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math_utils import not_zero, wrap_to_pi
from model import Resnet
import torch
from torchvision import transforms
from torchvision.utils import save_image

class RGBPolicy(BasePolicy):

    MAX_SPEED = 10
    PATH = "model.pt"
    
    def __init__(self, control_object, random_seed):
        super(RGBPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.model = Resnet(mode='linear',pretrained=True)
        checkpoint = torch.load(self.PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.target_speed = self.NORMAL_SPEED

    def act(self, *args, **kwargs):
        # all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        # img: PNMImage
        img = self.control_object.image_sensors["rgb_camera"].get_image(self.control_object)
        print(img)
        # myTextureObject = Texture()
        # myTextureObject.load(img)
        # OnscreenImage(image = myTextureObject)

        # PNMImage to tensor
        img = self.__convert_img_to_tensor(img)

        data_transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img = data_transform(img)

        action = self.model(img)[0].detach().numpy()
        action[0] = action[0]/100
        action[1] = action[1]/10

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

    def __convert_img_to_tensor(self, img):
        height = img.get_read_x_size()
        width = img.get_read_y_size()

        img_tensor = torch.zeros((1, 3, height, width))

        for x in range(height):
            for y in range(width):
                img_tensor[0,0,y,x] = img.get_red(x,y)
                img_tensor[0,1,y,x] = img.get_green(x,y)
                img_tensor[0,2,y,x] = img.get_blue(x,y)
        
        return img_tensor
