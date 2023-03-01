import numpy as np

from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math_utils import not_zero, wrap_to_pi
from model import Resnet, Resnet_Categorize, ViT
import torch
from torchvision import transforms
from torchvision.utils import save_image
from panda3d.core import Texture
from direct.gui.OnscreenImage import OnscreenImage


class RGBPolicy(BasePolicy):

    MAX_SPEED = 30
    PATH = "model_vit.pt"
    
    def __init__(self, control_object, random_seed):
        super(RGBPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        # self.model = Resnet_Categorize(mode='linear',pretrained=True)
        # checkpoint = torch.load(self.PATH, map_location=torch.device('cpu'))
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.target_speed = self.NORMAL_SPEED

        if 'categorize' in RGBPolicy.PATH:
            self.model = Resnet_Categorize()
        elif 'vit' in RGBPolicy.PATH:
            self.model = ViT(image_size = 128,
                                patch_size = 16,
                                num_classes = 2,
                                dim = 192,
                                depth = 8,
                                heads = 4,
                                dim_head = 48,
                                mlp_dim = 768,
                                pool = 'cls',
                                dropout = 0.1,
                                emb_dropout = 0.1
                            )
        else:
            self.model = Resnet()
        self.model.load_state_dict(torch.load(RGBPolicy.PATH, map_location=torch.device('cpu')))
        self.model.eval()

    def act(self, *args, **kwargs):
        # all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        img = self.control_object.image_sensors["rgb_camera"].get_image(self.control_object)

        myTextureObject = Texture()
        myTextureObject.load(img)
        #OnscreenImage(image = myTextureObject

        # PNMImage to tensor
        img = self.__convert_img_to_tensor(myTextureObject)

        if 'vit' in RGBPolicy.PATH:
            data_transform = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            data_transform = transforms.Compose([
                transforms.CenterCrop((96,96)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img = data_transform(img)

        # From the notebook
        mapping = {
            0: (0.12388312854632436, 0.2091152917264203),
            1: (-0.12438725769093287, 0.34663663748581514),
            2: (-0.00500231837265083, 0.3247432458997098)}

        if 'categorize' not in RGBPolicy.PATH:
            if 'vit' in RGBPolicy.PATH:
                action = self.model(img).detach().numpy()
                action[0] = action[0]*0.06545050570131895 + 0.005975310273351187
                action[1] = action[1]*0.37149717438120655 + 0.3121460530513671
            else:
                action = self.model(img)[0].detach().numpy()
                action[0] = action[0]*0.06545050570131895 + 0.005975310273351187
                action[0] *= 2.3
                action[1] = action[1]*0.37149717438120655 + 0.3121460530513671

            print("MODEL PREDICTION:")
            print(action)

        else:
            logits = self.model(img)[0].detach().numpy()
            # TODO: make those scaling factors learnable
            logits[1] += 1.6
            logits[2] -= 1.2
            category = np.argmax(logits)

            print("MODEL PREDICTION:")
            print(logits)

            action = np.zeros(2)
            action[0] = mapping[category][0] * 1.1
            action[1] = mapping[category][1]

        if self.control_object.speed > RGBPolicy.MAX_SPEED:
            action[1] = 0

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