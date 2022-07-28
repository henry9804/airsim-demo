import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

import matplotlib.pyplot as plt

class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
            "result": None
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(6)

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        if action == 0:                         # brake
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:                       # go straight
            self.car_controls.steering = 0
        elif action == 2:                       # turn right rapidly
            self.car_controls.steering = 0.5
        elif action == 3:                       # turn left rapidly
            self.car_controls.steering = -0.5
        elif action == 4:                       # turn right slowly
            self.car_controls.steering = 0.25
        else:                                   # turn left slowly
            self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([1, 84, 84])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()
        gps_data = self.car.getGpsData("Gps", "Car").gnss.geo_point

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = np.array([gps_data.altitude, gps_data.latitude, gps_data.longitude])
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):
        target_pt = np.array([-82.4, -102.7, 0])
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = np.linalg.norm(target_pt - car_pt)

        reward = 1
        done = 0
        if dist < 1:
            done = 1
            self.state["info"] = "arrive"
        if self.state["collision"]:
            reward = 0
            done = 1
            self.state["info"] = "crash"

        reward += 1/dist

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(1)
        return self._get_obs()
