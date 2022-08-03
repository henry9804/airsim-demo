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
    def __init__(self, ip_address, image_shape, target):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.elapsed_t = 0
        self.target_pos = target

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": np.zeros(3),
            "prev_pose": np.zeros(3),
            "target_diff": np.zeros(2),
            "collision": False,
            "result": None
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(9)
        # self.action_space = spaces.Box(np.array([0, 0, -0.5]), np.array([1, 1, 0.5]))   #throttle, brake, steering

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        self.elapsed_t = 0
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.throttle = 0
        self.car_controls.brake = 0
        self.car_controls.steering = 0

        if action == 1:                         # brake
            self.car_controls.brake = 1        
        elif action == 2:                       # accelerate
            self.car_controls.throttle = 1

        elif action == 3:                       # turn right w accel
            self.car_controls.throttle = 1
            self.car_controls.steering = 0.5
        elif action == 4:                       # turn left w accel
            self.car_controls.throttle = 1
            self.car_controls.steering = -0.5

        elif action == 5:                       # turn right w/o accel
            self.car_controls.steering = 0.5
        elif action == 6:                       # turn left w/o accel
            self.car_controls.steering = -0.5
        
        elif action == 7:                       # turn right w brake
            self.car_controls.brake = 1
            self.car_controls.steering = 0.5
        elif action == 8:                       # turn left w brake
            self.car_controls.brake = 1
            self.car_controls.steering = -0.5
        
        self.car.setCarControls(self.car_controls)
        self.elapsed_t += 1
        time.sleep(0.05)

    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([1, 84, 84])

    def _get_obs(self):
        # STATES
        self.car_state = self.car.getCarState()
        gps_data = self.car.getGpsData("Gps", "Car").gnss.geo_point

        # position of car
        self.state["prev_position"] = self.state["position"]
        # using Odometry
        self.state["position"] = self.car_state.kinematics_estimated.position.to_numpy_array()
        # using GPS
        # self.state["position"] = np.array([gps_data.altitude, gps_data.latitude, gps_data.longitude])

        # pose of car
        self.state["prev_pose"] = self.state["pose"]
        # quartenion to euler
        quat = self.car_state.kinematics_estimated.orientation.to_numpy_array()
        t0 = 2.0 * (quat[3] * quat[0] + quat[1] * quat[2])
        t1 = 1.0 - 2.0 * (quat[0]**2 + quat[1]**2)
        roll_x = math.atan2(t0, t1)
        t2 = 2.0 * (quat[3] * quat[1] - quat[2] * quat[0])
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = math.asin(t2)
        t3 = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
        t4 = 1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
        yaw_z = math.atan2(t3, t4)
        self.state["pose"] = np.array([roll_x, pitch_y, yaw_z])

        # target direction
        target_vec = self.target_pos - self.state["position"]
        target_dist = np.linalg.norm(target_vec)
        target_dir = math.atan2(target_vec[1], target_vec[0])
        angle = (target_dir - self.state["pose"][2])
        if angle > np.pi:
            angle -= (2*np.pi)
        elif angle < -np.pi:
            angle += (2*np.pi)
        self.state["target_diff"] = np.array([target_dist, angle])

        # collision info
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        # OBSERVATIONS
        # camera image
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        return image

    def _compute_reward(self):
        reward = -0.1
        done = False
        if self.state["target_diff"][0] < 1:
            reward += 10
            done = True
            self.state["result"] = "arrive"
        elif self.elapsed_t > 500:
            done = True
            self.state["result"] = "time out"
        elif self.state["collision"]:
            reward -= 10
            done = True
            self.state["result"] = "crash"

        angle = np.abs(self.state["target_diff"][1])/np.pi                  # 0.0 ~ 1.0
        dist = self.state["target_diff"][0]/np.linalg.norm(self.target_pos) # 0.0 ~ inf, initially 1.0

        reward -= (angle + dist)

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, self.state

    def reset(self, seed=None, return_info=False, options=None):
        self._setup_car()
        self._do_action(0)
        return self._get_obs()
