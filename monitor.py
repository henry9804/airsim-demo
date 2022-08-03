import keyboard
import time
import numpy as np
import math

import airsim

ip_address = "127.0.0.1"
car = airsim.CarClient(ip=ip_address)
car_controls = airsim.CarControls()
car_state = None

def reset():
    car.reset()
    car.enableApiControl(True)
    car.armDisarm(True)
    print("reset")
    time.sleep(0.01)

def control_by_keyboard():
    car_controls.throttle = 0
    car_controls.brake = 0
    car_controls.steering = 0
    car_controls.manual_gear = 0
    car_controls.is_manual_gear = False

    if keyboard.is_pressed('w'):      # go straight
        car_controls.throttle = 1
    elif keyboard.is_pressed('s'):    # go backward
        car_controls.throttle = -1
        car_controls.is_manual_gear = True
        car_controls.manual_gear = -1
    elif keyboard.is_pressed('x'):    # brake
        car_controls.brake = 1
    if keyboard.is_pressed('a'):    # turn left
        car_controls.steering = -0.5
    elif keyboard.is_pressed('d'):    # turn right
        car_controls.steering = 0.5
    if keyboard.is_pressed('q'):    # reset
        reset()
        return

    car.setCarControls(car_controls)

reset()
while(1):
    car_state = car.getCarState()
    gps_data = car.getGpsData("Gps", "Car").gnss.geo_point

    quat = car_state.kinematics_estimated.orientation.to_numpy_array()
    
    t0 = 2.0 * (quat[3] * quat[0] + quat[1] * quat[2])
    t1 = 1.0 - 2.0 * (quat[0]**2 + quat[1]**2)
    roll_x = math.atan2(t0, t1)
    t2 = 2.0 * (quat[3] * quat[1] - quat[2] * quat[0])
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = math.asin(t2)
    t3 = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
    t4 = 1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
    yaw_z = math.atan2(t3, t4)

    print("XYZ: {}".format(car_state.kinematics_estimated.position.to_numpy_array()))
    print("GPS: {}".format(np.array([gps_data.altitude, gps_data.latitude, gps_data.longitude])))
    print("RPY: [{} {} {}]".format(roll_x, pitch_y, yaw_z))
    print("------------------------------------------------")
    control_by_keyboard()
    time.sleep(1)