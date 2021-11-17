import gym
from gym import spaces
import numpy as np
import sys
try:
    sys.path.append(r'D:\下载\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg')
    #sys.path.append(r'd:\下载\CARLA_0.9.8\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg')
except IndexError:
    pass
import carla
import time
import random
# 导入这个异步或者同步模式的类
from carla_sync_mode import CarlaSyncMode
from carla import ColorConverter as cc

# 最好把所有的内容都写到一个文件中，建议这种写法比较清晰的可读性



class CarlaOvertakeEnv(gym.Env):

    def __init__(self,
                 carla_port,
                 frame_skip,
                 observations_type,
                 map_name):

        super(CarlaOvertakeEnv, self).__init__()

        self.carla_port = carla_port
        self.map_name = map_name
        self.observations_type = observations_type
        self.frame_skip = frame_skip

        # [spawn_ego_location, spawn_actor_location]
        self.spawn_loc =  [carla.Transform(carla.Location(-88.5, -70.0, 0.1), carla.Rotation(yaw=90)),
                            carla.Transform(carla.Location(-88.5, -63.0, 0.1), carla.Rotation(yaw=90))]
        # the destination of the egp location
        self.goal = [20]

        # initialize client with timeout,  maybe 10 or 20 seconds is needed
        self.client = carla.Client('localhost', self.carla_port)
        self.client.set_timeout(20.0)

        # initialize world and map
        self.world = self.client.get_world()
        self.world = self.client.load_world(self.map_name)
        self.map = self.world.get_map()
        print("Now we choose the map '{}'".format(self.map))

        # freeze the traffic light to avoid some error
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.freeze(True)

        self.actor_list = []

        # Set transform of spectator
        spectator = self.world.get_spectator()
        self.spectator = spectator

        self.reset()

        self.spectator.set_transform(carla.Transform(self.ego.get_transform().location + carla.Location(z=50),
                                                carla.Rotation(yaw=90, pitch=-90)))

        if self.observations_type == 'state':
            obs = self._get_state_obs()
        else:
            obs = np.zeros((3, 84, 84))

        # gym environment specific variables
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        self.obs_dim = obs.shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.obs_dim, dtype='float32')

    def collision_callback(self,event):
        self.collision = True



    def reset(self, debug=False):
        self._reset_all(debug=debug)
        self.world.tick()
        self.count = 0
        self.collision = False
        obs, _, _, _ = self.step([0, 0])
        return obs

    def _reset_all(self, debug=False):

        actor_list = self.world.get_actors()

        vehicle_num = 0
        sensor_num = 0

        for vehicle in actor_list.filter('*vehicle*'):
            vehicle_num += 1
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            sensor_num += 1
            sensor.destroy()
        if debug:
            print('Warning: removing {} old vehicle and {} old sensor'.format(vehicle_num, sensor_num))

        # create vehicle
        # self.ego = None
        # self.actor = None
        ego_transform = random.choice(self.world.get_map().get_spawn_points())
        actor_transform = random.choice(self.world.get_map().get_spawn_points())

        # spawn ego and actor blueprint
        self.ego_bp = self.world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        self.actor_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")

        self.ego = self.world.spawn_actor(self.ego_bp, ego_transform)
        self.actor = self.world.spawn_actor(self.actor_bp, actor_transform)

        # After spawning the ego and actor at placeable position, we need to move to the right position
        #print("Moving to the new location...")

        self.ego.set_transform(self.spawn_loc[0])
        self.actor.set_transform(self.spawn_loc[1])




        # Add the ego and actor to the actor list
        self.actor_list.append(self.ego)
        self.actor_list.append(self.actor)

        # whether to save images if we are recording
        self.recording = False

        # this is the case [1]
        self.cam_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.cam_bp.set_attribute("image_size_x", str(84))
        self.cam_bp.set_attribute("image_size_y", str(84))
        self.cam_bp.set_attribute("fov", str(105))
        self.cam_bp.set_attribute("sensor_tick", str(1 / 30))

        cam_transform = carla.Transform(carla.Location(6, 0, 10), carla.Rotation(-90, 0, 0))
        self.camera = self.world.spawn_actor(self.cam_bp, cam_transform, attach_to=self.ego)


        # collision detection
        self.collision = False

        self.col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(self.col_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(self.collision_callback)

        # Add the camera and collision sensor to the actor list
        self.actor_list.append(self.camera)
        self.actor_list.append(self.collision_sensor)

        ###############################################################################



        # set Synchronous mode to ensure to get the full state message
        if self.observations_type == 'pixel':
            self.sync_mode = CarlaSyncMode(self.world, self.camera, fps=20)
        elif self.observations_type == 'state':
            self.sync_mode = CarlaSyncMode(self.world, fps=20)
        else:
            raise ValueError('Unknown observation_type. Choose between: state, pixel')


    def _move_vehicle(self):
        # reset vehicle and move them to the specific location


        '''
        Points = self.world.get_map().get_spawn_points()
        for i in range(len(Points)):
            print(Points[i])
        try_spawn_actor? spawn_actor? what are the difference between them?
        '''

        print("Moving to the new location...")

        self.ego.set_transform(self.spawn_loc[0])
        self.actor.set_transform(self.spawn_loc[1])

        # Set transform of spectator
        spectator = self.world.get_spectator()

        transform = self.ego.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                carla.Rotation(yaw=90, pitch=-90)))



    def _compute_action(self):
        return self.agent.run_step()

    def step(self, action):
        rewards = []
        next_obs, done, info = np.array([]), False, {}
        for _ in range(self.frame_skip):
            # TODOLIST: We maybe change this to the ruled-based overtake methods

            next_obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)
            if done:
                break
        return next_obs, np.mean(rewards), done, info

    def _simulator_step(self, action):



        # calculate actions
        throttle_brake = float(action[0])
        steer = float(action[1])
        if throttle_brake >= 0.0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        # apply control to simulation
        vehicle_control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        actor_control = carla.VehicleControl(
            throttle=float(0.3),
            steer=float(0.0),
            brake=float(0.0),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.spectator.set_transform(carla.Transform(self.ego.get_transform().location + carla.Location(z=50),
                                                carla.Rotation(yaw=90, pitch=-90)))

        self.ego.apply_control(vehicle_control)
        self.actor.apply_control(actor_control)


        if self.observations_type == 'pixel':
            snapshot, vision_image = self.sync_mode.tick(timeout=5.0)
        elif self.observations_type == 'state':
            self.sync_mode.tick(timeout=5.0)


        # get reward and next observation
        reward, done, info = self._get_reward()
        if self.observations_type == 'state':
            next_obs = self._get_state_obs()
        else:
            next_obs = self._get_pixel_obs(vision_image)


        # increase frame counter
        self.count += 1

        return next_obs, reward, done, info


    def _get_pixel_obs(self, vision_image):
        # TODOLIST: add learningBycheating image

        vision_image.convert(cc.CityScapesPalette)
        bgra = np.array(vision_image.raw_data).reshape(84, 84, 4)
        bgr = bgra[:, :, :3]
        rgb = np.flip(bgr, axis=2)
        return rgb

    def _get_state_obs(self, deg2rad=False):
        transform = self.ego.get_transform()
        location = transform.location
        rotation = transform.rotation
        x_pos = location.x
        y_pos = location.y
        z_pos = location.z

        if deg2rad:
            # if True, convert the degree to radians
            # such as 90 degree means pi/2=3.14/2=1.57
            pitch = np.radians(rotation.pitch)
            yaw = np.radians(rotation.yaw)
            roll = np.radians(rotation.roll)
        else:
            pitch = rotation.pitch
            yaw = rotation.yaw
            roll = rotation.roll

        acceleration = vector_to_scalar(self.ego.get_acceleration())
        angular_velocity = vector_to_scalar(self.ego.get_angular_velocity())
        velocity = vector_to_scalar(self.ego.get_velocity())
        return np.array([x_pos,
                         y_pos,
                         z_pos,
                         pitch,
                         yaw,
                         roll,
                         acceleration,
                         angular_velocity,
                         velocity], dtype=np.float64)

    def _get_reward(self):
        # We need to get the current location of ego and actor
        ego_loc = self.ego.get_location()
        actor_loc = self.actor.get_location()

        '''
        There are some cases for reward shaping.
        [0] whether used the expert data for pretrained learning?
        [1] collision reward
        [2] destination reward
        [3] offline reward 
        [4]...
        '''

        follow_waypoint_reward = self._get_follow_waypoint_reward(ego_loc)
        timestep_reward = self._get_timestep_reward()

        done_col, collision_reward = self._get_collision_reward()

        done_off, offline_reward = self._get_offline(ego_loc)

        done_dest, dest_reward = self._get_destination(ego_loc)

        if done_col or done_off or done_dest:
            done = True
        else:
            done = False

        # There is 3 kinds of done here, we need to get the final done result here



        total_reward = collision_reward + \
                       offline_reward + \
                       follow_waypoint_reward + \
                       dest_reward + \
                       timestep_reward



        info_dict = dict()

        info_dict['wp'] = np.round(follow_waypoint_reward,2)
        info_dict['col'] = np.round(collision_reward,2)
        info_dict['dest'] = np.round(dest_reward,2)
        info_dict['off'] = np.round(offline_reward, 2)

        return total_reward, done, info_dict

    def _get_follow_waypoint_reward(self, ego_location):
        # sample from the right trajectory
        # maybe we use the MultiFuturePrediction, we tried to predict the future position?...
        # use `demo_carla.py` file, later we will add this function
        reward = 0
        return reward

    def _get_timestep_reward(self):
        reward = -0.1
        return reward

    def _get_collision_reward(self):
        if not self.collision:
            return False, 0
        else:
            return True, -1


    def _get_offline(self, ego_location, debug=False):
        # TODOLIST: which is the really needed location? x position or y position, I am not sure...
        # Sure. The original code adpoted y location to find
        x_pos = ego_location.x
        y_pos = ego_location.y
        x_pos = np.round(x_pos,2)
        y_pos = np.round(y_pos,2)

        if debug:
            # print the ego current location to find the right offline location
            print("Ego position: [{},{}]".format(x_pos,y_pos))

        if x_pos > -84.5 or x_pos < -89.5:
            return True, -1
        else:
            return False, 0

    def _get_destination(self, ego_destination_location):
        ego_dest = ego_destination_location
        x_pos = ego_dest.x
        y_pos = ego_dest.y
        x_pos = np.round(x_pos,2)
        y_pos = np.round(y_pos,2)
        if y_pos < self.goal:
            return False, 0
        else:
            return True, 1

    # maybe we donot use this function here
    def close(self):
        for actor in self.actor_list:
            if hasattr(actor, 'destroy()'):
                actor.destroy()
        print('\ndestroying %d sensors or vehicles' % len(self.actor_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        time.sleep(0.5)





#### some utils for other things


def vector_to_scalar(vector):
    scalar = np.around(np.sqrt(vector.x ** 2 +
                               vector.y ** 2 +
                               vector.z ** 2), 2)
    return scalar



