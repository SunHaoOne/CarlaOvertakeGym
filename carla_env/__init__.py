from gym.envs.registration import register

# max episode need to be larger
register(
    id='CarlaEnv-state-v1',
    entry_point='carla_env.carla_env:CarlaOvertakeEnv',
    max_episode_steps=50000,
    kwargs={
        'carla_port': 2000,
        'frame_skip': 1,
        'observations_type': 'state',
        'map_name': 'Town03',
    }
)

register(
    id='CarlaEnv-pixel-v1',
    entry_point='carla_env.carla_env:CarlaOvertakeEnv',
    max_episode_steps=50000,
    kwargs={
        'carla_port': 2000,
        'frame_skip': 8,
        'observations_type': 'pixel',
        'map_name': 'Town03',
    }
)