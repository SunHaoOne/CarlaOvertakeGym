import gym
import carla_env

if __name__ == '__main__':
    env = gym.make('CarlaEnv-pixel-v1')
    env.reset()
    done = False

    for i in range(100):
        env.reset()
        while not done:

            # max steps:50000
            next_obs, reward, done, info = env.step([0.5, -0.1])
            #print(next_obs)

        done = False
    env.close()
