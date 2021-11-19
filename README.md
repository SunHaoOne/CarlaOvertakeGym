# CarlaOvertakeGym
A simple implementation of carla overtake scenario.

Forked from https://github.com/janwithb/carla-gym-wrapper 


### Observation Space
Pixel: (3, 84, 84)

| Index         | Value             |
| ------------- |:-----------------:|
| 0             | r channel 84 x 84 |
| 1             | g channel 84 x 84 |
| 2             | b channel 84 x 84 |

State: (9, )

| Index         | Value             |
| ------------- |:-----------------:|
| 0             | x_pos             |
| 1             | y_pos             |
| 2             | z_pos             |
| 3             | pitch             |
| 4             | yaw               |
| 5             | roll              |
| 6             | acceleration      |
| 7             | angular_velocity  |
| 8             | velocity          |

### Action Space
Action: (2, )

| Index         | Value             | Min               | Max               |
| ------------- |:-----------------:|:-----------------:|:-----------------:|
| 0             | throttle_brake    | -1                | 1                 |
| 1             | steer             | -1                | 1                 |

### Reward Shaping

There are 5 kinds of reward in this carla overtake environment, including:
- follow waypoint reward
- timestep reward
- collision reward
- offline reward
- destination reward.

```
follow_waypoint_reward = self._get_follow_waypoint_reward(ego_loc)
timestep_reward = self._get_timestep_reward()

done_col, collision_reward = self._get_collision_reward()
done_off, offline_reward = self._get_offline(ego_loc)
done_dest, dest_reward = self._get_destination(ego_loc)
```
