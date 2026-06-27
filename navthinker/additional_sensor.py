#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
from gym import spaces

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.nav.nav import PointGoalSensor
from hydra.core.config_store import ConfigStore
import habitat_sim

from dataclasses import dataclass
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface


@dataclass
class OracleShortestPathSensorConfig(LabSensorConfig):
    
    type: str = "OracleShortestPathSensor"

@dataclass
class OracleFollowerSensorConfig(LabSensorConfig):
    
    type: str = "OracleFollowerSensor"


@dataclass
class HumanVelocitySensorConfig(LabSensorConfig):
    type: str = "HumanVelocitySensor"

@dataclass
class HumanNumSensorConfig(LabSensorConfig):
    type: str = "HumanNumSensor"
    max_num: int = 6

@dataclass
class RiskSensorConfig(LabSensorConfig):
    type: str = "RiskSensor"
    thres: float = 3.0
    use_geo_distance: bool = True

@dataclass
class SocialCompassSensorConfig(LabSensorConfig):
    type: str = "SocialCompassSensor"
    thres: float = 9.0
    num_bins: int = 8

@dataclass
class OracleHumanoidFutureTrajectorySensorConfig(LabSensorConfig):
    type: str = "OracleHumanoidFutureTrajectorySensor"
    future_step: int = 4

@registry.register_sensor(name="OracleShortestPathSensor")
class OracleShortestPathSensor(Sensor):
    r"""Sensor that used for A* and ORCA
    """
    cls_uuid: str = "oracle_shortest_path_sensor"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (2,3)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )
    
    def _path_to_point_2(self, point_a, point_b):
        """Get the shortest path between two points."""
        path = habitat_sim.ShortestPath()  # habitat_sim
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        return path.points[:2] if found_path else [point_a, point_b]
    
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = np.array(agent_state.position, dtype=np.float32)
        # rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # return [agent_position, goal_position]
        return self._path_to_point_2(
            agent_position, goal_position
        )

@registry.register_sensor(name="OracleFollowerSensor")
class OracleFollowerSensor(PointGoalSensor):
    r"""Sensor that used for A* and ORCA
    """
    cls_uuid: str = "oracle_follower_sensor"
        
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (2,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )
    
    def _path_to_point_1(self, point_a, point_b):
        """Get the shortest path between two points."""
        path = habitat_sim.ShortestPath()  # habitat_sim
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        return path.points[1] if found_path else [point_b]
    
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, self._path_to_point_1(agent_position,goal_position)
        )

@registry.register_sensor
class HumanVelocitySensor(UsesArticulatedAgentInterface, Sensor):
    """
    The position and angle of the articulated_agent in world coordinates.
    """

    cls_uuid = "human_velocity_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.value = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 6, dtype=np.float64)

    def _get_uuid(self, *args, **kwargs):
        return HumanVelocitySensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(6,6),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        # human_num = kwargs["task"]._human_num
        for i in range(self._sim.num_articulated_agents-1):
            articulated_agent = self._sim.get_agent_data(i+1).articulated_agent
            human_pos = np.array(articulated_agent.base_pos, dtype=np.float64)
            human_rot = np.array([float(articulated_agent.base_rot)], dtype=np.float64)
            human_vel = np.array(kwargs['task'].measurements.measures['human_velocity_measure']._metric[i],dtype=np.float64)
            self.value[i] = np.concatenate((human_pos, human_rot, human_vel))
        return self.value
    
@registry.register_sensor
class HumanNumSensor(UsesArticulatedAgentInterface, Sensor):
    """
    The num of the other agent in world.
    (in our setup, agents except agent_0 are humanoids)
    """

    cls_uuid = "human_num_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return HumanNumSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,), low=0, high=6, dtype=np.int32
        )

    def get_observation(self, observations, episode, *args, **kwargs):    
        if "human_num" in episode.info:
            human_num = min(episode.info['human_num'], 6)
        else:
            human_num = min(self._sim.num_articulated_agents - 1, 6)
        # Ensure the returned value is a tensor with shape (1,)
        return np.array([human_num], dtype=np.int32)

@registry.register_sensor
class RiskSensor(UsesArticulatedAgentInterface, Sensor):
    r"""Sensor for observing social risk to which the agent is subjected".

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "risk_sensor"

    def __init__(
        self, sim, config, *args, **kwargs
    ):
        self._sim = sim
        self._robot_idx = 0
        self.thres = config.thres
        self._use_geo_distance = config.use_geo_distance
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def get_observation(
        self, observations, episode, *args, **kwargs
    ):
        self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        if self._human_nums == 0:
            return np.array([0], dtype=np.float32)
        else:
            robot_pos = self._sim.get_agent_state(0).position

            human_pos = []
            human_dis = []

            for i in range(self._human_nums):
                human_position = self._sim.get_agent_state(i+1).position
                human_pos.append(human_position)

                if self._use_geo_distance:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = robot_pos
                    path.requested_end = human_position
                    found_path = self._sim.pathfinder.find_path(path)

                    if found_path:
                        distance = self._sim.geodesic_distance(robot_pos, human_position)
                    else:
                        distance = np.linalg.norm(human_position - robot_pos, ord=2)
                else:
                    distance = np.linalg.norm(human_position - robot_pos, ord=2)

                human_dis.append(distance)

            return np.array([max(1 - min(human_dis) / self.thres, 0)],
                            dtype=np.float32)

@registry.register_sensor
class SocialCompassSensor(UsesArticulatedAgentInterface, Sensor):
    r"""
    Implementation of people relative position sensor
    """

    cls_uuid: str = "social_compass_sensor"

    def __init__(
        self, sim, config, *args, **kwargs
    ):
        self._sim = sim
        # parameters
        self.thres = config.thres
        self.num_bins = config.num_bins
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0, high=np.inf, shape=(self.num_bins,),
                          dtype=np.float32)

    def get_polar_angle(self, agent_id = 0):
        agent_state = self._sim.get_agent_state(agent_id)
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip
    
    def get_heading_error(self, source, target):
        r"""Computes the difference between two headings (radians); can be negative
        or positive.
        """
        diff = target - source
        if diff > np.pi:
            diff -= np.pi*2
        elif diff < -np.pi:
            diff += np.pi*2
        return diff
    
    def get_observation(self, observations, episode, *args, **kwargs):
        self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        angles = [0] * self.num_bins
        if self._human_nums == 0:
            return np.array(angles, dtype=np.float32)
        else:
            a_pos = self._sim.get_agent_state(0).position
            a_head = self._sim.get_agent_state(0).rotation  # 2*np.arccos(self._sim.get_agent_state().rotation.w)

            a_head = -self.get_polar_angle(0) + np.pi / 2  # -quat_to_rad(a_head) + np.pi / 2

            for i in range(self._human_nums):
                pos = self._sim.get_agent_state(i+1).position
                theta = math.atan2(pos[2] - a_pos[2], pos[0] - a_pos[0])
                theta = self.get_heading_error(a_head, theta)
                theta = theta if theta > 0 else 2 * np.pi + theta

                bin = int(theta / (2 * np.pi / self.num_bins))

                dist = np.sqrt((pos[2] - a_pos[2]) ** 2 + (pos[0] - a_pos[
                    0]) ** 2)  # self._sim.geodesic_distance(a_pos, pos)
                norm_dist = max(1 - dist / self.thres, 0)
                if norm_dist > angles[bin]:
                    angles[bin] = norm_dist

            return np.array(angles, dtype=np.float32)

@registry.register_sensor
class OracleHumanoidFutureTrajectorySensor(UsesArticulatedAgentInterface, Sensor):
    """
    Assumed Oracle Humanoid Future Trajectory Sensor.
    """

    cls_uuid: str = "oracle_humanoid_future_trajectory"

    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        self.future_step = kwargs['config']['future_step'] 
        self.max_human_num = 6
        self.human_num = task._human_num
        self.result_list = None  

        super().__init__(*args, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OracleHumanoidFutureTrajectorySensor.cls_uuid

    @staticmethod
    def _get_sensor_type(*args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self.max_human_num, self.future_step, 2),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _initialize_result_list(human_num, future_step, max_human_num):
        """Initialize the result list with default values."""
        result = np.full((max_human_num, future_step, 2), -100, dtype=np.float32)
        return result

    def get_observation(self, task, *args, **kwargs):
        human_num = self._task._human_num

        if self.result_list is None or human_num != self.human_num:
            self.result_list = self._initialize_result_list(human_num, self.future_step, self.max_human_num)
            self.human_num = human_num

        self.result_list[:] = -100

        if self.human_num == 0:
            return self.result_list

        human_future_trajectory = task.measurements.measures.get("human_future_trajectory")._metric
        if not human_future_trajectory:
            return self.result_list

        robot_pos = np.array(self._sim.get_agent_data(0).articulated_agent.base_pos)[[0, 2]]

        for key, wp_list in human_future_trajectory.items():
            next_wp = np.array(wp_list[0], dtype=np.float32)
            next_wp_2d = next_wp[[0, 2]]

            human_pos = np.array(
                self._sim.get_agent_data(key).articulated_agent.base_pos
            )[[0, 2]]

            n_seg = self.future_step
            for s in range(n_seg):
                alpha = (s + 1) / n_seg
                interp = human_pos + alpha * (next_wp_2d - human_pos)
                self.result_list[key - 1, s, :] = interp - robot_pos

        return self.result_list.tolist()

cs = ConfigStore.instance()

cs.store(
    package="habitat.task.lab_sensors.oracle_shortest_path_sensor",
    group="habitat/task/lab_sensors",
    name="oracle_shortest_path_sensor",
    node=OracleShortestPathSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_follower_sensor",
    group="habitat/task/lab_sensors",
    name="oracle_follower_sensor",
    node=OracleFollowerSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.human_velocity_sensor",
    group="habitat/task/lab_sensors",
    name="human_velocity_sensor",
    node=HumanVelocitySensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.human_num_sensor",
    group="habitat/task/lab_sensors",
    name="human_num_sensor",
    node=HumanNumSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.risk_sensor",
    group="habitat/task/lab_sensors",
    name="risk_sensor",
    node=RiskSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.social_compass_sensor",
    group="habitat/task/lab_sensors",
    name="social_compass_sensor",
    node=SocialCompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_humanoid_future_trajectory",
    group="habitat/task/lab_sensors",
    name="oracle_humanoid_future_trajectory",
    node=OracleHumanoidFutureTrajectorySensorConfig,
)


# ============================================================================
# HumanStateGoalSensor - 融合传感器
# ============================================================================

@dataclass
class HumanStateGoalSensorConfig(LabSensorConfig):
    """Configuration for Human State and Goal Sensor"""
    type: str = "HumanStateGoalSensor"
    max_human_num: int = 6


@registry.register_sensor
class HumanStateGoalSensor(UsesArticulatedAgentInterface, Sensor):
    """
    融合传感器：提供人类的完整状态和目标信息
    
    Output shape: (max_human_num, 8)
    Format per human: [pos_x, pos_z, vel_x, vel_z, goal_x, goal_z, rotation, dist_to_goal]
    - pos_x, pos_z: 当前位置 (相对机器人)
    - vel_x, vel_z: 当前速度 (2D速度向量)
    - goal_x, goal_z: 目标位置 (相对机器人)
    - rotation: 朝向角度 (弧度)
    - dist_to_goal: 到目标的距离
    
    Invalid/missing humans are filled with -100
    
    Advantages:
    - 单个sensor提供完整信息，减少sensor数量
    - 整合位置、速度、目标，适合端到端预测
    - 包含距离信息，便于判断目标达成
    - 所有坐标统一相对机器人，便于处理
    """
    
    cls_uuid: str = "human_state_goal"
    
    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        config = kwargs.get('config')
        self.max_human_num = config.max_human_num if config and hasattr(config, 'max_human_num') else 6
        super().__init__(*args, task=task, **kwargs)
    
    def _get_uuid(self, *args, **kwargs):
        return HumanStateGoalSensor.cls_uuid
    
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR
    
    def _get_observation_space(self, *args, config=None, **kwargs):
        return spaces.Box(
            shape=(self.max_human_num, 8),
            low=-100,
            high=100,
            dtype=np.float32,
        )
    
    def get_observation(self, task, *args, **kwargs):
        """
        获取所有人类的状态和目标信息
        
        Returns:
            np.array of shape (max_human_num, 8)
            Each row: [pos_x, pos_z, vel_x, vel_z, goal_x, goal_z, rotation, dist_to_goal]
        """
        result = np.full((self.max_human_num, 8), -100, dtype=np.float32)
        
        # 获取机器人位置 (agent 0)
        robot_data = self._sim.get_agent_data(0)
        robot_pos = np.array(robot_data.articulated_agent.base_pos)[[0, 2]]
        
        # 获取每个人类agent的信息 (从agent 1开始)
        num_agents = len(self._sim.agents_mgr)
        
        for i in range(1, min(self.max_human_num + 1, num_agents)):
            try:
                # 获取articulated agent
                agent_data = self._sim.get_agent_data(i)
                human = agent_data.articulated_agent
                
                # 1. 当前位置 (相对机器人)
                human_pos = np.array(human.base_pos)[[0, 2]]
                pos_relative = human_pos - robot_pos
                
                # 2. 当前速度
                vel = self._get_human_velocity(i, task)
                if vel is None:
                    vel = np.array([0.0, 0.0])
                
                # 3. 朝向角度
                rotation = float(human.base_rot)
                
                # 4. 目标位置 = 下一个路径点 (相对机器人)
                next_wp = self._get_next_waypoint(i, task)
                if next_wp is not None:
                    next_wp_2d = np.array(next_wp)[[0, 2]]
                    goal_relative = next_wp_2d - robot_pos
                    dist_to_goal = np.linalg.norm(next_wp_2d - human_pos)
                else:
                    goal_pos = self._get_human_goal_position(i)
                    if goal_pos is not None:
                        goal_pos_2d = np.array(goal_pos)[[0, 2]]
                        goal_relative = goal_pos_2d - robot_pos
                        dist_to_goal = np.linalg.norm(goal_pos_2d - human_pos)
                    else:
                        goal_relative = np.array([0.0, 0.0])
                        dist_to_goal = 0.0
                
                # 组装特征向量
                result[i-1] = [
                    pos_relative[0],      # pos_x
                    pos_relative[1],      # pos_z
                    vel[0],               # vel_x
                    vel[1],               # vel_z
                    goal_relative[0],     # goal_x
                    goal_relative[1],     # goal_z
                    rotation,             # rotation
                    dist_to_goal          # dist_to_goal
                ]
                
            except (IndexError, AttributeError, KeyError):
                continue
        
        return result
    
    def _get_next_waypoint(self, agent_idx, task):
        """从 human_future_trajectory measurement 获取下一个路径点 (3D)。"""
        try:
            if hasattr(task, 'measurements') and hasattr(task.measurements, 'measures'):
                measure = task.measurements.measures.get('human_future_trajectory')
                if measure is not None:
                    metric = measure._metric
                    if agent_idx in metric:
                        wp_list = metric[agent_idx]
                        if wp_list and len(wp_list) > 0:
                            return np.array(wp_list[0], dtype=np.float32)
        except Exception:
            pass
        return None

    def _get_human_velocity(self, agent_idx, task):
        """获取人类的速度信息"""
        try:
            if hasattr(task, 'measurements') and hasattr(task.measurements, 'measures'):
                if 'human_velocity_measure' in task.measurements.measures:
                    vel_data = task.measurements.measures['human_velocity_measure']._metric
                    if agent_idx - 1 < len(vel_data):
                        vel = vel_data[agent_idx - 1]
                        if isinstance(vel, (list, np.ndarray)) and len(vel) >= 2:
                            return np.array(vel[:2], dtype=np.float32)
        except Exception:
            pass
        return None
    
    def _get_human_goal_position(self, agent_idx):
        """获取人类的目标位置（优先使用实时goals from action）"""
        # 方法1: 从action暴露的实时goals获取（最优先）
        if hasattr(self._task, '_human_goals'):
            if agent_idx in self._task._human_goals:
                goals = self._task._human_goals[agent_idx]
                if goals is not None and len(goals) > 0:
                    return np.array(goals[0])
        
        # 方法2: 从task缓存获取
        if hasattr(self._task, '_human_goal_positions'):
            if agent_idx in self._task._human_goal_positions:
                return self._task._human_goal_positions[agent_idx]
        
        # 方法3: 从episode获取
        if hasattr(self._task, '_episode'):
            episode = self._task._episode
            for attr_name in [
                f'agent_{agent_idx}_goal',
                f'agent{agent_idx}_goal_position',
                'agent_goal_positions',
            ]:
                if hasattr(episode, attr_name):
                    attr_val = getattr(episode, attr_name)
                    if isinstance(attr_val, list) and agent_idx < len(attr_val):
                        return np.array(attr_val[agent_idx])
                    elif isinstance(attr_val, np.ndarray):
                        return attr_val
        
        # 方法4: 从simulator的agent配置获取
        try:
            if hasattr(self._sim, 'agents_mgr'):
                agent_name = self._sim.agents_mgr.agent_names[agent_idx]
                agent_config = self._sim.habitat_config.agents[agent_name]
                if hasattr(agent_config, 'goal_position'):
                    return np.array(agent_config.goal_position)
        except:
            pass
        
        return None


cs.store(
    package="habitat.task.lab_sensors.human_state_goal_sensor",
    group="habitat/task/lab_sensors",
    name="human_state_goal_sensor",
    node=HumanStateGoalSensorConfig,
)