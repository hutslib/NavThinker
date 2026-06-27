# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.utils.common import get_num_actions
from habitat.tasks.utils import cartesian_to_polar

@baseline_registry.register_policy
class ORCAPolicy(nn.Module, Policy):
    """
    :only apply one action, on my case, that is oracle random walk for humanoids.
    """

    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        orig_action_space: ActionSpace,
        num_envs: int,
        aux_loss_config,
        agent_name: Optional[str],
    ):
        Policy.__init__(self, action_space)
        nn.Module.__init__(self)
        self._num_envs: int = num_envs
        self._recurrent_hidden_size = (
            full_config.habitat_baselines.rl.ppo.hidden_size
        )
        self._device = None
        self.action_distribution_type = "categorical"
        self.turn_threshold = np.deg2rad(15.0)
        self.repeated_turn_count = [0] *  self._num_envs
        self.angle_diff = [0] *  self._num_envs
        self.max_repeated_turns = 3
        self.angle_threshold_for_forced_forward = np.deg2rad(5.0)
        self.safe_distance_threshold = 2.0

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        agent_name=None,
        **kwargs,
    ):
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]
        return cls(
            config=config.habitat_baselines.rl.policy[agent_name],
            full_config=config,
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=orig_action_space,
            num_envs=config.habitat_baselines.num_environments,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            agent_name=agent_name,
        )
    
    def to(self, device):
        self._device = device
        return super().to(device)
    
    @property
    def hidden_state_shape(self):
        return (
            self.num_recurrent_layers,
            self.recurrent_hidden_size,
        )

    @property
    def hidden_state_shape_lens(self):
        return [self.recurrent_hidden_size]

    @property
    def recurrent_hidden_size(self) -> int:
        return self._recurrent_hidden_size

    @property
    def num_recurrent_layers(self):
        return 2 # (0)

    @property
    def should_load_agent_state(self):
        return False
    
    @property
    def policy_action_space(self):
        """
        Fetches the policy action space for learning. If we are learning the HL
        policy, it will return its custom action space for learning.
        """
        return super().policy_action_space
    
    @property
    def policy_action_space_shape_lens(self):
        return [self._action_space]
    
    def parameters(self):
        return iter([nn.Parameter(torch.zeros((1,), device=self._device))])
    
    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # We assign a value of 0. This is needed so that we can concatenate values in multiagent
        # policies
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )

    def compute_orca_velocity(self, current_position, current_velocity, other_agents_pos, other_agents_rot, other_agents_vel, max_speed=0.25, time_horizon=4.0):
        """
        Calculates velocity using the ORCA algorithm to avoid other agents.

        :param current_position: current agent's position (x, z)
        :param current_velocity: current agent's velocity (vx, vz)
        :param other_agents_pos: list of other agents' positions (x, z)
        :param other_agents_rot: list of other agents' rotations (in radians)
        :param other_agents_vel: list of other agents' velocities (vx, vz)
        :param max_speed: maximum speed
        :param time_horizon: prediction time horizon
        :return: new speed (vx, vz)
        """
        new_velocity = current_velocity.clone()
        combined_avoidance_velocity = torch.zeros_like(current_velocity)

        for i in range(len(other_agents_rot)):
            
            rotation_radians = other_agents_rot[i] 
            direction_vector = torch.tensor([torch.sin(rotation_radians), torch.cos(rotation_radians)], device=current_position.device)  # 方向向量(-z方向)
            
            relative_velocity_other = other_agents_vel[i][0] * direction_vector

            relative_position = (other_agents_pos[i] - current_position)[[0, 2]]
            relative_velocity = current_velocity - relative_velocity_other

            distance = torch.norm(relative_position)
            combined_radius = 0.6  
            relative_position_normalized = relative_position / distance

            if distance > combined_radius:
                avoidance_velocity = relative_velocity + relative_position_normalized * (combined_radius - distance) / time_horizon
                combined_avoidance_velocity += avoidance_velocity

        adjusted_velocity = new_velocity + combined_avoidance_velocity / len(other_agents_rot)  

        if torch.norm(adjusted_velocity) > max_speed:
            adjusted_velocity = adjusted_velocity / torch.norm(adjusted_velocity) * max_speed

        return adjusted_velocity


    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs,
    ):
        
        batch_size = masks.shape[0]
        
        actions = torch.zeros(
            size=prev_actions.shape,
            device=masks.device,
            dtype=torch.int64,
        )
        for i in range(batch_size):
            orca_angle = 0.0
            target_angle = 0.0
            human_num = 0
            if observations['pointgoal_with_gps_compass'][i, 0] <= 0.2:
                actions[i] = 0  
            else:
                current_point = observations['oracle_shortest_path_sensor'][i, 0, :]
                current_rot = observations['localization_sensor'][i, -1]
                next_point = observations['oracle_shortest_path_sensor'][i, 1, :]
                min_distance = float('inf') 
                human_pos = []
                human_rot = []
                human_vel = []
                
                for j in range(6): # for max human num
                    if observations['human_velocity_sensor'][i][j][0] < -90:
                        break
                    else:
                        human_pos.append(observations['human_velocity_sensor'][i][j][:3])
                        human_rot.append(observations['human_velocity_sensor'][i][j][3])
                        human_vel.append(observations['human_velocity_sensor'][i][j][-2:])
                        distance = torch.norm((current_point - human_pos[-1])[[0, 2]])
                        min_distance = min(min_distance, distance)
                        human_num += 1

                delta_x = next_point[0] - current_point[0]
                delta_y = next_point[2] - current_point[2]
                
                target_angle = torch.atan2(-delta_y,delta_x)
                    
                if human_num != 0 and min_distance < self.safe_distance_threshold:
                    if prev_actions[i] == 1:
                        direction_vector = torch.tensor([torch.sin(current_rot), torch.cos(current_rot)], device=current_rot.device)
                        current_vel = 0.25 * direction_vector
                    else:
                        current_vel = torch.tensor([0.0, 0.0], device=current_rot.device)
                    orca_velocity = self.compute_orca_velocity(current_point, current_vel, human_pos, human_rot, human_vel)
                    if torch.norm(orca_velocity) < 0.1:
                        actions[i] = 0  # stop
                    else:
                        orca_angle = torch.atan2(-orca_velocity[1], orca_velocity[0]) 
                        weight = 0.8  
                        target_angle = orca_angle * weight + (1 - weight) * target_angle

                self.angle_diff[i]  = target_angle - current_rot
                if abs(self.angle_diff[i]) < self.turn_threshold or abs(self.angle_diff[i]) > 2 * torch.pi - self.turn_threshold:
                    actions[i] = 1  
                    self.repeated_turn_count[i] = 0  
                elif (self.angle_diff[i] > -2 * torch.pi + self.turn_threshold and self.angle_diff[i] < - torch.pi) or (self.angle_diff[i] > self.turn_threshold and self.angle_diff[i] < torch.pi):
                    if prev_actions[i] == 3:
                        self.repeated_turn_count[i] += 1
                    else:
                        self.repeated_turn_count[i] = 0
                    actions[i] = 2  # 
                else:  
                    if prev_actions[i] == 2:
                        self.repeated_turn_count[i] += 1
                    else:
                        self.repeated_turn_count[i] = 0
                    actions[i] = 3  # TURN_RIGHT 

                if self.repeated_turn_count[i] >= self.max_repeated_turns:
                    actions[i] = 1  
                    self.repeated_turn_count[i] = 0  

        # This will update the prev action
        use_action = actions

        return PolicyActionData(
            take_actions=actions,
            actions=use_action,
            rnn_hidden_states=rnn_hidden_states,
        )
