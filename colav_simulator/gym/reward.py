"""reward.py.

Summary:
This file contains some simple rewarder classes for giving reward signals to the agent in the colav-simulator environment.

Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

import colav_simulator.gym.action as csgym_action
import colav_simulator.gym.observation as csgym_obs

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


@dataclass
class ExistenceRewardParams:
    """Parameters for the Existence rewarder."""

    r_exists: float = 1.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExistenceRewardParams":  # noqa: D102
        cfg = ExistenceRewardParams()
        cfg.r_exists = config_dict["r_exists"]
        return cfg

    def to_dict(self) -> dict:  # noqa: D102
        return {"r_exists": self.r_exists}


@dataclass
class CollisionRewardParams:
    """Parameters for the Collision rewarder."""

    r_collision: float = 500.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CollisionRewardParams":  # noqa: D102
        cfg = CollisionRewardParams()
        cfg.r_collision = config_dict["r_collision"]
        return cfg

    def to_dict(self) -> dict:  # noqa: D102
        return {"r_collision": self.r_collision}


@dataclass
class GroundingRewardParams:
    """Parameters for the Grounding rewarder."""

    r_grounding: float = 500.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GroundingRewardParams":  # noqa: D102
        cfg = GroundingRewardParams()
        cfg.r_grounding = config_dict["r_grounding"]
        return cfg

    def to_dict(self) -> dict:  # noqa: D102
        return {"r_grounding": self.r_grounding}


@dataclass
class DistanceToGoalRewardParams:
    """Parameters for the distance to goal rewarder."""

    r_d2g: float = 1.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DistanceToGoalRewardParams":  # noqa: D102
        cfg = DistanceToGoalRewardParams()
        cfg.r_d2g = config_dict["r_d2g"]
        return cfg

    def to_dict(self) -> dict:  # noqa: D102
        return {"r_d2g": self.r_d2g}


@dataclass
class TrajectoryTrackingRewardParams:
    """Parameters for the trajectory tracking rewarder."""

    gamma_r: float = 1.0  # Reward constant
    gamma_e: float = 0.5  # Cross-track error coefficient
    gamma_u: float = 0.8  # Speed error coefficient
    gamma_chi: float = 1.5  # Course angle error coefficient

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrajectoryTrackingRewardParams":  # noqa: D102
        config = TrajectoryTrackingRewardParams()
        config.gamma_r = config_dict["gamma_r"]
        config.gamma_e = config_dict["gamma_e"]
        config.gamma_u = config_dict["gamma_u"]
        config.gamma_chi = config_dict["gamma_chi"]
        return config


@dataclass
class StaticColavRewardParams:
    """Parameters for the static COLAV rewarder."""

    alpha_x = 75
    gamma_x = 0.1
    gamma_theta = 10

    @classmethod
    def from_dict(cls, config_dict: dict) -> "StaticColavRewardParams":  # noqa: D102
        config = StaticColavRewardParams()

        config.alpha_x = config_dict["alpha_x"]
        config.gamma_x = config_dict["gamma_x"]
        config.gamma_theta = config_dict["gamma_theta"]

        return config


@dataclass
class DynamicColavRewardParams:
    """Parameters for the dynamic COLAV rewarder."""

    # Raw COLAV penalty scaling
    alpha_x: float = 75.0
    # Sensor angle coefficient
    gamma_theta: float = 1.0

    # Velocity multipliers
    gamma_v_stb: float = 0.09
    gamma_v_port: float = 0.01
    gamma_v_stern: float = 0.02

    # Distance multipliers
    gamma_x_stb: float = 7e-3
    gamma_x_port: float = 9e-3
    gamma_x_stern: float = 1e-2

    # Trade-off coefficient parameters.
    # The lists are given as [-, +] for negative and positive TS velocities.
    gamma_lambda: list = field(default_factory=lambda: [3e-5, 3e-3])
    alpha_lambda: list = field(default_factory=lambda: [2, 4])

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DynamicColavRewardParams":  # noqa: D102
        config = DynamicColavRewardParams()

        config.alpha_x = config_dict["alpha_x"]
        config.gamma_theta = config_dict["gamma_theta"]
        config.gamma_v_stb = config_dict["gamma_v_stb"]
        config.gamma_v_port = config_dict["gamma_v_port"]
        config.gamma_v_stern = config_dict["gamma_v_stern"]
        config.gamma_x_stb = config_dict["gamma_x_stb"]
        config.gamma_x_port = config_dict["gamma_x_port"]
        config.gamma_x_stern = config_dict["gamma_x_stern"]
        config.gamma_lambda = config_dict["gamma_lambda"]
        config.alpha_lambda = config_dict["alpha_lambda"]

        return config


@dataclass
class AutopilotReferenceRewardParams:
    """Parameters for the autopilot reference rewarder."""

    course_coefficient: float = 1.0
    speed_coefficient: float = 1.0

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AutopilotReferenceRewardParams":  # noqa: D102
        cfg = AutopilotReferenceRewardParams()

        cfg.course_coefficient = config_dict["course_coefficient"]
        cfg.speed_coefficient = config_dict["speed_coefficient"]

        return cfg


class IReward(ABC):
    """Interface/base class for reward functions. The rewarder parameters should be public."""

    def __init__(self, env: "COLAVEnvironment") -> None:
        self.env = env

    @abstractmethod
    def __call__(self, state: csgym_obs.Observation, action: csgym_action.Action | None = None, **kwargs) -> float:
        """Get the reward for the current state-action pair. Additional arguments can be passed if necessary."""

    @abstractmethod
    def get_last_rewards_as_dict(self) -> dict:
        """Get the last rewards as a dictionary with the individual reward components and their values."""


@dataclass
class Config:
    """Configuration class of parameters for the rewarder."""

    rewarders: list = field(default_factory=lambda: [ExistenceRewardParams()])

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":  # noqa: D102
        cfg = Config()
        rewarders = config_dict["rewarders"]
        for rewarder in rewarders:
            if "existence_rewarder" in rewarder:
                cfg.rewarders.append(ExistenceRewardParams.from_dict(rewarder))
            elif "distance_to_goal_rewarder" in rewarder:
                cfg.rewarders.append(DistanceToGoalRewardParams.from_dict(rewarder))
            elif "collision_rewarder" in rewarder:
                cfg.rewarders.append(CollisionRewardParams.from_dict(rewarder))
            elif "grounding_rewarder" in rewarder:
                cfg.rewarders.append(GroundingRewardParams.from_dict(rewarder))
            elif "trajectory_tracking_rewarder" in rewarder:
                cfg.rewarders.append(TrajectoryTrackingRewardParams.from_dict(rewarder))
            elif "static_colav_rewarder" in rewarder:
                cfg.rewarders.append(StaticColavRewardParams.from_dict(rewarder))
            elif "dynamic_colav_rewarder" in rewarder:
                cfg.rewarders.append(DynamicColavRewardParams.from_dict(rewarder))
            elif "autopilot_reference_rewarder" in rewarder:
                cfg.rewarders.append(AutopilotReferenceRewardParams.from_dict(rewarder))
        return cfg

    def to_dict(self) -> dict:  # noqa: D102
        rewarders = []
        for rewarder in self.rewarders:
            if isinstance(rewarder, ExistenceRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, DistanceToGoalRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, CollisionRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, GroundingRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, TrajectoryTrackingRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, StaticColavRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, DynamicColavRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, AutopilotReferenceRewardParams):
                rewarders.append(rewarder.to_dict())
        return {"rewarders": rewarders}


class ExistenceRewarder(IReward):
    """Reward the agent negatively for existing."""

    def __init__(self, env: "COLAVEnvironment", params: ExistenceRewardParams | None = None) -> None:
        """Initializes the reward function.

        Args:
            env: The COLAV environment.
            params: The reward parameters.
        """
        super().__init__(env)
        self.last_reward = 0.0
        self.params = params if params else ExistenceRewardParams()

    def __call__(self, state: csgym_obs.Observation, action: csgym_action.Action | None = None, **_kwargs) -> float:  # noqa: ARG002, ARG003
        self.last_reward = -self.params.r_exists
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:  # noqa: D102
        return {"r_exists": self.last_reward}


class DistanceToGoalRewarder(IReward):
    """Reward the agent for getting closer to the goal."""

    def __init__(self, env: "COLAVEnvironment", params: DistanceToGoalRewardParams | None = None) -> None:
        super().__init__(env)
        self.last_reward = 0.0
        self.goal = np.array([0.0, 0.0])
        self.params = params if params else DistanceToGoalRewardParams()

    def __call__(self, state: csgym_obs.Observation, action: csgym_action.Action | None = None, **kwargs) -> float:  # noqa: ARG002, ARG003
        if self.env.ownship.goal_csog_state.size > 0:
            self.goal = self.env.ownship.goal_csog_state[:2]
        elif self.env.ownship.waypoints.size > 1:
            self.goal = self.env.ownship.waypoints[:, -1]
        else:
            raise ValueError("No goal state or waypoints found")
        self.last_reward = -self.params.r_d2g * float(np.linalg.norm(self.env.ownship.csog_state[:2] - self.goal))
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:  # noqa: D102
        return {"r_d2g": self.last_reward}


class CollisionRewarder(IReward):
    """Reward the agent negatively for colliding."""

    def __init__(self, env: "COLAVEnvironment", params: CollisionRewardParams | None = None) -> None:
        """Initializes the reward function.

        Args:
            env (COLAVEnvironment): The environment
            params (CollisionRewardParams): The reward parameters.
        """
        super().__init__(env)
        self.last_reward = 0.0
        self.params = params if params else CollisionRewardParams()

    def __call__(self, state: csgym_obs.Observation, action: csgym_action.Action | None = None, **kwargs) -> float:  # noqa: ARG002, ARG003
        """Reward for the current state-action pair. Calls function from simulator class to get collision or not."""
        collision = self.env.simulator.determine_ship_collision()
        if collision:
            self.last_reward = -self.params.r_collision
        else:
            self.last_reward = 0.0
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:  # noqa: D102
        return {"r_collision": self.last_reward}


class GroundingRewarder(IReward):
    """Reward the agent negatively for grounding."""

    def __init__(self, env: "COLAVEnvironment", params: GroundingRewardParams | None = None) -> None:
        super().__init__(env)
        self.last_reward = 0.0
        self.params = params if params else GroundingRewardParams()

    def __call__(self, state: csgym_obs.Observation, action: csgym_action.Action | None = None, **kwargs) -> float:  # noqa: ARG002, ARG003
        """Reward for the current state-action pair. Calls function from simulator class to get grounding or not."""
        grounding = self.env.simulator.determine_ship_grounding()
        if grounding:
            self.last_reward = -self.params.r_grounding
        else:
            self.last_reward = 0.0
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:  # noqa: D102
        return {"r_grounding": self.last_reward}


class TrajectoryTrackingRewarder(IReward):
    """Trajectory tracking rewarder based on Meyer et.al.(2020).

    Extended to include speed error term.
    NOTE: Both the DynamicColavRewarder and NavigationPathObservation are required
    for this rewarder to function.
    """

    def __init__(self, env: "COLAVEnvironment", params: TrajectoryTrackingRewardParams | None = None) -> None:
        super().__init__(env)

        # Find the NavigationPathObservation object
        if isinstance(env.observation_type, csgym_obs.DictObservation):
            for observation_type in env.observation_type.observation_types:
                if isinstance(observation_type, csgym_obs.NavigationPathObservation):
                    self.path_observation = observation_type
                    break
        elif isinstance(env.observation_type, csgym_obs.NavigationPathObservation):
            self.path_observation = env.observation_type

        self.params = params
        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, _action: csgym_action.Action | None = None, **_kwargs) -> float:
        """Calculate trajectory tracking reward."""
        if self.path_observation is None:
            msg = "NavigationPathObservation is not part of the observation!"
            raise ValueError(msg)

        # If no Dynamic COLAV rewarder, the tradeoff coefficient defaults to 1.0
        tradeoff_coefficient = 1.0
        # Find the DynamicColavRewarder object
        for rewarder in self.env.rewarder.rewarders:
            if isinstance(rewarder, DynamicColavRewarder):
                if rewarder.last_reward_calculation_time != self.env.simulator.t:
                    msg = "Dynamic rewarder needs to be called before path rewarder!"
                    raise ValueError(msg)
                # Get tradeoff coefficient lambda from dynamic rewarder
                tradeoff_coefficient = rewarder.tradeoff_coefficient
                break

        # Unnormalize navigation observation
        navigation_observation = self.path_observation.unnormalize(state["NavigationPathObservation"])

        # Unpack observation
        cross_track_error = navigation_observation[3]
        course_angle_error = navigation_observation[4]
        speed_error = navigation_observation[6]

        # Calculate reward terms
        course_angle_term = self.params.gamma_chi * np.cos(course_angle_error) + self.params.gamma_r
        cte_term = np.exp(-self.params.gamma_e * abs(cross_track_error)) + self.params.gamma_r
        path_reward = course_angle_term * cte_term - self.params.gamma_r**2
        speed_reward = -self.params.gamma_u * (speed_error**2)

        self.last_reward = tradeoff_coefficient * (path_reward + speed_reward)
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:  # noqa: D102
        return {"r_trajectory_tracking": self.last_reward}


class StaticColavRewarder(IReward):
    """Static obstacle COLAV rewarder as implemented in Meyer et.al.(2020).

    NOTE: The LidarLikeObservation is required as part of the environment observation.
    """

    def __init__(self, env: "COLAVEnvironment", params: StaticColavRewardParams | None = None) -> None:
        super().__init__(env)
        # Extract the LidarLikeObservation object
        if isinstance(env.observation_type, csgym_obs.DictObservation):
            for observation_type in env.observation_type.observation_types:
                if isinstance(observation_type, csgym_obs.LidarLikeObservation):
                    self.sensor_suite = observation_type
                    self.sensor_angles = deepcopy(observation_type._sensor_angles)
                    break
        elif isinstance(env.observation_type, csgym_obs.LidarLikeObservation):
            self.sensor_suite = env.observation_type
            self.sensor_angles = deepcopy(env.observation_type._sensor_angles)

        self.params = params

        # Functions for reward calculation
        self.closeness_penalty = lambda x: self.params.alpha_x * np.exp(-self.params.gamma_x * x)
        self.weighting = lambda theta: 1 / (1 + self.params.gamma_theta * abs(theta))

        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: csgym_action.Action | None = None, **kwargs) -> float:  # noqa: ARG002, ARG003
        """Reward for the current state-action pair. Calls function from simulator class to get static COLAV reward."""
        if self.sensor_suite is None:
            msg = "LidarLikeObservation is not part of the observation!"
            raise ValueError(msg)
        # Extract current measured distances and sensor angles from LidarLikeObservation object
        dist_measurements = deepcopy(self.sensor_suite.current_dist_measurements)

        num = 0
        den = 0

        for distance, angle in zip(dist_measurements, self.sensor_angles, strict=False):
            num += self.weighting(angle) * self.closeness_penalty(distance)
            den += self.weighting(angle)

        self.last_reward = -num / den
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:  # noqa: D102
        return {"r_static_colav": self.last_reward}


class DynamicColavRewarder(IReward):
    """Rewards the agent for avoiding dynamic obstacles based on Meyer et al. (2020).

    Partial COLREGs compliance is also implemented.
    NOTE: The raw penalty calculation is altered, and does not take into account
    negative TS velocities. The gamma_v parameters are consequently simplified.
    """

    def __init__(self, env: "COLAVEnvironment", params: DynamicColavRewardParams | None = None) -> None:
        super().__init__(env)
        if isinstance(env.observation_type, csgym_obs.DictObservation):
            for observation_type in env.observation_type.observation_types:
                if isinstance(observation_type, csgym_obs.LidarLikeObservation):
                    self.sensor_suite = observation_type
                    self.sensor_angles = deepcopy(observation_type._sensor_angles)
                    break
        elif isinstance(env.observation_type, csgym_obs.LidarLikeObservation):
            self.sensor_suite = env.observation_type
            self.sensor_angles = deepcopy(env.observation_type._sensor_angles)

        self.params = params
        self.tradeoff_coefficient = 1.0
        self.last_reward_calculation_time = self.env.simulator.t

        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: csgym_action.Action | None = None, **kwargs) -> float:  # noqa: ARG002, ARG003
        if self.sensor_suite is None:
            msg = "LidarLikeObservation is not part of the observation!"
            raise ValueError(msg)
        # Extract current measured distances and velocities from LidarLikeObservation object
        dist_measurements = deepcopy(self.sensor_suite.current_dist_measurements)
        obstacle_velocities = deepcopy(self.sensor_suite.current_obstacle_velocities)

        self.last_reward_calculation_time = self.env.simulator.t
        num = 0
        den = 0
        lambdas = []

        for x, v, sensor_angle in zip(dist_measurements, obstacle_velocities, self.sensor_angles, strict=False):
            v_y = v[1]
            if v_y != 0.0:
                zeta_v, zeta_x = self._determine_scaling(theta=sensor_angle)
                weight = 1 / (1 + np.exp(self.params.gamma_theta * abs(sensor_angle)))

                raw_penalty = self.params.alpha_x * np.exp(zeta_v * np.max((0, v_y)) - zeta_x * x)

                lambda_i = self._determine_tradeoff_coefficient(v_y=v_y, x_i=x)
                lambdas.append(lambda_i)

                num += weight * (1 - lambda_i) * raw_penalty
                den += weight

        if den == 0.0:
            # No dynamic obstacles detected
            self.tradeoff_coefficient = 1.0
            self.last_reward = 0.0
        else:
            self.tradeoff_coefficient = np.min(lambdas)
            self.last_reward = -num / den

        return self.last_reward

    def _determine_scaling(self, theta: float) -> tuple[float, float]:
        """Determine the velocity and distance scaling parameters.

        Based on the sensor angle theta for the reward calculation.

        Args:
            theta (float): angle of the obstacle relative to the ownship in radians,
                           where 0 is in front of the ownship.

        Returns:
            Tuple(float, float): Velocity scaling parameter zeta_v and distance
            scaling parameter zeta_x
        """
        if theta >= 0.0 and theta < np.deg2rad(112.5):
            return self.params.gamma_v_stb, self.params.gamma_x_stb
        elif theta < 0.0 and theta > -np.deg2rad(112.5):
            return self.params.gamma_v_port, self.params.gamma_x_port
        else:
            return self.params.gamma_v_stern, self.params.gamma_x_stern

    def _determine_tradeoff_coefficient(self, v_y: float, x_i: float) -> float:
        """Calculate the tradeoff coefficient lambda.

        Based on the distance and velocity of the TS.

        Args:
            v_y (float): The body-relative velocity of the TS
            x_i (float): The distance to the TS

        Returns:
            float: The tradeoff coefficient lambda
        """
        if v_y >= 0.0:
            # Choose alpha_lambda+ and gamma_lambda+
            alpha_lambda = self.params.alpha_lambda[1]
            gamma_lambda = self.params.gamma_lambda[1]
        else:
            alpha_lambda = self.params.alpha_lambda[0]
            gamma_lambda = self.params.gamma_lambda[0]

        den = 1 + np.exp(-gamma_lambda * x_i + alpha_lambda)

        return 1 / den

    def get_last_rewards_as_dict(self) -> dict:  # noqa: D102
        return {"r_dynamic_colav": self.last_reward}


class AutopilotReferenceRewarder(IReward):
    """Rewards the agent negatively for rapid changes in the output signals."""

    def __init__(self, env: "COLAVEnvironment", params: AutopilotReferenceRewardParams | None = None) -> None:
        super().__init__(env)

        self.course_coefficient = params.course_coefficient
        self.speed_coefficient = params.speed_coefficient
        self.h = env.simulator.dt

        self.course_range = env.action_type.course_range

        self.prev_course_delta = 0.0
        self.prev_speed_delta = 0.0

        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: csgym_action.Action | None = None, **kwargs) -> float:  # noqa: ARG002, ARG003
        if action is None:
            msg = "Action was not defined!"
            raise ValueError(msg)

        # Unpack the action
        speed_delta = action[1]
        course_delta = action[0]

        # Approximating the derivatives
        speed_delta_dot = (speed_delta - self.prev_speed_delta) / self.h
        course_delta_dot = (course_delta - self.prev_course_delta) / self.h

        # Calculate the rewards
        speed_dot_reward = -self.speed_coefficient * (speed_delta_dot**2)
        course_dot_reward = -self.course_coefficient * (course_delta_dot**2)

        self.prev_speed_delta = speed_delta
        self.prev_course_delta = course_delta

        self.last_reward = speed_dot_reward + course_dot_reward
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:  # noqa: D102
        return {"r_autopilot_reference": self.last_reward}


class Rewarder(IReward):
    """The rewarder class."""

    def __init__(self, env: "COLAVEnvironment", config: Config | None = None) -> None:
        self.config = config if config else Config()
        self._last_reward = 0.0
        self.rewarders = []
        for rewarder in self.config.rewarders:
            if isinstance(rewarder, ExistenceRewardParams):
                self.rewarders.append(ExistenceRewarder(rewarder))
            elif isinstance(rewarder, DistanceToGoalRewardParams):
                self.rewarders.append(DistanceToGoalRewarder(env, rewarder))
            elif isinstance(rewarder, CollisionRewardParams):
                self.rewarders.append(CollisionRewarder(env, rewarder))
            elif isinstance(rewarder, GroundingRewardParams):
                self.rewarders.append(GroundingRewarder(env, rewarder))
            elif isinstance(rewarder, TrajectoryTrackingRewardParams):
                self.rewarders.append(TrajectoryTrackingRewarder(env, rewarder))
            elif isinstance(rewarder, StaticColavRewardParams):
                self.rewarders.append(StaticColavRewarder(env, rewarder))
            elif isinstance(rewarder, DynamicColavRewardParams):
                self.rewarders.append(DynamicColavRewarder(env, rewarder))
            elif isinstance(rewarder, AutopilotReferenceRewardParams):
                self.rewarders.append(AutopilotReferenceRewarder(env, rewarder))

    def __call__(self, state: csgym_obs.Observation, action: csgym_action.Action | None = None, **kwargs) -> float:
        reward = 0.0
        for rewarder in self.rewarders:
            reward += rewarder(state, action, **kwargs)
        self._last_reward = reward
        return reward

    def get_last_rewards_as_dict(self) -> dict:  # noqa: D102
        rewards = {}
        for rewarder in self.rewarders:
            rewards.update(rewarder.get_last_rewards_as_dict())
        return rewards

    @property
    def last_reward(self) -> float:  # noqa: D102
        return self._last_reward
