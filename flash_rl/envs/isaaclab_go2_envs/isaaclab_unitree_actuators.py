from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from .go2_actuator_math import unitree_go2hv_clip_effort


class MotorStrengthUnitreeActuator(DelayedPDActuator):
    """Unitree actuator with env-owned motor-strength scaling before torque-speed clipping."""

    cfg: "UnitreeActuatorCfg"

    def __init__(self, cfg: "UnitreeActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self.motor_strength = torch.ones_like(self.computed_effort)
        self.scaled_motor_strength = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)
        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)

    def reset(self, env_ids):
        super().reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        feedforward_effort = control_action.joint_efforts
        if feedforward_effort is None:
            feedforward_effort = torch.zeros_like(control_action.joint_positions)

        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(feedforward_effort)

        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        raw_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        self.computed_effort = raw_effort * self.motor_strength
        self.applied_effort = unitree_go2hv_clip_effort(
            self.computed_effort,
            joint_vel,
            x1=self._velocity_x1,
            x2=self._velocity_x2,
            y1=self._effort_y1,
            y2=self._effort_y2,
        )
        self.applied_effort -= (
            self._friction_static * torch.tanh(joint_vel / self._activation_vel) + self._friction_dynamic * joint_vel
        )

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class UnitreeActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for Unitree torque-speed actuators."""

    class_type: type = MotorStrengthUnitreeActuator

    X1: float = 1e9
    """Maximum speed at full torque, in rad/s."""

    X2: float = 1e9
    """No-load speed where the effort limit reaches zero, in rad/s."""

    Y1: float = MISSING
    """Peak torque when requested torque and joint velocity have the same sign, in N*m."""

    Y2: float | None = None
    """Peak torque when requested torque and joint velocity have opposite signs, in N*m."""

    Fs: float = 0.0
    """Static friction coefficient for the optional explicit friction term."""

    Fd: float = 0.0
    """Dynamic friction coefficient for the optional explicit friction term."""

    Va: float = 0.01
    """Velocity where the optional explicit friction term is fully activated."""


@configclass
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    X1: float = 13.5
    X2: float = 30.0
    Y1: float = 20.2
    Y2: float = 23.4


__all__ = [
    "MotorStrengthUnitreeActuator",
    "UnitreeActuatorCfg",
    "UnitreeActuatorCfg_Go2HV",
]
