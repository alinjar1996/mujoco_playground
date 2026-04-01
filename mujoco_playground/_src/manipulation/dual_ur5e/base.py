# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for Dual UR5e."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.dual_ur5e import ur5e_constants as consts

import os


def get_assets() -> Dict[str, bytes]:
  """Returns a dictionary of all assets used by the environment."""
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "ur5e"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  path = mjx_env.ROOT_PATH / "manipulation" / "ur5e" / "xmls"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class DualUR5eEnv(mjx_env.MjxEnv):
  """Base class for ALOHA environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._model_assets = get_assets()
    # self._mj_model = mujoco.MjModel.from_xml_string(
    #     epath.Path(xml_path).read_text(), assets=self._model_assets
    # )


    model_path = xml_path

    self._mj_model = mujoco.MjModel.from_xml_path(model_path)

    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    self._xml_path = model_path

    self._post_init_dualur5e()

  
  def _post_init_dualur5e(self, keyframe: str = None):
    """Initialize useful MuJoCo handles (IDs, indices, limits)."""

    model = self._mj_model

    # ---------------------------
    # 🔹 Sites (End-effectors)
    # ---------------------------
    # Adjust names to your XML
    self._tcp_site_0 = model.site("tcp_0").id
    self._tcp_site_1 = model.site("tcp_1").id

    joint_names_pos = list()
    joint_names_vel = list()
    for i in range(self._mj_model.njnt):
        joint_type = self._mj_model.jnt_type[i]
        n_pos = 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 4 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
        n_vel = 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
        
        for _ in range(n_pos):
            joint_names_pos.append(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))
        for _ in range(n_vel):
            joint_names_vel.append(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))

    # ---------------------------
    # 🔹 Geoms
    # ---------------------------
    # self._table_geom = model.geom("table_0").id

    self._init_q = jp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])
    self._init_ctrl =jp.zeros_like(self._init_q)
    # self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T
    # self._robot_joints = [self._mj_model.joint(j).id for j in consts.ARM_JOINTS]

    self._robot_joints = consts.ARM_JOINTS

    self._joint_mask_pos = np.isin(joint_names_pos, self._robot_joints)
    self._joint_mask_vel = np.isin(joint_names_vel, self._robot_joints)

    self._ball_qpos_idx = self._mj_model.body_dofadr[self._mj_model.body(name="ball").id]

    # Robot geoms (already partially done in your code)
    self._robot_geom_ids = [
        i for i in range(self._mj_model.ngeom)
        if mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        and mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, i).startswith("robot")
    ]

    # self._arm_qadr = jp.array(
    #     [self._mj_model.jnt_qposadr[joint_id] for joint_id in self._robot_geom_ids]
    # )
    data = mujoco.MjData(self._mj_model)

    self._ball_init_pose = data.qpos[self._ball_qpos_idx:self._ball_qpos_idx+7].copy()
    self._ball_base_pose = self._ball_init_pose.copy()

    # ---------------------------
    # 🔹 Mocap IDs
    # ---------------------------
    # self._target_mocap_id = model.body_mocapid[self._target_body_id]


  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    # return self._mjx_model.nv
    # return int(np.sum(self._joint_mask_vel))
    # ACtion is only the cost weights for the planner, not the actual control inputs to the robot, so we can set it to a fixed size.
    return 12
  
  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  # def hand_table_collision(self, data) -> jp.ndarray:
  #   # Check for collisions with the floor.
  #   hand_table_collisions = [
  #       data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
  #       for sensorid in self._table_finger_found_sensor
  #   ]
  #   return (sum(hand_table_collisions) > 0).astype(float)
