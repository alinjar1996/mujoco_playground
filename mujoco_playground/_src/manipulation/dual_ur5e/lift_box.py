from typing import Any, Dict, Optional, Union
import numpy as np
import mujoco
from mujoco import mjx
import time
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.dual_ur5e.external.manipulator_mujoco.real_demo.sampling_based_planner.mpc_planner import run_cem_planner
import os
from mujoco_playground._src.manipulation.dual_ur5e.external.manipulator_mujoco.real_demo.sampling_based_planner.quat_math import quaternion_multiply, rotation_quaternion, quaternion_distance
from ml_collections import config_dict
import jax
from jax import config, numpy as jp
from mujoco_playground._src.manipulation.dual_ur5e import base as dual_ur5e_base

REASON_OK = 0
REASON_FALL = 1
REASON_COLLISION = 2
REASON_TIMEOUT = 3

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.1,
      sim_dt=0.1,
      episode_length=50,  # 5 sec.
      action_repeat=1,
      action_scale=1, #0.015,

      
      reward_config=config_dict.create(
        scales=config_dict.create(
            collision=500,
            theta=0.3,
            velocity=0.1,
            z_axis=10.0,
            distance=5.0,
            orientation=4.0,
            eef_to_obj=7.0,
            obj_to_targ=3.0,
            eef_to_obj_move=3.0,
            object_orientation=6.0,
            ),
        ),

      impl='jax',
      nconmax=24 * 8192,
      njmax=88,
  )

class LiftBox(dual_ur5e_base.DualUR5eEnv):

    def __init__(self, config=default_config(), config_overrides=None, view=False):

        super().__init__(
            xml_path=os.path.abspath("../../mujoco_playground/_src/manipulation/dual_ur5e/xmls/scene.xml"),  
            config=config,
            config_overrides=config_overrides,
        )



        self.use_hardware = False
        self.record_data_bench = False
        self.record_data_traj = False


        # Planner params
        self.num_dof = 12
        self.home_joint_position = jp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])
        self.init_joint_position = self.home_joint_position
        self.init_noise = jp.full(self.num_dof, 0.3)        
        self.ball_init_pos_noise = jp.full(3, 0.01)

        print("INIT NOISE", self.init_noise)



        self.num_batch=2000
        self.num_steps=12
        self.maxiter_cem=3
        self.maxiter_projection=5
        self.num_elite=0.5
        self.sim_timestep=config.sim_dt
        self.timestep=self.sim_timestep
        self.position_threshold=0.06
        self.rotation_threshold=0.1
        self.flow_inference=False
        self.flow_weight_path=None
        self.flow_inference_fraction = 0.8 if self.flow_inference else 0.0


        



        self.grab_pos_thresh = 0.05
        self.grab_rot_thresh = 0.1
        self.grab_dist_thresh = 0.05
        self.thetadot = jp.zeros(self.num_dof)
        

        # Initialize robot connection
        self.rtde_c_0 = None
        self.rtde_r_0 = None

        self.rtde_c_1 = None
        self.rtde_r_1 = None

        self.grippers = {
            '0': {
                'srv': None,
                'state': 'open'
            },
            '1': {
                'srv': None,
                'state': 'open'
            }
        }


        # model_path = os.path.abspath("../../mujoco_playground/_src/manipulation/dual_ur5e/xmls/scene.xml")

        # self._mj_model = mujoco.MjModel.from_xml_path(model_path)
        # self._mj_model.opt.timestep = self.timestep #self.timestep

        # self._mj_model = mjx_env.MjxModel(self._mj_model)

  
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
        
        
        robot_joints = np.array(['shoulder_pan_joint_1', 'shoulder_lift_joint_1', 'elbow_joint_1', 'wrist_1_joint_1', 'wrist_2_joint_1', 'wrist_3_joint_1',
                                'shoulder_pan_joint_2', 'shoulder_lift_joint_2', 'elbow_joint_2', 'wrist_1_joint_2', 'wrist_2_joint_2', 'wrist_3_joint_2'])
        
        self.joint_mask_pos = np.isin(joint_names_pos, robot_joints)
        self.joint_mask_vel = np.isin(joint_names_vel, robot_joints)

        self.ball_qpos_idx = self._mj_model.body_dofadr[self._mj_model.body(name="ball").id]

        self.data = mujoco.MjData(self._mj_model)

        self.data.qpos[self.joint_mask_pos] = self.init_joint_position


        mujoco.mj_forward(self._mj_model, self.data)

        self.ball_init_pose = self.data.qpos[self.ball_qpos_idx:self.ball_qpos_idx+7].copy()
        self.ball_base_pose = self.ball_init_pose.copy()

        ball_init_pose = jp.array(self.ball_base_pose)

        ball_init_pose = ball_init_pose.at[:2].add(
            jp.array(self.ball_init_pos_noise[:2])
        )

        
        # self.success = 0
        # self.reason = 'na'

        

        self.flow_inference = False
        self.flow_inference_fraction = 0.8 if self.flow_inference else 0


        
# Initialize CEM/MPC planner
        self.planner = run_cem_planner(
            model=self._mj_model,
            data=self.data,
            num_dof=self.num_dof,
            num_batch=self.num_batch,
            num_steps=self.num_steps,
            maxiter_cem=self.maxiter_cem,
            maxiter_projection=self.maxiter_projection,
            num_elite=self.num_elite,
            timestep=self.timestep,
            position_threshold=self.position_threshold,
            rotation_threshold=self.rotation_threshold
        )

        self.robot_geom_ids = set()

        for i in range(self._mj_model.ngeom):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.startswith("robot"):
                self.robot_geom_ids.add(i)

        # self.reset_simulation()

        

        # Planning timer - runs at lower frequency
        
        # Control timer - runs at simulation frequency
        
        if view:
        # Setup viewer
            self.viewer = mujoco.viewer.launch_passive(self._mj_model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.viewer.cam.lookat[:] = self._mj_model.body(name='table_0').pos
            self.viewer.cam.distance = 3.0 
            self.viewer.cam.azimuth = 90.0 
            self.viewer.cam.elevation = -30.0
    
    
    
    def _generate_targets(self, rng):

        rng, key_pos, key_mag, key_sign = jax.random.split(rng, 4)

        area_center_1 = jp.array([-0.3, -0.1, 0.35])
        area_size_1 = jp.array([0.05, 0.05, 0.05])

        # sample position
        target_pos = area_center_1 + jax.random.uniform(
            key_pos,
            shape=(3,),
            minval=-area_size_1,
            maxval=area_size_1
        )

        # sample rotation magnitude
        magnitude = jax.random.uniform(
            key_mag,
            shape=(),
            minval=5.0,
            maxval=20.0
        )

        # sample sign (-1 or 1)
        sign = jax.random.choice(key_sign, jp.array([-1.0, 1.0]))

        angle_deg = sign * magnitude

        target_rot = quaternion_multiply(
            jp.array([1.0, 0.0, 0.0, 0.0]),
            rotation_quaternion(angle_deg, jp.array([0.0, 0.0, 1.0]))
        )

        return target_pos, target_rot, rng
    
    def reset(self, rng: jax.Array) -> mjx_env.State:

        rng, rng_joint = jax.random.split(rng)



        penalty_z = 0
        penalty_r_s = 0
        penalty_col = 0

        penalty_collision_real_time = 0

        ball_qpos_idx = self._mj_model.body_dofadr[self._mj_model.body(name="ball").id]

        
        ball_base_pose = self.ball_base_pose  # already stored as numpy
        ball_init_pose = jp.array(ball_base_pose)

        ball_init_pos_noise = jp.full(3, 0.01)

        ball_init_pose = ball_init_pose.at[:2].add(ball_init_pos_noise[:2])

        # self._mj_model.body(name='ball').pos += ball_init_pose[:3]


        init_joint_position = self.home_joint_position + jax.random.uniform(
            rng_joint,
            (self.num_dof,),
            minval=-self.init_noise,
            maxval=self.init_noise
        )

        qpos = jp.zeros(self._mjx_model.nq)
        qvel = jp.zeros(self._mjx_model.nv)
        qpos = qpos.at[self.joint_mask_pos].set(init_joint_position)


        # self.task='pick'

        # qpos[self.ball_qpos_idx : self.ball_qpos_idx+7] = self.ball_init_pose
        # qpos[self.joint_mask_pos] = self.init_joint_position
        # qvel[self.joint_mask_vel] = jp.zeros(self.init_joint_position.shape)

        qpos = qpos.at[ball_qpos_idx : ball_qpos_idx+7].set(ball_init_pose)
        qpos = qpos.at[self.joint_mask_pos].set(init_joint_position)

        qvel = qvel.at[self.joint_mask_vel].set(
            jp.zeros_like(init_joint_position))

        
        target_pos, target_rot, rng = self._generate_targets(rng)
        
        

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel
        #    impl=self._mjx_model.impl.value,
        )

        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]].set(target_rot)
            )
        
        # info['target_0'] = data.mocap_pos[self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]]

        
        # self.planner.target_0[:3] = data.mocap_pos[self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]]
        # self.planner.target_0[3:] = data.mocap_quat[self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]]
        
        # -------------------------------
        # Planner state (JAX)
        # -------------------------------
        num_dof = self.num_dof
        nvar_single = self.planner.cem.nvar_single
        nvar = self.planner.cem.nvar

        nvar_single = 12
        nvar= num_dof * nvar_single

        cov_scalar_coeff= 10

        xi_cov = jp.kron(
            jp.eye(num_dof),
            cov_scalar_coeff * jp.eye(nvar_single)
        )

        xi_mean = jp.zeros(nvar)


        info = {
            'rng': rng,
            '_steps': jp.array(0),

            # planner state
            'xi_cov': xi_cov,
            'xi_mean': xi_mean,
            'task': jp.array(0),  # 0=pick, 1=move

            # episode state
            'success': jp.array(0),
            # 'reason': jp.array(0),  # 0=ok,1=fall,2=collision,3=timeout
            'reason': jp.array(REASON_OK),

            # penalties
            'penalty_z': jp.array(0.0),
            'penalty_r_s': jp.array(0.0),
            'penalty_col': jp.array(0.0),
            'penalty_collision_real_time': jp.array(0.0),
            # 'target_0': self.planner.target_0,
            'target_0': jp.concatenate([target_pos, target_rot])
        }

        obs = self._get_obs(data, info)

        print("obs shape:", obs.shape)

        reward, done = jp.zeros(2)

        metrics = {
            'out_of_bounds': jp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }

        return mjx_env.State(
            data,
            obs,
            reward,
            done,
            metrics,
            info,
        )
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        info = state.info.copy()

        task = info['task']
        success = info['success']
        reason = info['reason']

        penalty_z = info['penalty_z']
        penalty_r_s = info['penalty_r_s']
        penalty_col = info['penalty_col']
        penalty_collision_real_time = info['penalty_collision_real_time']


        newly_reset = state.info['_steps'] == 0
        state.info['episode_picked'] = jp.where(
            newly_reset, 0, state.info['episode_picked']
        )
        state.info['prev_potential'] = jp.where(
            newly_reset, 0.0, state.info['prev_potential']
        )


        # Scale action → joint velocities
        qvel = state.data.qvel

        # Apply only on controlled joints
        qvel = qvel.at[self.joint_mask_vel].set(action * self._config.action_scale)

        # Update data with new velocities
        data = state.data.replace(qvel=qvel)

        # Step physics
        data = mjx_env.step(self._mjx_model,data,self.n_substeps)

        # raw_rewards, success = self._get_reward(data, state.info)
        raw_rewards = self._get_reward(data, state.info)

        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in raw_rewards.items()
        }
        potential = sum(rewards.values()) / sum(
            self._config.reward_config.scales.values()
        )

        # # Reward progress. Clip at zero to not penalize mistakes like dropping
        # # during exploration.
        reward = jp.maximum(
            potential - state.info['prev_potential'], jp.zeros_like(potential)
        )



        state.info['prev_potential'] = jp.maximum(
            potential, state.info['prev_potential']
        )

        reward = jp.where(newly_reset, 0.0, reward)  # Prevent first-step artifact

        # # No reward information if you've dropped a block after you've picked it up.
        # picked = box_pos[2] > 0.15
        # state.info['episode_picked'] = jp.logical_or(
        #     state.info['episode_picked'], picked
        # )
        # dropped = (box_pos[2] < 0.05) & state.info['episode_picked']
        # reward += dropped.astype(float) * -0.1  # Small penalty.

        # out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        # out_of_bounds |= box_pos[2] < 0.0
        # done = (
        #     out_of_bounds
        #     | jp.isnan(data.qpos).any()
        #     | jp.isnan(data.qvel).any()
        #     | dropped
        # )
        cost_c = -raw_rewards['collision']


        penalty_collision_real_time += self._collision_check(state)

        
        target_1_pos = state.data.site_xpos[self.planner.cem.box_site_1_id]
        target_2_pos = state.data.site_xpos[self.planner.cem.box_site_2_id]
        
        current_cost_g_0 = jp.linalg.norm(state.data.site_xpos[self.planner.tcp_id_0]-target_1_pos)
        
        current_cost_g_1 = jp.linalg.norm(state.data.site_xpos[self.planner.tcp_id_1]-target_2_pos)
        
        current_cost_g = (current_cost_g_0 + current_cost_g_1)/2


        current_cost_r_0 = quaternion_distance(state.data.xquat[self.planner.hande_id_0], jp.array([0.183, -0.683, -0.683, 0.183]))
        current_cost_r_1 = quaternion_distance(state.data.xquat[self.planner.hande_id_1], jp.array([0.183, -0.683, 0.683, -0.183]))

        cost_g_ball = jp.linalg.norm(state.data.xpos[self._mj_model.body(name='ball').id] - info['target_0'][:3])
        cost_r_ball = jp.linalg.norm(state.data.xquat[self._mj_model.body(name='ball').id] - info['target_0'][3:])


        if task==0: # pick task
            target_reached = (
                    current_cost_g < self.grab_pos_thresh \
                    # and cost_dist < 0.04 \
                    and current_cost_r_0 < self.grab_rot_thresh \
                    and current_cost_r_1 < self.grab_rot_thresh
            )

            if target_reached:
                task = 1  # move task

        elif task == 1:  # move task
            target_reached = cost_g_ball < 0.04 and cost_r_ball<0.1
            if target_reached:
                print("======================= TARGET REACHED =======================", flush=True)
                success = 1
                reason = REASON_OK
                # self.reset_simulation()
                
            elif current_cost_g > 0.2:
                print("======================= TARGET FAILED: BALL FALL =======================", flush=True)
                success = 0
                reason = REASON_FALL
                penalty_z += 5.0
                # self.reset_simulation()
        
        # if time.time() - info['time'] > 30:
        #     print("======================= TARGET FAILED: TIMEOUT =======================", flush=True)
        #     success = 0
        #     reason = REASON_TIMEOUT
        #     # self.reset_simulation()

        if cost_c > 300:
            print("======================= TARGET FAILED: COLLISION =======================", flush=True)
            success = 0
            reason = REASON_COLLISION
            penalty_col += 0.5
            # self.reset_simulation()
        
        out_of_bounds = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        
        failed = (reason == REASON_FALL) | (reason == REASON_COLLISION) | (reason == REASON_TIMEOUT)
        
        # done = (target_reached|failed|out_of_bounds)

        target_reached = jp.array(target_reached)
        failed = jp.array(failed)

        done = target_reached | failed | out_of_bounds


        info.update({
            'task': task,
            'success': success,
            'reason': reason,
            'penalty_z': penalty_z,
            'penalty_r_s': penalty_r_s,
            'penalty_col': penalty_col,
            'penalty_collision_real_time': penalty_collision_real_time,
            # 'time': time.time()
        })


        reward += -penalty_collision_real_time #penalty for collision in real time
        reward += failed.astype(float) * -5000  # Large penalty.for failure

        state.info['_steps'] += self._config.action_repeat
        state.info['_steps'] = jp.where(
            done | (state.info['_steps'] >= self._config.episode_length),
            0,
            state.info['_steps'],
        )

        state.metrics.update(**rewards, out_of_bounds=out_of_bounds.astype(float))

        obs = self._get_obs(data, state.info)
        return mjx_env.State(
            data, obs, reward, done.astype(float), state.metrics, info
        )
    
    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:


        # (xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples, init_pos, init_vel, 
        #             target_ball_dest, ball_pos, ball_pick_init, cost_weights, cost_task_weights) = carry
        
        xi_cov = info['xi_cov']
        xi_mean = info['xi_mean']

        

        state_term = data.qvel[self.joint_mask_vel] #thetadot_init
        lamda_init = self.planner.lamda_init
        s_init = self.planner.s_init
        xi_samples = self.planner.xi_samples
        init_pos = data.qpos[self.joint_mask_pos]
        init_vel = data.qvel[self.joint_mask_vel]

        target_ball_dest = info['target_0']

        ball_pos = data.qpos[self.ball_qpos_idx:self.ball_qpos_idx+3]
        
        ball_pick_init = self.ball_init_pose[:3]
        cost_weights = None 

        cost_task_weights = {'pick': 0,'move': 0}
        # cost_task_weights = 

        xi_mean_prev = xi_mean 
        xi_cov_prev = xi_cov

        xi_samples_reshaped = xi_samples.reshape(1, self.num_dof, self.planner.cem.P.shape[1])
        xi_samples_batched_over_dof = jp.transpose(xi_samples_reshaped, (1, 0, 2)) # shape: (DoF, B, P.shape[1])

        state_term_reshaped = state_term.reshape(1, self.num_dof, 1)
        state_term_batched_over_dof = jp.transpose(state_term_reshaped, (1, 0, 2)) #Shape: (DoF, B, 1)

        lamda_init_reshaped = lamda_init.reshape(1, self.num_dof, self.planner.cem.P.shape[1])
        lamda_init_batched_over_dof = jp.transpose(lamda_init_reshaped, (1, 0, 2)) # shape: (DoF, B, P.shape[1])

        s_init_reshaped = s_init.reshape(1, self.num_dof, self.planner.cem.num_total_constraints_per_dof )
        s_init_batched_over_dof = jp.transpose(s_init_reshaped, (1, 0, 2)) # shape: (DoF, B, num_total_constraints_per_dof)


        
        # Pass all arguments as positional arguments; not keyword arguments
        xi_filtered, primal_residuals, fixed_point_residuals = self.planner.cem.compute_projection_batched_over_dof(
                                                                xi_samples_batched_over_dof, 
                                                                state_term_batched_over_dof, 
                                                                lamda_init_batched_over_dof, 
                                                                s_init_batched_over_dof, 
                                                                init_pos)
        
        xi_filtered = xi_filtered.transpose(1, 0, 2).reshape(self.num_batch, -1) # shape: (B, num*num_dof)
        
        primal_residuals = jp.linalg.norm(primal_residuals, axis = 0)
        fixed_point_residuals = jp.linalg.norm(fixed_point_residuals, axis = 0)
                
        avg_res_primal = jp.sum(primal_residuals, axis = 0)/self.maxiter_projection
        
        avg_res_fixed_point = jp.sum(fixed_point_residuals, axis = 0)/self.maxiter_projection

        thetadot = jp.dot(self.planner.cem.A_thetadot, xi_filtered.T).T
        thetaddot = jp.dot(self.planner.cem.A_thetaddot, xi_filtered.T).T
        # thetadot = jjp.dot(self.A_thetadot, xi_samples.T).T


        (theta, eef_0, eef_vel_lin_0, eef_vel_ang_0, eef_1, 
         eef_vel_lin_1, eef_vel_ang_1, ball, 
        target_1_pos, target_2_pos, 
        target_1_rot, target_2_rot, collision) = self.planner.cem.compute_rollout_single(thetadot,init_pos, 
                                                                                                    init_vel, 
                                                                                                    target_ball_dest, 
                                                                                                    ball_pos, 
                                                                                                    ball_pick_init)
        
        cost, cost_list = self.planner.cem.compute_cost_single(theta, thetaddot, eef_0, eef_vel_lin_0, eef_vel_ang_0, 
                                                        eef_1, eef_vel_lin_1, eef_vel_ang_1, ball,
                                                        target_1_pos, target_2_pos, target_1_rot, target_2_rot,
                                                        collision, target_ball_dest, ball_pos, cost_weights, cost_task_weights)

        rewards = {
            'collision': - (cost_list[0]+cost_list[1]),
            'theta': - cost_list[2],
            'velocity': -cost_list[3],
            'z-axis': -cost_list[4],
            'distance': -(cost_list[5]+10*cost_list[6]),
            'orientation': -cost_list[7],
            'eef_to_obj': -cost_list[8],
            'obj_to_targ': -cost_list[9],
            'eef_to_obj_move': -cost_list[10],
            'object_orientation': -cost_list[11],
        }

        return rewards
    


    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:


        obs = jp.concatenate([
            data.qpos,
            data.qvel,
            data.mocap_pos[self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]], #target pos
            data.mocap_quat[self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]], #target rot
            (info['_steps'].reshape((1,)) / self._config.episode_length).astype(
                float
            ),
        ])

        return obs

    
    def _collision_check(self,  state:mjx_env.State):

        has_collision = False
        penalty_collision_real_time = 0
        num_penetration = 0

        for i in range(state.data.ncon):
            contact = state.data.contact[i]

            if (contact.geom1 in self.robot_geom_ids or
                contact.geom2 in self.robot_geom_ids):

                if contact.dist < -1e-6:   # penetration threshold
                    has_collision = True
                    
                    num_penetration+=1

                    name1 = mujoco.mj_id2name(
                        self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                    name2 = mujoco.mj_id2name(
                        self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                    if state.info['task'] == 0:
                        print(f"Collision detected between {name1} and {name2}")
                    # break

        if has_collision and state.info['task'] == 0:
            print("================== COLLISION (SIM) ==================", flush=True)
            # self.success = 0
            # self.reason = REASON_COLLISION
            # self.reset_simulation()
            penalty_collision_real_time = 10*num_penetration

        return penalty_collision_real_time

    
