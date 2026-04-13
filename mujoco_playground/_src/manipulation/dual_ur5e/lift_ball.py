from typing import Any, Dict, Optional, Union
import numpy as np
import mujoco
from mujoco import mjx
import time
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.dual_ur5e.external.manipulator_mujoco.real_demo import data
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
      # sim_dt is actual physics timestep, ctrl_dt is how often we call the policy
      ctrl_dt=0.2,
      sim_dt=0.1,
      episode_length=100,  
      action_repeat=1,
      action_scale=1, #0.015,

      
      reward_config=config_dict.create(
        scales=config_dict.create(
            robot_eef_pos=1.0,
            robot_eef_rot=1.0,
            ball_target_pos=1.0,
            ball_target_rot=1.0,
            ball_pick_success=20.0,
            ball_reach_success=40.0,
            collision_real_time=5.0,
            collision_plan_horizon=1.0,
            fall_plan_horizon=1.0,
            r_s_plan_horizon=1.0,
            ),
        ),

      impl='jax',
      nconmax=24 * 8192,
      njmax=88,
  )

class LiftBall(dual_ur5e_base.DualUR5eEnv):

    def __init__(self, config=default_config(), config_overrides=None, 
                 home_joint_position=np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])):

        super().__init__(
            # xml_path=os.path.abspath("../../mujoco_playground/_src/manipulation/dual_ur5e/xmls/scene.xml"),  
            xml_path=os.path.abspath("../../mujoco_playground/_src/manipulation/dual_ur5e/external/" \
                                        "manipulator_mujoco/real_demo/ur5e_hande_mjx/scene.xml"),  
            config=config,
            config_overrides=config_overrides,
        )



        self.use_hardware = False
        self.record_data_bench = False
        self.record_data_traj = False
        

        # Planner params
        self.num_dof = 12
        # self.home_joint_position = jp.array([ 1.26155823, -1.63279201,  1.71236772, -1.87456712, 
        #                                      -1.35176384, -0.26180875, -1.24849328, -1.65907332,  
        #                                      1.45611901, -1.6316204,  -1.64648593,  0.23419371])
        # self.home_joint_position = home_joint_position
        # self.home_joint_position = jp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])
        
        self.home_joint_position = home_joint_position
        self.init_joint_position = self.home_joint_position
        self.init_noise = jp.full(self.num_dof, 0.01)        
        self.ball_init_pos_noise = jp.full(3, 0.01)

        print("INIT NOISE", self.init_noise)

        self.num_batch=100
        self.num_steps=15
        self.maxiter_cem=2
        self.maxiter_projection=5
        self.num_elite=0.5
        self.sim_timestep=config.sim_dt
        self.timestep=self.sim_timestep
        self.position_threshold=0.06
        self.rotation_threshold=0.1
        self.flow_inference=False
        self.flow_weight_path=None
        self.flow_inference_fraction = 0.8 if self.flow_inference else 0.0


        self.grab_pos_thresh = 0.02
        self.grab_rot_thresh = 0.1
        self.grab_dist_thresh = 0.03
        self.thetadot = jp.zeros(self.num_dof)


        cost_weights_dict = {
            'collision_pick': 200,
            'collision_move': 200,
            'theta': 0.2,
            'velocity': 0.04,
            'z_axis': 1.5,
            'distance_pick': 5.0,
            'distance_move': 5.0,
            'orientation': 0.5,
            'eef_to_obj': 5.0,
            'obj_to_targ': 5.0,
            'smoothness': 0.1,
        }

        # Base scales for softplus transformation (order matches compute_cost_single cost_list):
        # [collision_pick, collision_move, theta, velocity, z_axis,
        #  distance_pick, distance_move, orientation, eef_to_obj,
        #  obj_to_targ, object_orientation, smoothness]
        self._base_scales = jp.array([
            200.0, 200.0, 0.2, 0.04, 1.5,
            5.0, 5.0, 0.5, 5.0, 5.0, 0.1
        ])

        self.cost_weights = self._base_scales.copy()  # Initialize cost weights to base scales
        

        data=mujoco.MjData(self._mj_model)

        self.joint_mask_pos = self._joint_mask_pos
        self.joint_mask_vel = self._joint_mask_vel

        data.qpos[self.joint_mask_pos] = self.init_joint_position

        mujoco.mj_forward(self._mj_model, data)



# Initialize CEM/MPC planner
        self.planner = run_cem_planner(
            model=self._mj_model,
            data=data,
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

    
    def _build_cost_weights(self, scales):
        def get(x, k):
            return x[k] if isinstance(x, dict) else getattr(x, k)

        return jp.array([
            get(scales, 'collision'),
            get(scales, 'collision'),
            get(scales, 'theta'),
            get(scales, 'velocity'),
            get(scales, 'z_axis'),
            get(scales, 'distance'),
            get(scales, 'distance'),
            get(scales, 'orientation'),
            get(scales, 'eef_to_obj'),
            get(scales, 'obj_to_targ'),
            get(scales, 'object_orientation'),
            get(scales, 'smoothness'),
        ])
    
    
    def _generate_targets(self, rng):

        rng, key_pos,  = jax.random.split(rng, 2)

        area_center_1 = jp.array([-0.3, -0.0, 0.26])
        area_size_1 = jp.array([0.05, 0.05, 0.03])

        target_pos = area_center_1 + jax.random.uniform(
            key_pos,
            shape=(3,),
            minval=-area_size_1,
            maxval=area_size_1
        )

        return target_pos, rng

        
    
    def reset(self, rng: jax.Array) -> mjx_env.State:

        rng, rng_joint = jax.random.split(rng)


        ball_qpos_idx = self._mj_model.body_dofadr[self._mj_model.body(name="ball").id]

        
        ball_base_pose =jp.array(self._ball_base_pose)  # already stored as numpy
        ball_init_pose = ball_base_pose

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


        qpos = qpos.at[ball_qpos_idx : ball_qpos_idx+7].set(ball_init_pose)
        qpos = qpos.at[self.joint_mask_pos].set(init_joint_position)

        qvel = qvel.at[self.joint_mask_vel].set(
            jp.zeros_like(init_joint_position))

        
        target_pos, rng = self._generate_targets(rng)


        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros((self._mjx_model.nu,), dtype=jp.float32)
        #    impl=self._mjx_model.impl.value,
        )

        mocap_id = self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]
        data = data.replace(
            mocap_pos=data.mocap_pos.at[mocap_id].set(target_pos),
            )
        
        data = mjx.forward(self._mjx_model, data)
        
        target_pos_0 = data.xpos[self._mj_model.body(name="target_0").id]
        target_rot_0 = data.xquat[self._mj_model.body(name="target_0").id].copy()
        target_0 = jp.concatenate([target_pos_0, target_rot_0])

        target_pos_2 = data.xpos[self._mj_model.body(name="ball").id]
        target_rot_2 = data.xquat[self._mj_model.body(name="ball").id].copy()
        target_2 = jp.concatenate([target_pos_2, target_rot_2])

        ball_pick_init = target_2

        

        # Rebuild target_0 from the forward-stepped data (pos + quat), JAX-compatible
        target_0 = jp.concatenate([
            data.mocap_pos[mocap_id],
            data.mocap_quat[mocap_id],
        ])
        
        # -------------------------------
        # Planner state (JAX)
        # -------------------------------
        num_dof = self.num_dof
        nvar_single = self.planner.cem.nvar_single
        nvar = self.planner.cem.nvar

        # nvar= num_dof * nvar_single

        cov_scalar_coeff= self.planner.cov_coeff_scalar

        xi_cov = jp.kron(
            jp.eye(num_dof),
            cov_scalar_coeff * jp.eye(nvar_single)
        )

        xi_mean = jp.zeros(nvar)


        info = {
            'rng': rng,
            '_steps': jp.array(0, dtype=int),

            # planner state
            'xi_cov': xi_cov,
            'xi_mean': xi_mean,
            'task': jp.array(0),  # 0=pick, 1=move

            # episode state
            'success': jp.array(0),
            # 'reason': jp.array(0),  # 0=ok,1=fall,2=collision,3=timeout
            'reason': jp.array(REASON_OK),
            #Current cost states
            'current_cost_g': jp.array(0.0),
            'current_cost_r': jp.array(0.0),
            'cost_g_ball': jp.array(0.0),
            'cost_r_ball': jp.array(0.0),

            # penalties
            'penalty_z': jp.array(0.0),
            'penalty_r_s': jp.array(0.0),
            'penalty_col': jp.array(0.0),
            'penalty_collision_real_time': jp.array(0.0),
            # 'target_0': self.planner.target_0,
            'target_0': target_0,
            'target_2': target_2,
            'ball_pick_init': ball_pick_init,
            'prev_potential': jp.array(0.0, dtype=float),
            'prev_reward': jp.array(0.0, dtype=float),
            'eef_0_planned': jp.zeros((self.num_steps, 7)),
            'eef_1_planned': jp.zeros((self.num_steps, 7)),
            # 'target_1_pos': jp.zeros(3),
            # 'target_2_pos': jp.zeros(3),
            
        }

        obs = self._get_obs(data, info)

        # print("obs shape:", obs.shape)

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
    
    
    
    def _run_cem_planning(self, data: mjx.Data, info: Dict[str, Any]) -> tuple:
        """Run CEM planning iterations and return the best first-step action and updated planner state."""
        task = info['task']
        cost_task_weights = {
            'pick': (task == 0).astype(jp.float32),
            'move': (task == 1).astype(jp.float32),
        }
        out = self.planner.cem.compute_cem(xi_mean=info['xi_mean'],
                                    xi_cov=info['xi_cov'],
                                    init_pos=data.qpos[self.joint_mask_pos],
                                    init_vel=data.qvel[self.joint_mask_vel],
                                    init_acc=jp.zeros_like(data.qvel[self.joint_mask_vel]),
                                    target_0=info['target_0'],
                                    target_2=info['target_2'],
                                    ball_pick_init = info['ball_pick_init'],
                                    lamda_init=self.planner.lamda_init,
                                    s_init=self.planner.s_init,
                                    xi_samples=self.planner.xi_samples,
                                    cost_weights= self.cost_weights, #self._build_cost_weights(info['cost_weights']),
                                    cost_task_weights=cost_task_weights
                                    )
        
        (cost, best_cost_list, best_vels, best_traj, xi_mean, xi_cov, thetadot, theta,
		avg_res_primal,avg_res_fixed,primal_residuals,fixed_point_residuals,idx_min,
		ball_out, eef_0_planned,eef_1_planned,eef_0,eef_1) = out


        cem_action = jp.mean(best_vels[1:6], axis=0)

        return cem_action, xi_mean, xi_cov, best_cost_list, eef_0_planned, eef_1_planned

       

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        # info = state.info.copy()

        task = state.info['task']        
        success = state.info['success']
        reason = state.info['reason']

        penalty_z = state.info['penalty_z']
        penalty_r_s = state.info['penalty_r_s']
        penalty_col = state.info['penalty_col']
        penalty_collision_real_time = state.info['penalty_collision_real_time']

        target_2 = state.info['target_2'].at[:3].set(
            state.data.xpos[self._mj_model.body(name='ball').id]
        )


        newly_reset = state.info['_steps'] == 0
        state.info['prev_potential'] = jp.where(
            newly_reset, 0.0, state.info['prev_potential']
        )

        raw_weights = action

        # softplus ensures positive weights; multiply by base scales to recover correct magnitudes
        # self.cost_weights = jax.nn.softplus(raw_weights) * self._base_scales * self._config.action_scale
        self.cost_weights = jax.nn.softplus(raw_weights) * self._base_scales
        #softplus is log(1 + exp(x)), which smoothly maps real numbers to positive numbers

        # self.cost_weights = raw_weights


        # ---- Run CEM planning to get optimal joint velocities ----
        (cem_action, xi_mean_new, 
         xi_cov_new, best_cost_list,
         eef_0_planned, eef_1_planned) = self._run_cem_planning(state.data, state.info)
        
    
        # Update planner state in info
        state.info['xi_mean'] = xi_mean_new
        state.info['xi_cov'] = xi_cov_new
        state.info['eef_0_planned'] = eef_0_planned
        state.info['eef_1_planned'] = eef_1_planned

        # Apply CEM-optimized velocities on controlled joints
        qvel = state.data.qvel
        qvel = qvel.at[self.joint_mask_vel].set(cem_action) #Jut cem output

        # Update data with new velocities
        data = state.data.replace(qvel=qvel)

        # Step physics (no torque control, velocity-driven)
        data = mjx_env.step(model = self._mjx_model,
                            data =data,
                            action=jp.zeros((self._mjx_model.nu,), dtype=jp.float32),
                            n_substeps = self.n_substeps)


        penalty_collision_real_time = self._collision_check(data)


        center_pos = (data.site_xpos[self.planner.tcp_id_0]+data.site_xpos[self.planner.tcp_id_1])/2
        cost_g = jp.linalg.norm(center_pos - (data.xpos[self._mj_model.body(name='ball').id]-np.array([0, 0, 0.05])))
        

        current_cost_r_0 = quaternion_distance(data.xquat[self.planner.hande_id_0], jp.array([0.183, -0.683, -0.683, 0.183]))
        current_cost_r_1 = quaternion_distance(data.xquat[self.planner.hande_id_1], jp.array([0.183, -0.683, 0.683, -0.183]))

        current_cost_r = (current_cost_r_0 + current_cost_r_1)/2

        distances = jp.linalg.norm(data.site_xpos[self.planner.tcp_id_0] - data.site_xpos[self.planner.tcp_id_1])
        cost_dist = jp.abs(distances - 0.25)

        cost_zy = jp.linalg.norm(data.site_xpos[self.planner.tcp_id_0][1:3] - data.site_xpos[self.planner.tcp_id_1][1:3])
        

        cost_g_ball = jp.linalg.norm(data.xpos[self._mj_model.body(name='ball').id] - state.info['target_0'][:3])
        # cost_r_ball = jp.linalg.norm(data.xquat[self._mj_model.body(name='ball').id] - state.info['target_0'][3:])

        
        
        def pick_branch(_):
            target_reached = (
                (cost_g < self.grab_pos_thresh) &
                (cost_dist < self.grab_dist_thresh) &
                (current_cost_r_0 < self.grab_rot_thresh) &
                (current_cost_r_1 < self.grab_rot_thresh)
            )

            new_task = jp.where(target_reached, 1, task)

            return new_task, success, reason, penalty_z, target_reached


        def move_branch(_):
            target_reached = (cost_g_ball < 0.04) 

            success_new = jp.where(target_reached, 1, success)
            reason_new = jp.where(target_reached, REASON_OK, reason)

            fall_condition = cost_g > 0.2

            success_new = jp.where(fall_condition, 0, success_new)
            reason_new = jp.where(fall_condition, REASON_FALL, reason_new)
            
            penalty_z_new = jp.where(fall_condition, 5.0, 0.0)

            return task, success_new, reason_new, penalty_z_new, target_reached
        
        old_task = task 
        task, success, reason, penalty_z, target_reached = jax.lax.cond(
            task == 0,
            pick_branch,
            move_branch,
            operand=None
        )
        
        cost_c = best_cost_list[0] + best_cost_list[1]
        
        collision_fail = cost_c > 3000

        success = jp.where(collision_fail, 0, success)
        reason = jp.where(collision_fail, REASON_COLLISION, reason)
        penalty_col = jp.where(collision_fail, penalty_col + 0.5, penalty_col)
        
        out_of_bounds = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        
        failed = (reason == REASON_FALL) | (reason == REASON_COLLISION) | (reason == REASON_TIMEOUT)
        
        # done = (target_reached|failed|out_of_bounds)

        target_reached = jp.array(target_reached)
        failed = jp.array(failed)

        # done if target_reached and success 
        done = target_reached * old_task | failed | out_of_bounds



        state.info.update({
            'task': task,
            'success': success,
            'reason': reason,
            'cost_g_ball': cost_g_ball,
            'current_cost_g': cost_g,
            'current_cost_r': current_cost_r,
            'penalty_z': penalty_z,
            'penalty_r_s': penalty_r_s,
            'penalty_col': penalty_col,
            'penalty_collision_real_time': penalty_collision_real_time,
            'target_2': target_2,
            'ball_pick_init': state.info['ball_pick_init'],
            # 'time': time.time()
        })

        raw_rewards = self._get_reward(data, state.info)

        rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
        }

        potential = sum(rewards.values()) / sum(
            self._config.reward_config.scales.values()
        )

        # reward = jp.maximum(potential - state.info['prev_potential'], jp.zeros_like(potential))
        
        reward = potential - state.info['prev_potential']
        state.info['prev_potential'] = potential
        


        state.info['_steps'] += self._config.action_repeat
        state.info['_steps'] = jp.where(
            done | (state.info['_steps'] >= self._config.episode_length),
            0,
            state.info['_steps'],
        )

        state.metrics.update(**rewards, out_of_bounds=out_of_bounds.astype(float))

        obs = self._get_obs(data, state.info)
        return mjx_env.State(
            data, obs, reward, done.astype(float), state.metrics, state.info
        )
    
    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        
        # wc_pos=0.5
        # wc_rot=0.5

        cost_pos = info['cost_g_ball']
        cost_rot = info['cost_r_ball']
 


        # penalty_failure = info['penalty_col'] + info['penalty_z'] + info['penalty_r_s']

        # # penalty_success_weight = 50

        # cost = wc_pos*cost_pos + wc_rot*cost_rot  
        # cost += penalty_success_weight*(1-info['success'])+ penalty_failure + info['penalty_collision_real_time']
        
        # reward= -cost

        return {
        'robot_eef_pos': -info['current_cost_g'],
        'robot_eef_rot': -info['current_cost_r'],
        #box target pose are activated only in 'move' phase
        'ball_target_pos': -info['cost_g_ball']*info['task'],
        'ball_target_rot': -info['cost_r_ball']*info['task'],
        'ball_pick_success': info['task'],
        'ball_reach_success': info['success'] ,
        'collision_real_time': -info['penalty_collision_real_time'],
        'collision_plan_horizon': -info['penalty_col'],
        'fall_plan_horizon': -info['penalty_z'],
        'r_s_plan_horizon': -info['penalty_r_s']
        }

    


    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:


        def quaternion_inverse(q):
            return jp.concatenate([q[:1], -q[1:]]) / jp.dot(q, q)

        target_pos = data.mocap_pos[self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]] #target pos
        target_rot = data.mocap_quat[self._mj_model.body_mocapid[self._mj_model.body(name='target_0').id]] #target rot
        
        eef_0_pos = data.site_xpos[self.planner.tcp_id_0]
        eef_1_pos = data.site_xpos[self.planner.tcp_id_1]

        # eef_centre_pos = (eef_0_pos + eef_1_pos) / 2

        eef_0_quat = data.xquat[self.planner.hande_id_0]
        eef_1_quat = data.xquat[self.planner.hande_id_1]

        # eef_mean_quat = (eef_0_quat + eef_1_quat) 

        # eef_mean_quat = eef_mean_quat / jp.linalg.norm(eef_mean_quat)  # Normalize the mean quaternion

        joint_pos = data.qpos[self._joint_mask_pos]
        ball_pose = data.qpos[self._ball_qpos_idx:self._ball_qpos_idx+7]
        ball_pos = ball_pose[:3]
        ball_quat = ball_pose[3:]

        # ball_eef_rel_pos = ball_pos - eef_centre_pos
        
        # ball_eef_rel_quat = quaternion_multiply(ball_quat, quaternion_inverse(eef_mean_quat))

        target_ball_rel_pos = target_pos - ball_pos
        target_ball_rel_quat = quaternion_multiply(target_rot, quaternion_inverse(ball_quat))

        # --- Arm 0 ---
        eef_0_targ_quat = jp.array([0.183, -0.683, -0.683, 0.183])
        eef_1_targ_quat = jp.array([0.183, -0.683, 0.683, -0.183])
        eef0_ball_rel_pos = ball_pos - eef_0_pos
        eef0_ball_rel_quat = quaternion_multiply(
            eef_0_targ_quat,
            quaternion_inverse(eef_0_quat)
        )

        # --- Arm 1 ---
        eef1_ball_rel_pos = ball_pos - eef_1_pos
        eef1_ball_rel_quat = quaternion_multiply(
            eef_1_targ_quat,
            quaternion_inverse(eef_1_quat)
        )

        base_obs = jp.concatenate([
            joint_pos,
            # eef0_ball_rel_pos,
            # eef0_ball_rel_quat,
            # eef1_ball_rel_pos,
            # eef1_ball_rel_quat,
            ball_pose,
            data.qvel,
            # target_ball_rel_pos,
            # target_ball_rel_quat, 
            target_pos,
            target_rot,
            (info['_steps'].reshape((1,)) / self._config.episode_length).astype(
                float
            ),
        ])

        privileged_obs = jp.concatenate([
            base_obs,
            info['task'].reshape((1,)).astype(float),
            info['success'].reshape((1,)).astype(float),
            info['cost_g_ball'].reshape((1,)),
            info['cost_r_ball'].reshape((1,)),
            info['current_cost_g'].reshape((1,)),
            info['current_cost_r'].reshape((1,)),
            info['penalty_collision_real_time'].reshape((1,)),
        ])

        return {
        "state": base_obs,
        "privileged_state": privileged_obs
        }
    
    def _collision_check(self, data: mjx.Data):
        contact = data.contact

        geom1 = contact.geom1
        geom2 = contact.geom2
        dist = contact.dist

        # mask: robot involved
        robot_geom_ids = jp.array(list(self._robot_geom_ids))

        is_robot_1 = jp.isin(geom1, robot_geom_ids)
        is_robot_2 = jp.isin(geom2, robot_geom_ids)

        is_robot_contact = is_robot_1 | is_robot_2

        # penetration condition
        penetration = dist < -1e-6

        # valid collisions
        collisions = is_robot_contact & penetration

        num_penetration = jp.sum(collisions)

        penalty_collision_real_time = jp.where(
            num_penetration > 0,
            10.0 * num_penetration,
            0.0
        )

        return penalty_collision_real_time

    
    

    
