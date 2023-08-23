from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from dataclasses import dataclass
from stable_baselines3.common.type_aliases import TensorDict
import numpy as np
import torch
from typing import Dict, Any

def add_to_buffer(rollout_buffer, episode_wise_transitions, rollout_info
):

    advantages_computed = False
    for ep_ix, transitions in enumerate(episode_wise_transitions):
        ep_length = len(transitions)
        total_reward = 0.0
        total_kl_reward = 0.0
        for transition_ix, transition in enumerate(transitions):
            total_reward += transition.task_reward
            total_kl_reward += transition.kl_reward
            rollout_info["rollout_info/kl_div_mean"].append(transition.kl_div)
            rollout_info["rollout_info/log_prob"].append(transition.log_prob)
            rollout_info["rollout_info/ref_log_prob"].append(
                transition.ref_log_prob
            )
            rollout_info["rollout_info/values"].append(transition.value.numpy())

            if not rollout_buffer.full:
                rollout_buffer.add(
                    transition.observation,
                    transition.action,
                    transition.total_reward,
                    transition.episode_start,
                    transition.value,
                    transition.log_prob,
                    action_masks=transition.action_mask,
                )
            #if rollout_buffer.full:
            #    print('what is happening')

            # if the buffer is full, compute advantages
            #if rollout_buffer.full and not advantages_computed:

            #    # we fetch the last value for the last time step
            #    # values come from the next transitions's values
            #    next_values = (
            #        transitions[transition_ix + 1].value
            #        if (transition_ix + 1) < ep_length
            #        else torch.tensor([0.0])
            #    )

            #    rollout_buffer.compute_returns_and_advantage(
            #        last_values=next_values, dones=transition.done
            #    )
            #    advantages_computed = True

        rollout_info["rollout_info/ep_rew"].append(total_reward)
        rollout_info["rollout_info/ep_lens"].append(ep_length)
        rollout_info["rollout_info/ep_kl_rew"].append(total_kl_reward)
    return rollout_info, rollout_buffer




def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def get_policy_kwargs(
    obs,
    action,
    past_state,
    action_mask,
):

    policy_kwargs = {
        "obs": obs,
        "actions": action,
        "past_model_kwargs": past_state,
    }
    if action_mask is not None:
        policy_kwargs["action_masks"] = action_mask
    return policy_kwargs


def unpack_observations(obs_tensor, n_envs: int):
    """
    Unpacks vectorized dict observations into separate dict observations
    """
    unpacked_obs = []
    keys = obs_tensor.keys()
    for env_ix in range(n_envs):
        obs_dict = {}
        for key in keys:
            obs_dict[key] = obs_tensor[key].reshape(1, -1).cpu()
        unpacked_obs.append(obs_dict)
    return unpacked_obs

@dataclass
class TransitionInfo:
    observation: TensorDict
    action: np.ndarray
    task_reward: np.ndarray
    total_reward: np.ndarray
    kl_div: np.ndarray
    episode_start: np.ndarray
    value: torch.Tensor
    log_prob: torch.Tensor
    done: np.ndarray
    ref_log_prob: torch.Tensor
    kl_reward: np.ndarray
    action_mask: np.ndarray
    info: Dict[str, Any]


