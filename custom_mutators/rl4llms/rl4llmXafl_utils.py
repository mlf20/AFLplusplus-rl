from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction






def add_to_buffer(
    self, rollout_buffer, episode_wise_transitions, rollout_info
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

            # if the buffer is full, compute advantages
            if rollout_buffer.full and not advantages_computed:

                # normalize the rewards
                if self._norm_reward:
                    mean = rollout_buffer.rewards.mean()
                    std = rollout_buffer.rewards.std()
                    rollout_buffer.rewards = (rollout_buffer.rewards - mean) / (
                        std + 1e-8
                    )

                # we fetch the last value for the last time step
                # values come from the next transitions's values
                next_values = (
                    transitions[transition_ix + 1].value
                    if (transition_ix + 1) < ep_length
                    else torch.tensor([0.0])
                )

                rollout_buffer.compute_returns_and_advantage(
                    last_values=next_values, dones=transition.done
                )
                advantages_computed = True

        rollout_info["rollout_info/ep_rew"].append(total_reward)
        rollout_info["rollout_info/ep_lens"].append(ep_length)
        rollout_info["rollout_info/ep_kl_rew"].append(total_kl_reward)
    return rollout_info




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