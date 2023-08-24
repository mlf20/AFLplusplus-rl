
from typing import NamedTuple
from torch import Tensor, tensor, as_tensor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
import numpy as np

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class MaskableRolloutBufferSamples(NamedTuple):
    observations: Tensor
    actions: Tensor
    old_values: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    returns: Tensor
    action_masks: Tensor


class MaskableDictRolloutBufferSamples(MaskableRolloutBufferSamples):
    observations: TensorDict
    actions: Tensor
    old_values: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    returns: Tensor
    action_masks: Tensor

class RolloutStorage(object):
    def __init__(self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        gae_lambda = 1,
        gamma = 0.99,):
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None

        self.action_masks = None
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.reset()

    def reset(self) -> None:
        mask_dims = self.action_space.n
        self.mask_dims = mask_dims
        self.action_masks = np.ones(
            (self.buffer_size, self.mask_dims))  # .to(self.device)

        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size,) + obs_input_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)

        self.pos = 0
        self.full = False


    def add(self, obs, action ,reward, episode_start, value, log_prob, action_masks = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        #if action_masks is not None:
        #    self.action_masks[self.pos] = action_masks.reshape(
        #        (1, self.mask_dims))
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            self.observations[key][self.pos] = obs_

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size = None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)


        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def to_torch(self, array, copy=True):
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return tensor(array).to(self.device)
        return as_tensor(array).to(self.device)

    def _get_samples(self, batch_inds):

        return MaskableDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (
                key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(
                self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )

    def compute_returns_and_advantage(self,
                        last_values,
                        dones):
        last_values =  last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
