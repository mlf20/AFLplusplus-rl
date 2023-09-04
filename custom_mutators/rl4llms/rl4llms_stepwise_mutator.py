#!/usr/bin/env python3
# encoding: utf-8
"""
Example Python Module for AFLFuzz

@author:     Christian Holler (:decoder)

@license:

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

@contact:    choller@mozilla.com
"""

# General Imports
import random
import os
import logging
# RL imports
import gym
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from gym.spaces.dict import Dict as DictSpace
from gym import spaces
from transformers import AutoTokenizer
from rl4lms.envs.text_generation.policy.causal_policy import CausalLMActorCriticPolicy
#from rl4lms.algorithms.common.maskable.buffers import MaskableDictRolloutBuffer
from rl4lms.envs.text_generation.kl_controllers import KLController
from rl4lms.envs.text_generation.observation import Observation
from rl4lms.data_pools.text_generation_pool import Sample
from rl4llmXafl_utils import add_to_buffer, linear_schedule, get_policy_kwargs, unpack_observations, TransitionInfo
import os.path as path
#from stable_baselines3.common.on_policy_algorithm.on_policy_algorithm import *
from stable_baselines3.common.utils import obs_as_tensor
import pickle as pkl
from agent.storage import RolloutStorage

# Agent implementation
AGENT = None
ROLLOUTS = None
ROLLOUT_INFO = None

# LLM Parameters
#LLM = pipeline(
file_dir = path.abspath(path.join(__file__, "../"))
TOKENIZER = AutoTokenizer.from_pretrained(file_dir+'/byte_gpt2')
MODEL_MAX_LENGTH = TOKENIZER.model_max_length
KLCONTROLLER = KLController(0.1, 0.1)
MAX_TEXT_LENGTH = 64
# Environment Parameters
STATE = []
MAX_ACTIONS         = TOKENIZER.vocab_size
#OBSERVATION_SPACE   = gym.spaces.Box(0, 3000, (9216, ), dtype=int) # max seen in testing 8348 int so 2^13*1.125 = 9216  8348??
ACTION_SPACE        = gym.spaces.Discrete(MAX_ACTIONS)
MAX_STEPS           = 86 # Taken from maximum value of mutations that can be done by AFL. Though AFL takes a random number of steps (4,8,16,26,46,86) when doing so
STEP_COUNTER        = 0
ROLLOUT_COUNTER     = 0
GEN_OUTPUT          = None
DONE                = False
EPISODE_WISE_TRANSITIONS = []

# Agent Parameters
CLIP_PARAM          = 0.2
PAST_STATE          = {}
PPO_EPOCH           = 4
NUM_MINI_BATCH      = 1
VALUE_LOSS_COEF     = 0.5
ENTROPY_COEF        = 0.01
LEARNING_RATE       = 0.00001
EPSILON             = 1e-5
MAX_GRAD_NORM       = 0.5
RECURRENT_POLICY    = False
GAMMA               = 0.99
BATCH_SIZE          = 32
USE_GAE             = False
GAE_LAMBDA          = 0.95
USE_PROPER_TIME_LIMITS = False
OBSERVATION_SPACE = DictSpace(
            {
                # we have to provide fixed sized inputs (padded) because sb3 support for DictObsersevation is limited
                # while creating rollout buffers, observations are concatenated for each key
                "prompt_or_input_encoded_pt": spaces.Box(
                    low=0, high=TOKENIZER.vocab_size, shape=(MAX_TEXT_LENGTH,)
                ),
                "prompt_or_input_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(MAX_TEXT_LENGTH,)
                ),
                "context_encoded_pt": spaces.Box(
                    low=0, high=TOKENIZER.vocab_size, shape=(MAX_STEPS,)
                ),
                "context_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(MAX_STEPS,)
                ),
                "input_encoded_pt": spaces.Box(
                    low=0,
                    high=TOKENIZER.vocab_size,
                    shape=(MAX_TEXT_LENGTH + MAX_STEPS,),
                ),
                "input_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(MAX_TEXT_LENGTH + MAX_STEPS,)
                ),
            }
        )
OBSERVATION = None
RAW_OBSERVATION = None
# Training loop params
PREVIOUS_STATE      = None # Unsure if we need this
TOTAL_STEP_COUNTER  = 0
SAVE_FREQ           = 500
SAVE_DIR            = f'logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}_PPO/'
TF_WRITER           = None
PREV_VIRGIN_BITS    = 0
PREV_TOTAL_CRASHES  = 0
TOTAL_EXECUTIONS    = 0
DEVICE              = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                               out[nan_mask.nonzero()[:, 0].unique(sorted=True)])




def init(seed):
    """
    Called once when AFLFuzz starts up. Used to seed our RNG.

    @type seed: int
    @param seed: A 32-bit random value
    """
    random.seed(seed)
    torch.cuda.empty_cache()
    global AGENT
    global ROLLOUTS
    global SAVE_DIR


    global TF_WRITER
    lr_schedule = linear_schedule(LEARNING_RATE)
    # needs to be rl4llms agent wrapped
    file_dir = path.abspath(path.join(__file__, "../"))
    AGENT =  CausalLMActorCriticPolicy(
            observation_space=OBSERVATION_SPACE,
            action_space=ACTION_SPACE,
            model_name= file_dir+ '/byte_gpt2',
            lr_schedule=lr_schedule,
            generation_kwargs={'do_sample': True,
                               'min_length': 1,
                               'max_new_tokens': 86}
        )
    
    for submodule in AGENT.modules():
        submodule.register_forward_hook(nan_hook)
    ROLLOUTS = RolloutStorage(
                MAX_STEPS,
                OBSERVATION_SPACE,
                ACTION_SPACE,
                device=DEVICE,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA
            )

    TF_WRITER = SummaryWriter(log_dir=SAVE_DIR)
    logging.basicConfig(level=logging.DEBUG, filename=f'{SAVE_DIR}/error.txt')
    print('INIT STARTED')


def deinit():
    global TF_WRITER
    global SAVE_DIR

    print('DEINIT STARTED')

    TF_WRITER.close()

    save_path =  path.abspath(path.join(__file__, "../"))+ f'/{SAVE_DIR}'
    try:
        os.makedirs(save_path)
    except OSError:
        print(save_path)
    print(save_path)

    AGENT.get_language_model().save_pretrained(save_path, 'model'+ ".pt")
    #torch.save(AGENT.actor_critic, os.path.join(save_path, 'final_model'+ ".pt"))

def havoc_mutation(buf, max_size):
    global GEN_OUTPUT
    global ROLLOUT_COUNTER
    global STEP_COUNTER
    global AGENT
    global GEN_OUTPUT
    global OBSERVATION
    global OBSERVATION_SPACE
    global RAW_OBSERVATION
    global DONE
    global ROLLOUTS
    global TOTAL_STEP_COUNTER
    global ROLLOUT_INFO
    global EPISODE_WISE_TRANSITIONS
    global TOTAL_EXECUTIONS
    global PAST_STATE
    if STEP_COUNTER == 0:
        episode_starts = np.ones((1,), dtype=bool)
        #episode_wise_transitions = []
        #ep_terminated = np.zeros((1,), dtype=bool)
        value_past_state = None
        ref_past_state = None
        policy_past_state = None
    else:
        episode_starts = np.zeros((1,), dtype=bool)
        value_past_state = PAST_STATE["value_past_state"]
        ref_past_state = PAST_STATE["ref_past_state"]
        policy_past_state = PAST_STATE["policy_past_state"]



    obs_tensor = OBSERVATION
    actions_tensor, _, action_mask, gen_tokens = GEN_OUTPUT[STEP_COUNTER -1]

    if DONE:
        TOTAL_EXECUTIONS += 1
        byte_str = ''.join(TOKENIZER.decode(
            [token for (_, _, _, token) in GEN_OUTPUT[:STEP_COUNTER]]))  # .split('\\x')).replace('\\', '')
        try:
            byte_str = bytearray.fromhex("".join(byte_str.split('\\x')).replace('\\', ''))[:max_size]
        except:
            byte_str = bytearray(byte_str.encode('utf-8'))[:max_size]
        return byte_str

    # if all episodes are done, just break and do not continue
    #if np.all(ep_terminated):
    #    break
    # evaluate actions with actions from rollout
    with torch.no_grad():
        #print(torch.cuda.memory_summary(abbreviated=False))
        torch.cuda.empty_cache()
        #obs_tensor = obs_as_tensor(obs.to_dict(), DEVICE)
        #for key, value in obs_tensor.items():
        #    obs_tensor[key] = value.unsqueeze(1)
        # get log probs (TBD: generalize this a bit)
        policy_kwargs = get_policy_kwargs(
            obs_tensor, actions_tensor, policy_past_state, action_mask
        )
        policy_outputs = AGENT.forward_policy(
            **policy_kwargs
        )
        raw_log_probs, log_probs, policy_past_state = (
            policy_outputs.raw_log_probs,
            policy_outputs.log_probs,
            policy_outputs.past_model_kwargs,
        )

        # sanity check
        assert torch.all(
            torch.isfinite(log_probs)
        ), "Infinite values in log probs"

        # sanity check
        assert torch.all(
            torch.isfinite(raw_log_probs)
        ), "Infinite values in log probs"

        # get values
        value_outputs = AGENT.forward_value(
            obs_tensor, value_past_state
        )
        values, value_past_state = (
            value_outputs.values,
            value_outputs.past_model_kwargs,
        )

        # get reference log probs
        ref_policy_outputs = (
            AGENT.get_log_probs_ref_model(
                obs_tensor, actions_tensor, ref_past_state
            )
        )
        ref_log_probs, ref_past_state = (
            ref_policy_outputs.log_probs,
            ref_policy_outputs.past_model_kwargs,
        )

        # sanity check
        assert torch.all(
            torch.isfinite(ref_log_probs)
        ), "Infinite values in log probs"

        PAST_STATE["value_past_state"] = value_past_state
        PAST_STATE["ref_past_state"] = ref_past_state
        PAST_STATE["policy_past_state"] = policy_past_state
        # compute KL rewards
        kl_div = raw_log_probs - ref_log_probs
        kl_rewards = -1 * KLCONTROLLER.kl_coeff * kl_div
        torch.cuda.empty_cache()

    actions = actions_tensor.cpu().numpy()
    rewards =  np.zeros((1,))
    dones = np.array([(actions == TOKENIZER.eos_token_id) or (STEP_COUNTER + 1 == MAX_STEPS)])
    total_rewards = rewards + kl_rewards.cpu().numpy()
    infos = [{}]

    # unpack individual observations
    unpacked_obs = unpack_observations(obs_tensor)
    # store episode wise transitions separately
    transtion = TransitionInfo(
        observation=unpacked_obs,
        action=actions,
        task_reward=rewards,
        total_reward=total_rewards,
        kl_div=kl_div.cpu().numpy(),
        episode_start=episode_starts,
        value=values.cpu(),
        log_prob=log_probs.cpu(),
        done=dones,
        ref_log_prob=ref_log_probs.cpu(),
        kl_reward=kl_rewards.cpu().numpy(),
        action_mask=action_mask.cpu().numpy()
        if action_mask is not None else None,
        info=infos,
    )

    EPISODE_WISE_TRANSITIONS.append(transtion)

    #episode_starts = np.zeros((1,), dtype=bool)
    obs = RAW_OBSERVATION.update(actions[0], TOKENIZER)
    OBSERVATION = obs_as_tensor({
       "prompt_or_input_encoded_pt": obs.prompt_or_input_encoded_pt.numpy(),
       "prompt_or_input_attention_mask_pt": obs.prompt_or_input_attention_mask_pt.numpy(),
       "context_encoded_pt": obs.context_encoded_pt.numpy(),
       "context_attention_mask_pt": obs.context_attention_mask_pt.numpy(),
       "input_encoded_pt": obs.input_encoded_pt.numpy(),
       "input_attention_mask_pt": obs.input_attention_mask_pt.numpy()
    }, DEVICE)

    if dones or STEP_COUNTER == MAX_STEPS:
        # now we flush all episode wise info to the 1-D buffer

        DONE = True
    #byte_arr = bytearray(gen_output.gen_texts[0].encode('utf-8'))
    #byte_arr=byte_arr[:max_size]

    #[TOKENIZER.decode(gen_tokens[0][:STEP_COUNTER+1], skip_special_tokens=True) for output in gen_tokens.tolist()]
    #return byte_arr
    STEP_COUNTER += 1
    TOTAL_STEP_COUNTER += 1
    byte_str = ''.join(TOKENIZER.decode([token for (_,_,_,token) in GEN_OUTPUT[:STEP_COUNTER]])) #.split('\\x')).replace('\\', '')
    try:
        byte_str = bytearray.fromhex("".join(byte_str.split('\\x')).replace('\\', ''))[:max_size]
    except:
        byte_str = bytearray(byte_str.encode('utf-8'))[:max_size]
    #print(byte_str)
    return byte_str


def fuzz(buf, add_buf, max_size):
    if add_buf:
        buf = buf[:random.choice(range(len(buf)))] + add_buf[:random.choice(range(len(buf)))]


    global STEP_COUNTER
    global AGENT
    global GEN_OUTPUT
    global OBSERVATION_SPACE
    global ROLLOUTS
    global OBSERVATION
    global RAW_OBSERVATION
    global TOTAL_STEP_COUNTER
    global ROLLOUT_INFO


    # Convert state to numpy fixed size
    #str_buff = "".join([str(hex(x)) for x in list(buf)])
    str_buff = str(buf)[12:-2]
    str_buff = Sample(1, str_buff, ['byte_string'])
    obs = Observation.init_from_sample(sample=str_buff, tokenizer=TOKENIZER, max_input_length=MAX_TEXT_LENGTH, max_context_length=MAX_STEPS, prompt_truncation_side='right')
    #print([tensor.shape for key, tensor in obs.to_dict().items()])
    #padded_state = np.pad(int_list, (0,OBSERVATION_SPACE.shape[0] - len(int_list) % OBSERVATION_SPACE.shape[0]), 'constant')
    #obs_tensor = obs_as_tensor(obs.to_dict(), DEVICE)
    OBSERVATION = obs_as_tensor({
        "prompt_or_input_encoded_pt": obs.prompt_or_input_encoded_pt.numpy(),
        "prompt_or_input_attention_mask_pt": obs.prompt_or_input_attention_mask_pt.numpy(),
        "context_encoded_pt": obs.context_encoded_pt.numpy(),
        "context_attention_mask_pt": obs.context_attention_mask_pt.numpy(),
        "input_encoded_pt": obs.input_encoded_pt.numpy(),
        "input_attention_mask_pt": obs.input_attention_mask_pt.numpy()
    }, DEVICE)
    RAW_OBSERVATION = obs
    gen_output = AGENT.generate(
        input_ids=OBSERVATION["input_encoded_pt"],
        attention_mask=OBSERVATION["input_attention_mask_pt"],
        tokenizer=TOKENIZER,
    )


    #episode_starts = np.ones((1,), dtype=bool)
    #episode_wise_transitions = []
    #ep_terminated = np.zeros((1,), dtype=bool)
    #value_past_state = None
    #ref_past_state = None
    #policy_past_state = None
    masks = (
        gen_output.action_masks
        if gen_output.action_masks is not None
        else [None] * len(gen_output.step_wise_logprobs)
    )
    GEN_OUTPUT =  list(zip(gen_output.step_wise_actions, gen_output.step_wise_logprobs, masks, gen_output.gen_tokens[0]))



    byte_arr = bytearray(gen_output.gen_texts[0].encode('utf-8'))

    byte_arr=byte_arr[:max_size]
    return byte_arr

def havoc_mutation_probability():
    '''
    Called for each `havoc_mutation`. Return the probability (in percentage)
    that `havoc_mutation` is called in havoc. Be default it is 6%.

    @rtype: int
    @return: The probability (0-100)
    '''


    return 100


def havoc_mutation_reset():
    '''
    Called for each `havoc_mutation`. Reset the python
    '''
    global STEP_COUNTER
    global TOTAL_STEP_COUNTER
    global SAVE_DIR
    global AGENT
    global ROLLOUT_INFO
    global GEN_OUTPUT
    global ROLLOUT_COUNTER
    global EPISODE_WISE_TRANSITIONS
    global OBSERVATION
    global RAW_OBSERVATION
    global DONE
    global PAST_STATE
    PAST_STATE = {}
    AGENT.set_training_mode(False)
    ROLLOUT_COUNTER     = 0
    GEN_OUTPUT          = None
    DONE                = False
    OBSERVATION         = None
    RAW_OBSERVATION         = None
    ROLLOUT_INFO = {
        "rollout_info/ep_rew": [],
        "rollout_info/kl_div_mean": [],
        "rollout_info/ep_lens": [],
        "rollout_info/ep_kl_rew": [],
        "rollout_info/log_prob": [],
        "rollout_info/ref_log_prob": [],
        "rollout_info/values": [],
    }
    save_path = path.abspath(path.join(__file__, "../"))+ f'/{SAVE_DIR}'

    STEP_COUNTER = 0
    EPISODE_WISE_TRANSITIONS = []
    if TOTAL_STEP_COUNTER % SAVE_FREQ == 0:
        AGENT.get_language_model().save_pretrained(save_path, 'model'+ ".pt")
        full_model = AGENT.get_state_dict()
        with open(save_path + '/model_state_dict.pkl', 'wb') as f:
            pkl.dump(full_model, f)
            
    
def fuzz_count(buff):
    return MAX_STEPS

def havoc_mutation_reward(total_crashes, virgin_bits):
    '''
    Called for each `havoc_mutation`. pass vars for computing the reward function
    '''
    global ROLLOUTS
    global AGENT
    global GAE_LAMBDA
    global GAMMA
    global CLIP_PARAM
    global USE_GAE
    global USE_PROPER_TIME_LIMITS
    global TF_WRITER
    global STEP_COUNTER
    global TOTAL_STEP_COUNTER
    global PREV_VIRGIN_BITS
    global PREV_TOTAL_CRASHES
    global TOTAL_EXECUTIONS
    global DONE
    global ROLLOUT_INFO
    global EPISODE_WISE_TRANSITIONS
    #virgin_bits = [int(str(hex(x)), 16) for x in list(virgin_bits)][0]
    #print(total_crashes)

    #total_crashes = [int(str(hex(x)), 16) for x in list(total_crashes)][0]
    #print(virgin_bits)
    #print(total_crashes)


    # Compute reward
    if total_crashes > PREV_TOTAL_CRASHES: # New crash found
        reward = 10
        #int_list = [int(str(hex(x)), 16) for x in list(buf)]
        PREV_VIRGIN_BITS = virgin_bits
        PREV_TOTAL_CRASHES = total_crashes

    elif int(virgin_bits) > int(PREV_VIRGIN_BITS): # New bits found compared to previous
        reward = 1
        #int_list = [int(str(hex(x)), 16) for x in list(buf)]
        PREV_VIRGIN_BITS = virgin_bits
    else:
        reward = -10

    print(reward, virgin_bits, PREV_VIRGIN_BITS, type(virgin_bits), type(PREV_VIRGIN_BITS), TOTAL_STEP_COUNTER)

    # Update the last transition with correct reward and done
    
    #ROLLOUTS.rewards[ROLLOUTS.pos - 1] = np.array([reward])

    EPISODE_WISE_TRANSITIONS[STEP_COUNTER - 1].task_reward = reward
    EPISODE_WISE_TRANSITIONS[STEP_COUNTER - 1].total_reward = reward + EPISODE_WISE_TRANSITIONS[STEP_COUNTER - 1].total_reward
    #ROLLOUTS.action_mask[-1] = torch.FloatTensor([0.0])


    #ROLLOUTS.action_masks[ROLLOUTS.pos - 1] = np.zeros((1,))
    if DONE:
        ROLLOUT_INFO, ROLLOUTS = add_to_buffer(
            ROLLOUTS, EPISODE_WISE_TRANSITIONS, ROLLOUT_INFO
        )
        next_values = (
                            ROLLOUTS[ROLLOUTS.pos].value
                            if (ROLLOUTS.pos) < MAX_STEPS
                            else torch.tensor([0.0])
                        )
        ROLLOUTS.compute_returns_and_advantage(last_values=next_values, dones=0.0)

        aggregated_rollout_info = {}
        for key, values in ROLLOUT_INFO.items():
            aggregated_rollout_info[key] = np.mean(values).item()
            aggregated_rollout_info[f"{key}_std"] = np.std(values).item()
        aggregated_rollout_info[
            "rollout_info/kl_coeff"
        ] = KLCONTROLLER.kl_coeff

        # adapt the KL coeff
        KLCONTROLLER.step(
            torch.tensor(aggregated_rollout_info["rollout_info/kl_div_mean"])
        )

        AGENT.set_training_mode(True)
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(PPO_EPOCH):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for batch_ix, rollout_data in enumerate(list(ROLLOUTS.get(BATCH_SIZE))):
                # self.verify_rollout_data(rollout_data)
                actions = rollout_data.actions
                if isinstance(ACTION_SPACE, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                try:
                    with torch.autograd.detect_anomaly():
                        evaluation_output: EvaluateActionsOutput = AGENT.evaluate_actions(
                            rollout_data.observations, actions)
                except:
                    full_model = AGENT.get_state_dict()
                    save_path = path.abspath(path.join(__file__, "../"))+ f'/{SAVE_DIR}'
                    with open(save_path + '/model_state_dict_at_error.pkl', 'wb') as f:
                        pkl.dump(full_model, f)
                    logging.debug(rollout_data.observations)
                    logging.debug(actions)
                    logging.debug(rollout_data.advantages)
                    logging.exception("Oops:")
                    exit()
                values, log_prob, entropy = evaluation_output.values, evaluation_output.log_prob, evaluation_output.entropy
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages

                advantages = (advantages - advantages.mean()
                                  ) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                # if batch_ix == 0 and epoch == 0:
                #     assert th.allclose(th.mean(ratio), th.tensor(
                #         1.0), atol=1e-3), "Cannot reconstruct probability distribution. Please check your policy network implementation"

                #     assert th.allclose(values, rollout_data.old_values, atol=1e-3), "Cannot reconstruct values. Please check your value network implementation"

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * \
                    torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > CLIP_PARAM).float()).item()
                clip_fractions.append(clip_fraction)

                values_pred = values

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + ENTROPY_COEF * entropy_loss + VALUE_LOSS_COEF * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)


                # Optimization step
                AGENT.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    AGENT.parameters(), 0.5)
                AGENT.optimizer.step()

            if not continue_training:
                break
        ROLLOUTS.reset()




    TOTAL_EXECUTIONS += 1
    if ROLLOUTS.full:
        TF_WRITER.add_scalar('value_loss_steps', value_loss, TOTAL_STEP_COUNTER)
        TF_WRITER.add_scalar('value_loss_exec', value_loss, TOTAL_EXECUTIONS)

    TF_WRITER.add_scalar('episodic_return_steps', reward, TOTAL_STEP_COUNTER)
    TF_WRITER.add_scalar('bits_covered_steps', virgin_bits, TOTAL_STEP_COUNTER)
    TF_WRITER.add_scalar('crash_found_steps', total_crashes, TOTAL_STEP_COUNTER)
    TF_WRITER.add_scalar('episodic_return_exec', reward, TOTAL_EXECUTIONS)
    TF_WRITER.add_scalar('bits_covered_exec', virgin_bits, TOTAL_EXECUTIONS)
    TF_WRITER.add_scalar('crash_found_exec', total_crashes, TOTAL_EXECUTIONS)


    #ROLLOUTS.after_update()




def introspection():
    string = ''
    return string

if __name__ == '__main__':
    init(3)

    for i in range(1):
        testbyte = bytearray([1, 2, 3, 4])
        havoc_mutation_reset()
        fuzz(testbyte, None, 200)
        for j in range(MAX_STEPS):
            testbyte = havoc_mutation(testbyte, 200)
            print(f"action: {testbyte}")
            print(f"step counter: {STEP_COUNTER}")
            torch.cuda.empty_cache()
            havoc_mutation_reward(0,0)


# actions (25 possible actions):
# flip single bit
# set interesting byte value
# set word (2 bytes) to interesting value, little endian.
# Set word to interesting value, big endian.
# Set dword to interesting value, big endian.
# Randomly subtract from byte.
# Randomly add to byte.
# Randomly subtract from word, little endian
# Randomly subtract from word, big endian
# Randomly add to word, little endian

# Randomly add to word, big endian
# Randomly subtract from dword, little endian
# Randomly subtract from dword, big endian
# Randomly add to dword, little endian
# Randomly add to dword, big endian.
# Just set a random byte to a random value. Because, why not. We use XOR with 1-255 to eliminate the possibility of a no-op.
# Clone bytes
# Insert a block of constant bytes (25%).
# Overwrite bytes with a randomly selected chunk bytes.
# Overwrite bytes with fixed bytes.

# Increase byte by 1.
# Decrease byte by 1.
# Flip byte.
# Switch bytes.
# Delete bytes
# Do nothing


# Uncomment and implement the following methods if you want to use a custom
# trimming algorithm. See also the documentation for a better API description.

# def init_trim(buf):
#     '''
#     Called per trimming iteration.
#
#     @type buf: bytearray
#     @param buf: The buffer that should be trimmed.
#
#     @rtype: int
#     @return: The maximum number of trimming steps.
#     '''
#     global ...
#
#     # Initialize global variables
#
#     # Figure out how many trimming steps are possible.
#     # If this is not possible for your trimming, you can
#     # return 1 instead and always return 0 in post_trim
#     # until you are done (then you return 1).
#
#     return steps
#
# def trim():
#     '''
#     Called per trimming iteration.
#
#     @rtype: bytearray
#     @return: A new bytearray containing the trimmed data.
#     '''
#     global ...
#
#     # Implement the actual trimming here
#
#     return bytearray(...)
#
# def post_trim(success):
#     '''
#     Called after each trimming operation.
#
#     @type success: bool
#     @param success: Indicates if the last trim operation was successful.
#
#     @rtype: int
#     @return: The next trim index (0 to max number of steps) where max
#              number of steps indicates the trimming is done.
#     '''
#     global ...
#
#     if not success:
#         # Restore last known successful input, determine next index
#     else:
#         # Just determine the next index, based on what was successfully
#         # removed in the last step
#
#     return next_index
#
# def post_process(buf):
#     '''
#     Called just before the execution to write the test case in the format
#     expected by the target
#
#     @type buf: bytearray
#     @param buf: The buffer containing the test case to be executed
#
#     @rtype: bytearray
#     @return: The buffer containing the test case after
#     '''
#     return buf
#

# def queue_get(filename):
#     '''
#     Called at the beginning of each fuzz iteration to determine whether the
#     test case should be fuzzed
#
#     @type filename: str
#     @param filename: File name of the test case in the current queue entry
#
#     @rtype: bool
#     @return: Return True if the custom mutator decides to fuzz the test case,
#         and False otherwise
#     '''
#     return True
#
# def queue_new_entry(filename_new_queue, filename_orig_queue):
#     '''
#     Called after adding a new test case to the queue
#
#     @type filename_new_queue: str
#     @param filename_new_queue: File name of the new queue entry
#
#     @type filename_orig_queue: str
#     @param filename_orig_queue: File name of the original queue entry
#     '''
#     pass
