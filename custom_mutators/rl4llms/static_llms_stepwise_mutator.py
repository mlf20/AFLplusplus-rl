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
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from gym.spaces.dict import Dict as DictSpace
from gym import spaces
from transformers import AutoTokenizer
from rl4lms.envs.text_generation.policy.causal_policy import CausalLMActorCriticPolicy
# from rl4lms.algorithms.common.maskable.buffers import MaskableDictRolloutBuffer
from rl4lms.envs.text_generation.kl_controllers import KLController
from rl4lms.envs.text_generation.observation import Observation
from rl4lms.data_pools.text_generation_pool import Sample
from rl4llmXafl_utils import add_to_buffer, linear_schedule, get_policy_kwargs, unpack_observations, TransitionInfo
import os.path as path
# from stable_baselines3.common.on_policy_algorithm.on_policy_algorithm import *
from stable_baselines3.common.utils import obs_as_tensor
import pickle as pkl
from agent.storage import RolloutStorage

# Agent implementation
AGENT = None
ROLLOUTS = None
ROLLOUT_INFO = None

# LLM Parameters
# LLM = pipeline(
from transformers import pipeline
file_dir = path.abspath(path.join(__file__, "../"))
TOKENIZER = AutoTokenizer.from_pretrained(file_dir + '/byte_gpt2')
MODEL_MAX_LENGTH = TOKENIZER.model_max_length
KLCONTROLLER = KLController(0.1, 0.1)
MAX_TEXT_LENGTH = 64
# Environment Parameters
STATE = []
MAX_ACTIONS = TOKENIZER.vocab_size
# OBSERVATION_SPACE   = gym.spaces.Box(0, 3000, (9216, ), dtype=int) # max seen in testing 8348 int so 2^13*1.125 = 9216  8348??
ACTION_SPACE = gym.spaces.Discrete(MAX_ACTIONS)
MAX_STEPS = 86  # Taken from maximum value of mutations that can be done by AFL. Though AFL takes a random number of steps (4,8,16,26,46,86) when doing so
STEP_COUNTER = 0
ROLLOUT_COUNTER = 0
GEN_OUTPUT = None
DONE = False
EPISODE_WISE_TRANSITIONS = []

# Agent Parameters
CLIP_PARAM = 0.2
PAST_STATE = {}
PPO_EPOCH = 2
NUM_MINI_BATCH = 1
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
LEARNING_RATE = 0.00000001
EPSILON = 1e-5
MAX_GRAD_NORM = 0.5
RECURRENT_POLICY = False
GAMMA = 0.99
BATCH_SIZE = 32
USE_GAE = False
GAE_LAMBDA = 0.95
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
PREVIOUS_STATE = None  # Unsure if we need this
TOTAL_STEP_COUNTER = 0
SAVE_FREQ = 500
SAVE_DIR = f'logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}_static_lm/'
TF_WRITER = None
PREV_VIRGIN_BITS = 0
PREV_TOTAL_CRASHES = 0
TOTAL_EXECUTIONS = 0
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output
    for i, out in enumerate(outputs):
        if not isinstance(out, torch.Tensor) and not isinstance(out, tuple):
            out = out.to_tuple()[0]

        if not isinstance(out, tuple):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                                   out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
        else:
            for j, o in enumerate(out):
                nan_mask = torch.isnan(o)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                                       o[nan_mask.nonzero()[:, 0].unique(sorted=True)])


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
    file_dir = path.abspath(path.join(__file__, "../"))

    AGENT = pipeline('text-generation', model=file_dir+ '/byte_gpt2')

    TF_WRITER = SummaryWriter(log_dir=SAVE_DIR)
    logging.basicConfig(level=logging.DEBUG, filename=f'{SAVE_DIR}/error.txt')
    print('INIT STARTED')


def deinit():
    global TF_WRITER
    global SAVE_DIR

    print('DEINIT STARTED')

    TF_WRITER.close()

    # torch.save(AGENT.actor_critic, os.path.join(save_path, 'final_model'+ ".pt"))


def havoc_mutation(buf, max_size):
    global STEP_COUNTER
    global TOTAL_STEP_COUNTER
    global MAX_STEPS
    global GEN_OUTPUT
    STEP_COUNTER += 1
    TOTAL_STEP_COUNTER += 1
    #str_buff = str(buf)[12:-2]
    #GEN_OUTPUT = AGENT(str_buff, max_length=86, num_return_sequences=1, return_tensors=True)
    byte_str = ''.join(
       TOKENIZER.decode(GEN_OUTPUT[0]['generated_token_ids'][0][:STEP_COUNTER]))
    byte_arr = str_to_byte(byte_str)[:max_size]
    #GEN_OUTPUT = GEN_OUTPUT['generated_text'].split(str_buff)[-1]
    # print(byte_str)
    #GEN_OUTPUT[:STEP_COUNTER]
    return byte_arr


def str_to_byte(byte_str):
    hex_list = byte_str.split('\\x')
    byte_array = b''
    for entry in hex_list:
        try:
            for backslashsplit in entry.split('\\'):
                try:
                    if backslashsplit != '':
                        byteentry = hex(int(backslashsplit, 16))[2:]
                        if len(byteentry) == 1:
                            byteentry = f'0{byteentry}'
                        byte_array += bytes.fromhex(byteentry)
                except:
                    byte_array += bytes(backslashsplit.encode('utf-8'))
        except:
            byte_array += bytes(entry.encode('utf-8'))
    if byte_array == b'':  # always return something
        byte_array = bytes(byte_str.encode('utf-8'))
    return bytearray(byte_array)


def fuzz(buf, add_buf, max_size):
    global AGENT
    global TOTAL_EXECUTIONS
    global GEN_OUTPUT
    TOTAL_EXECUTIONS += 1
    str_buff = str(buf)[12:-2]
    GEN_OUTPUT = AGENT(str_buff, max_length=86, num_return_sequences=1, return_tensors=True)
    #GEN_OUTPUT = GEN_OUTPUT[''].split(str_buff)[-1]

    byte_arr = buf[:max_size]
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
    pass

def fuzz_count(buff):
    return MAX_STEPS


def havoc_mutation_reward(total_crashes, virgin_bits):
    '''
    Called for each `havoc_mutation`. pass vars for computing the reward function
    '''
    pass
    # ROLLOUTS.after_update()


def introspection():
    string = ''
    return string


if __name__ == '__main__':
    init(3)
    # with open('/firmwire/logs/2023-09-04_12-48-12_PPO/model_state_dict_at_error.pkl', 'rb') as f:
    #    state_dict = pkl.load(f)
    # AGENT.load_from_dict(state_dict)
    for i in range(1):
        testbyte = bytearray([1, 2, 3, 4])
        havoc_mutation_reset()
        fuzz(testbyte, None, 200)
        for j in range(MAX_STEPS):
            testbyte = havoc_mutation(testbyte, 200)
            print(f"action: {testbyte}")
            print(f"step counter: {STEP_COUNTER}")
            torch.cuda.empty_cache()
            havoc_mutation_reward(0, 0)

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
