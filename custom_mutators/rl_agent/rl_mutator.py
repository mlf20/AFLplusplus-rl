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

# RL imports
from agent.ppo import PPO
from agent.storage import RolloutStorage
import gym
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter







# Agent implementation
AGENT = None
ROLLOUTS = None


# Environment Parameters
STATE = []
MAX_ACTIONS         = 26
OBSERVATION_SPACE   = gym.spaces.Box(0, 3000, (9216, ), dtype=int) # max seen in testing 8348 int so 2^13*1.125 = 9216  8348??
ACTION_SPACE        = gym.spaces.Discrete(MAX_ACTIONS)
MAX_STEPS           = 86 # Taken from maximum value of mutations that can be done by AFL. Though AFL takes a random number of steps (4,8,16,26,46,86) when doing so
STEP_COUNTER        = 0

# Agent Parameters
CLIP_PARAM          = 0.2
PPO_EPOCH           = 4
NUM_MINI_BATCH      = 1   # can play with this to increase it...
VALUE_LOSS_COEF     = 0.5
ENTROPY_COEF        = 0.01
LEARNING_RATE       = 7e-4
EPSILON             = 1e-5
MAX_GRAD_NORM       = 0.5
RECURRENT_POLICY    = True
GAMMA               = 0.99
USE_GAE             = False
GAE_LAMBDA          = 0.95
USE_PROPER_TIME_LIMITS = False

# Training loop params
PREVIOUS_STATE      = None # Unsure if we need this
TOTAL_STEP_COUNTER  = 0
SAVE_FREQ           = 500
SAVE_DIR            = f'logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}_PPO/'
TF_WRITER           = None
PREV_VIRGIN_BITS    = 0
PREV_TOTAL_CRASHES  = 0
TOTAL_EXECUTIONS    = 0
def init(seed):
    """
    Called once when AFLFuzz starts up. Used to seed our RNG.

    @type seed: int
    @param seed: A 32-bit random value
    """
    random.seed(seed)
    global AGENT
    global ROLLOUTS
    global SAVE_DIR
    global TF_WRITER


    AGENT = PPO(
            OBSERVATION_SPACE.shape,
            ACTION_SPACE,
            CLIP_PARAM,
            PPO_EPOCH,
            NUM_MINI_BATCH,
            VALUE_LOSS_COEF,
            ENTROPY_COEF,
            lr=LEARNING_RATE,
            eps=EPSILON,
            max_grad_norm=MAX_GRAD_NORM,
            base_kwargs={'recurrent': RECURRENT_POLICY})
    ROLLOUTS = RolloutStorage(MAX_STEPS, 1,
                              OBSERVATION_SPACE.shape, ACTION_SPACE,
                              AGENT.actor_critic.recurrent_hidden_state_size)

    TF_WRITER = SummaryWriter(log_dir=SAVE_DIR)
    print('INIT STARTED')


def deinit():
    global TF_WRITER
    global SAVE_DIR

    print('DEINIT STARTED')

    TF_WRITER.close()

    save_path = SAVE_DIR
    try:
        os.makedirs(save_path)
    except OSError:
        print(save_path)
    print(save_path)

    torch.save(AGENT.actor_critic, os.path.join(save_path, 'final_model'+ ".pt"))




def fuzz(buf, add_buf, max_size):
    """
    Called per fuzzing iteration.

    @type buf: bytearray
    @param buf: The buffer that should be mutated.

    @type add_buf: bytearray
    @param add_buf: A second buffer that can be used as mutation source.

    @type max_size: int
    @param max_size: Maximum size of the mutated output. The mutation must not
        produce data larger than max_size.

    @rtype: bytearray
    @return: A new bytearray containing the mutated data
    """
    ret = bytearray(100)
    print(GLOBAL_ARRAY)
    GLOBAL_ARRAY.append(1)
    #ret[:3] = random.choice(COMMANDS)
    ret = bytearray(random.randint(0, MAX_ACTIONS))

    return ret

def havoc_mutation(buf, max_size):
    '''
    Perform a single custom mutation on a given input.

    @type buf: bytearray
    @param buf: The buffer that should be mutated.

    @type max_size: int
    @param max_size: Maximum size of the mutated output. The mutation must not
        produce data larger than max_size.

    @rtype: bytearray
    @return: A new bytearray containing the mutated data
    '''

    return mutated_buf

def havoc_mutation_probability():
    '''
    Called for each `havoc_mutation`. Return the probability (in percentage)
    that `havoc_mutation` is called in havoc. Be default it is 6%.

    @rtype: int
    @return: The probability (0-100)
    '''
    global MAX_ACTIONS
    prob = random.randint(0, MAX_ACTIONS)
    return prob

def havoc_mutation_action(buf):
    '''
    Called for each `havoc_mutation`.
    For a given buffer return the mutation action to be taken by AFL.
    @type buf: bytearray
    @param buf: The buffer that should be mutated.

    @rtype: int
    @return: The action (0-26)
    '''
    global STEP_COUNTER
    global AGENT
    global OBSERVATION_SPACE
    global ROLLOUTS
    global TOTAL_STEP_COUNTER


    # Convert state to numpy fixed size
    int_list = [int(str(hex(x)), 16) for x in list(buf)]
    #padded_state = np.pad(int_list, (0,OBSERVATION_SPACE.shape[0] - len(int_list) % OBSERVATION_SPACE.shape[0]), 'constant')

    padded_state = torch.nn.functional.pad(input=torch.tensor(int_list), pad=(0,OBSERVATION_SPACE.shape[0] - len(int_list) % OBSERVATION_SPACE.shape[0]), mode='constant', value=0)    
    padded_state = padded_state[:OBSERVATION_SPACE.shape[0]]

    if STEP_COUNTER == 0:
         ROLLOUTS.obs[0].copy_(padded_state)
    # Get next action
    masks = torch.FloatTensor([1.0])
    bad_masks = torch.FloatTensor([1.0])
    reward =  torch.FloatTensor([0.0])

    if RECURRENT_POLICY:
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = AGENT.actor_critic.act(
                ROLLOUTS.obs[STEP_COUNTER].unsqueeze(0), ROLLOUTS.recurrent_hidden_states[STEP_COUNTER].unsqueeze(0),
                ROLLOUTS.masks[STEP_COUNTER])

        ROLLOUTS.insert(padded_state, recurrent_hidden_states.squeeze(0), action.squeeze(0),
                        action_log_prob, value.squeeze(0), reward, masks, bad_masks)
    else:
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = AGENT.actor_critic.act(
                ROLLOUTS.obs[STEP_COUNTER], ROLLOUTS.recurrent_hidden_states[STEP_COUNTER],
                ROLLOUTS.masks[STEP_COUNTER])

        ROLLOUTS.insert(padded_state, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)



    # State cleanup
    STEP_COUNTER += 1
    TOTAL_STEP_COUNTER += 1


    #action = random.randint(0, MAX_ACTIONS)
    #GLOBAL_ARRAY.append(action)
    #print(GLOBAL_ARRAY)
    return int(action)

def havoc_mutation_reset():
    '''
    Called for each `havoc_mutation`. Reset the python
    '''
    global STEP_COUNTER
    global TOTAL_STEP_COUNTER
    global SAVE_FREQ
    global SAVE_DIR
    global AGENT


    STEP_COUNTER = 0
    if TOTAL_STEP_COUNTER % SAVE_FREQ == 0:
        save_path = SAVE_DIR
        try:
            os.makedirs(save_path)
        except OSError:
            print(save_path)
        print(save_path)

        torch.save(AGENT.actor_critic, os.path.join(save_path, 'model'+ ".pt"))


def havoc_mutation_reward(total_crashes, virgin_bits):
    '''
    Called for each `havoc_mutation`. pass vars for computing the reward function
    '''
    global ROLLOUTS
    global AGENT
    global GAE_LAMBDA
    global GAMMA
    global USE_GAE
    global USE_PROPER_TIME_LIMITS
    global TF_WRITER
    global TOTAL_STEP_COUNTER
    global PREV_VIRGIN_BITS
    global PREV_TOTAL_CRASHES
    global TOTAL_EXECUTIONS
    global RECURRENT_POLICY
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

    print(reward, virgin_bits, PREV_VIRGIN_BITS, type(virgin_bits), type(PREV_VIRGIN_BITS))

    # Update the last transition with correct reward and done
    ROLLOUTS.rewards[-1] = torch.tensor([reward])
    ROLLOUTS.masks[-1] = torch.FloatTensor([0.0])

    if RECURRENT_POLICY:
        with torch.no_grad():
            next_value = AGENT.actor_critic.get_value(
                ROLLOUTS.obs[-1].unsqueeze(0), ROLLOUTS.recurrent_hidden_states[-1].unsqueeze(0),
                ROLLOUTS.masks[-1]).detach()

    else:
        with torch.no_grad():
            next_value = AGENT.actor_critic.get_value(
                ROLLOUTS.obs[-1], ROLLOUTS.recurrent_hidden_states[-1],
                ROLLOUTS.masks[-1]).detach()

    ROLLOUTS.compute_returns(next_value, USE_GAE, GAMMA,
                                 GAE_LAMBDA, USE_PROPER_TIME_LIMITS)

    value_loss, action_loss, dist_entropy = AGENT.update(ROLLOUTS)

    # Logging

    TOTAL_EXECUTIONS += 1
    TF_WRITER.add_scalar('episodic_return_steps', reward, TOTAL_STEP_COUNTER)
    TF_WRITER.add_scalar('bits_covered_steps', virgin_bits, TOTAL_STEP_COUNTER)
    TF_WRITER.add_scalar('value_loss_steps', value_loss, TOTAL_STEP_COUNTER)
    TF_WRITER.add_scalar('action_loss_steps', action_loss, TOTAL_STEP_COUNTER)
    TF_WRITER.add_scalar('crash_found_steps', total_crashes, TOTAL_STEP_COUNTER)
    TF_WRITER.add_scalar('episodic_return_exec', reward, TOTAL_EXECUTIONS)
    TF_WRITER.add_scalar('bits_covered_exec', virgin_bits, TOTAL_EXECUTIONS)
    TF_WRITER.add_scalar('value_loss_exec', value_loss, TOTAL_EXECUTIONS)
    TF_WRITER.add_scalar('action_loss_exec', action_loss, TOTAL_EXECUTIONS)
    TF_WRITER.add_scalar('crash_found_exec', total_crashes, TOTAL_EXECUTIONS)


    ROLLOUTS.after_update()




def introspection():
    string = ''
    return string

if __name__ == '__main__':
    init(3)
    for i in range(10):
        testbyte = bytearray([1, 2, 3, 4])
        action = havoc_mutation_action(testbyte)
        print(f"action: {action}")
        print(f"step counter: {STEP_COUNTER}")
        havoc_mutation_reward(0,0)
        havoc_mutation_reset()


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
