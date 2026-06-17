
import sys
import json

model_name = sys.argv[1]

pmax = True

if 'rew' in sys.argv[1]:
    pmax = False

model_file = open(f"{model_name}/model.drn", "r")
helpers_file = open(f"{model_name}/helpers.json", "r")

model_lines = model_file.readlines()
helpers_parsed = json.load(helpers_file)

result_file = open(f"{model_name}/model.dpomdp", "w")

# print(helpers_parsed['state labeling'][0])

agents = 2
discount = 0.98 * 0.98
states = None
actions = [[] for _ in range(agents)]
observations = [[] for _ in range(agents)]
start_state = None
sink_state = None

target_states = helpers_parsed['target states']

for i, line in enumerate(model_lines):
    if line.startswith("@nr_states"):
        states = int(model_lines[i+1]) + 1
        sink_state = states - 1
    if line.startswith("state"):
        state_info = line.split()
        state_index = int(state_info[1])
        if "init" in state_info:
            start_state = state_index
            break

for agent in range(agents):
    for choice_label in helpers_parsed['choice labeling']:
        if choice_label[agent] not in actions[agent]:
            actions[agent].append(choice_label[agent])
    else:
        actions[agent].sort()
    for obs_label in helpers_parsed['state labeling']:
        if obs_label[agent] not in observations[agent]:
            observations[agent].append(obs_label[agent])
    else:
        observations[agent].sort()

observation_max = None
for agent in range(agents):
    if observation_max is None:
        observation_max = max(observations[agent])
    else:
        if max(observations[agent]) > observation_max:
            observation_max = len(observations[agent])

# header
result_file.write(f"agents: {agents}\n")
result_file.write(f"discount: {discount}\n")
result_file.write(f"values: reward\n")

result_file.write("states:")
for state in range(states):
    result_file.write(f" s{state}")
else:
    result_file.write("\n")

result_file.write("actions:\n")
for agent in range(agents):
    result_file.write(f"a{actions[agent][0]}")
    for action in actions[agent][1:]:
        result_file.write(f" a{action}")
    else:
        result_file.write("\n")

result_file.write("observations:\n")
for agent in range(agents):
    result_file.write(f"o0")
    for obs in range(1,observation_max+1):
        result_file.write(f" o{obs}")
    else:
        result_file.write("\n")

result_file.write("start:")
for state in range(states):
    if state == start_state:
        result_file.write(f" 1.0")
    else:
        result_file.write(f" 0.0")
else:
    result_file.write("\n")

state_action_rewards = {}

# transition matrix
choice_index = 0
for i, line in enumerate(model_lines):
    if line.startswith("state"):
        state_info = line.split()
        state_index = int(state_info[1])

        if state_index not in target_states:

            j = i+1
            state_actions_count = 0
            while (not model_lines[j].startswith("state")) and (j < len(model_lines)):
                processed_line = model_lines[j].strip()
                if processed_line.startswith("action"):
                    state_actions_count += 1
                    some_action_info = processed_line.replace('[', '').replace(']','').split() # this only works if all actions have the same reward
                j += 1
                if j == len(model_lines):
                    break

            for a1 in actions[0]:
                for a2 in actions[1]:
                    if [a1, a2, 0] in helpers_parsed['choice labeling'][choice_index:choice_index+state_actions_count]:
                        state_choice_index = helpers_parsed['choice labeling'][choice_index:choice_index+state_actions_count].index([a1, a2, 0])
                        j = i+1
                        while (not model_lines[j].startswith("state")) and (j < len(model_lines)):
                            processed_line = model_lines[j].strip()
                            if processed_line.startswith("action") and state_choice_index == 0:
                                action_info = processed_line.replace('[', '').replace(']','').split()
                                # RMIN
                                if not pmax:
                                    state_action_rewards[f'{state_index},{a1},{a2}'] = float(action_info[2])*-1
                                state_choice_index -= 1
                            elif processed_line.startswith("action") and state_choice_index > 0:
                                state_choice_index -= 1
                            elif processed_line.startswith("action") and state_choice_index < 0:
                                break
                            elif state_choice_index == -1:
                                transition_line = processed_line.split(' : ')
                                result_file.write(f"T: {a1} {a2} : {state_index} : {transition_line[0]} : {transition_line[1]}\n")
                            else:
                                pass
                            j += 1
                            if j == len(model_lines):
                                break
                    else:
                        # PMAX
                        if pmax:
                            result_file.write(f"T: {a1} {a2} : {state_index} : {sink_state} : 1\n")
                        # RMIN
                        else:
                            result_file.write(f"T: {a1} {a2} : {state_index} : {state_index} : 1\n")
                            if float(some_action_info[2]) == 0:
                                some_rew = 100
                            else:
                                some_rew = float(some_action_info[2])

                            state_action_rewards[f'{state_index},{a1},{a2}'] = some_rew*-1

        else:
            j = i+1
            state_actions_count = 0
            while (not model_lines[j].startswith("state")) and (j < len(model_lines)):
                processed_line = model_lines[j].strip()
                if processed_line.startswith("action"):
                    state_actions_count += 1
                    action_info = processed_line.replace('[', '').replace(']','').split()
                j += 1
                if j == len(model_lines):
                    break

            for a1 in actions[0]:
                for a2 in actions[1]:
                    result_file.write(f"T: {a1} {a2} : {state_index} : {sink_state} : 1\n")
                    # RMIN
                    if not pmax:
                        state_action_rewards[f'{state_index},{a1},{a2}'] = float(action_info[2])*-1

        choice_index += state_actions_count

for a1 in actions[0]:
    for a2 in actions[1]:
        result_file.write(f"T: {a1} {a2} : {sink_state} : {sink_state} : 1\n")


# observations
for state in range(states):
    if state != sink_state:
        for a1 in actions[0]:
            for a2 in actions[1]:
                state_obs = helpers_parsed['state labeling']
                result_file.write(f"O: {a1} {a2} : {state} : {state_obs[state][0]} {state_obs[state][1]} : 1\n")
    else:
        for a1 in actions[0]:
            for a2 in actions[1]:
                result_file.write(f"O: {a1} {a2} : {state} : 0 0 : 1\n")


# rewards
# PMAX
if pmax:
    for state in range(states):
        for a1 in actions[0]:
            for a2 in actions[1]:
                if state in target_states:
                    result_file.write(f"R: {a1} {a2} : {state} : * : * : 1.0\n")
                else:
                    result_file.write(f"R: {a1} {a2} : {state} : * : * : 0.0\n")
# RMIN
else:
    for state in range(states):
        for a1 in actions[0]:
            for a2 in actions[1]:
                if state == sink_state:
                    result_file.write(f"R: {a1} {a2} : {state} : * : * : 0.0\n")
                elif state in target_states:
                    result_file.write(f"R: {a1} {a2} : {state} : * : * : 0.0\n")
                else:
                    result_file.write(f"R: {a1} {a2} : {state} : * : * : {state_action_rewards[f'{state},{a1},{a2}']}\n")
