import stormpy
import payntbind

import paynt.family.family
import paynt.verification.property
import paynt.parser.jani

from itertools import product
from math import prod

import os
import re
import json
import logging
import zipfile

logger = logging.getLogger(__name__)

class PrismHyperParser:

    def __init__(self):
        self.sched_quant_dict = {}
        self.state_quant_dict = {}
        self.state_quant_restrictions = {}

        # a list of dictionaries. Dictionary i maps states of replica i to their holes.
        self.state_to_hole_indexes = []

        # each choice in the self-composition, or cross-product,
        # corresponds to a tuple of agents' actions in the original MDP
        # needed only for exporting the model to Inf-JESP.
        self.choice_to_action_tuple = []

        # each choice in the self-composition, or cross-product,
        # maps some holes to fix their option.
        self.choice_to_hole_options = []

        # each state in the self-composition, or cross-product
        # corresponds to a tuple of states of the original model
        self.product_id_to_state_tuple = []

        # parsed lines of the property
        self.lines = []

        self.composed_model = None
        self.target_sets = {}

        self.specification = None
        # parsed optimality properties
        self.optimality_property = None

    def load_specification(self, path):
        if not os.path.isfile(path):
            raise ValueError(f"the properties file {path} does not exist")
        with open(path) as file:
            for line in file:
                line = line.replace("\n", "")
                if not line or line == "" or line.startswith("//"):
                    continue
                self.lines.append(line)

    # the first line of the specifications contains the scheduler quantifications
    def parse_scheduler_quants(self):
        sched_re = re.compile(r'^(ES|AS)\s(\S+)(\s?)(.*$)')

        line = self.lines.pop(0)
        match = sched_re.search(line)
        if match is None:
            raise Exception(f"scheduler quantification is wrongly formatted: {line}!")

        while True:
            sched_quant = match.group(1)
            sched_name = match.group(2)

            if sched_name in list(self.sched_quant_dict.keys()):
                raise Exception("two scheduler variables cannot have the same name")

            # for now, we support only straighforward synthesis specifications
            # hence currently the dictionary values are never retrieved
            if sched_quant == "AS":
                # TODO: implement encoding of AS quantifications
                raise NotImplementedError

            # add this scheduler quantifier to the dictionary
            self.sched_quant_dict[sched_name] = sched_quant

            # end of line
            if match.group(4) == "":
                break

            # move to next scheduler
            line = match.group(4)
            match = sched_re.search(line)
            if match is None:
                raise Exception(f"scheduler quantification is wrongly formatted: {line}!")

    # the second line of the specifications contains state quantifications
    def parse_state_quants(self):
        # parse state quantifiers
        state_re = re.compile(r'^(E|A)\s(\S+)\((\S+)\)(\s?)(.*$)')
        line = self.lines.pop(0)
        match = state_re.search(line)
        if match is None:
            raise Exception(f"state quantifications are wrongly formatted: {line}")

        existential_quantifier = False
        while True:
            state_quant = match.group(1)
            state_var = match.group(2)
            sched_var = match.group(3)

            # every scheduler variable must be quantified
            if sched_var not in self.sched_quant_dict:
                raise Exception(f"a scheduler variable occurs free in the state quantifications: {line}.")

            # the implementation of HyperPaynt supports only specifications in conjunctive normal form
            if existential_quantifier and state_quant == "A":
                raise Exception(f"this nesting E*A* of state quantifications is not allowed: please use conjunctions of disjuctions (A*E*).")

            if state_var in self.state_quant_dict:
                raise Exception(f"two state variables cannot have the same name: {state_var}")

            if state_quant == "E":
                existential_quantifier = True

            # add the state quantifier to the dictionary
            self.state_quant_dict[state_var] = (state_quant, sched_var)

            # end of state quantifiers
            if match.group(5) == "":
                break

            # move to next state quantifier
            line = match.group(5)
            match = state_re.search(line)
            if match is None:
                raise Exception(f"this part of the input formula is wrongly formatted: {line}.\nExpecting some state quantifier.")

    # the third line of the specifications contains restrictions to state variables
    def parse_restrictions(self):
        restriction_re = re.compile(r'Restrict\s(\S+)\s(\S+)(\s?)(.*$)')
        line = self.lines.pop(0)
        match = restriction_re.search(line)
        if match is None:
            # no restriction specified
            self.lines = [line] + self.lines
            return

        while True:
            state_var = match.group(1)
            restriction_label = match.group(2)
            if state_var not in self.state_quant_dict:
                raise Exception(f"Trying to restrict a variable not in scope: {state_var}.")
            # list on a dictionary returns the list of elements in insertion order.
            replica_number = list(self.state_quant_dict).index(state_var)
            if replica_number in self.state_quant_restrictions:
                raise Exception(f"Trying to restrict two times the same variable: {state_var}.")

            # storing this restriction
            self.state_quant_restrictions[replica_number] = restriction_label

            if match.group(4) == "":
                # no other restrictions found
                return

            # move to next restriction
            line = match.group(4)
            match = restriction_re.search(line)
            if match is None:
                raise Exception(f"Wrong formatted restrictions: {line}.")

    def parse_property(self, line):
        '''
        Parse a line containing a single PCTL property.
        @return the property or None if no property was detected
        '''
        props = stormpy.parse_properties_without_context(line)
        if len(props) == 0:
            return None
        if len(props) > 1:
            logger.warning("multiple properties detected on one line, dropping all but the first one")
        return props[0]

    def parse_specification(self, relative_error, discount_factor):
        '''
        Expecting one property per line. The line may be terminated with a semicolon.
        Empty lines or comments are allowed.
        '''

        mdp_spec = re.compile(r'^\s*(min|max)\s+(.*)$')

        properties = []

        for line in self.lines:
            minmax = None
            match = mdp_spec.search(line)
            if match is not None:
                minmax = match.group(1)
                line = match.group(2)
            prop = self.parse_property(line)
            if prop is None:
                continue

            rf = prop.raw_formula
            assert rf.has_bound != rf.has_optimality_type, \
                "optimizing formula contains a bound or a comparison formula does not"
            if minmax is None:
                if rf.has_bound:
                    prop = paynt.verification.property.Property(prop, discount_factor)
                else:
                    prop = paynt.verification.property.OptimalityProperty(prop, discount_factor, relative_error)
            else:
                assert not rf.has_bound, "double-optimality objective cannot contain a bound"
                dminimizing = (minmax == "min")
                prop = paynt.verification.property.DoubleOptimalityProperty(prop, dminimizing, discount_factor,
                                                                            relative_error)
            properties.append(prop)

        self.specification = paynt.verification.property.Specification(properties)
        logger.info(f"Found the following specification: {self.specification}")
        assert not self.specification.has_double_optimality, "did not expect double-optimality property"

    def generate_partially_observable_family(self, single_model, nr_replicas):
        family = paynt.family.family.Family()
        self.state_to_hole_indexes = [{} for _ in range(nr_replicas)]
        assert single_model.has_observation_valuations, "Observations are not named."
        assert single_model.model_type == stormpy.ModelType.POMDP
        contains_stop = single_model.labeling.contains_label('stop')

        holes_dict = {}  # hole name -> hole index

        for state in single_model.states:
            obs = single_model.get_observation(state.id)
            obs_name = single_model.observation_valuations.get_string(obs)
            state_name = single_model.state_valuations.get_string(state.id)

            num_actions = single_model.get_nr_available_actions(state.id)
            option_labels = []
            if num_actions > 1:
                if contains_stop:
                    assert not single_model.labeling.has_state_label('stop', state)
                for offset in range(num_actions):
                    choice = single_model.get_choice_index(state, offset)
                    label = single_model.choice_labeling.get_labels_of_choice(choice)
                    option_labels.append(str(label))
                state_holes = {}
                for sched_name in self.sched_quant_dict.keys():
                    hole_name = f"{obs_name}({sched_name})"
                    hole_index = holes_dict.get(hole_name, None)
                    if hole_index is None:
                        hole_index = family.num_holes
                        family.add_hole(hole_name, option_labels)
                        holes_dict[hole_name] = hole_index
                    else:
                        assert family.hole_options_strings(hole_index) == option_labels, \
                            f"The same observation has different available actions in different states: " \
                            f"{family.hole_options_strings(hole_index)}. \n" \
                            f"In current state {state_name} we have available: {option_labels}"
                    state_holes[sched_name] = hole_index
                for index, (_, sched_name) in enumerate(self.state_quant_dict.values()):
                    self.state_to_hole_indexes[index][state.id] = state_holes[sched_name]

        return family

    def generate_locally_fully_observable_family(self, single_model, nr_replicas):
        family = paynt.family.family.Family()
        self.state_to_hole_indexes = [{} for _ in range(nr_replicas)]
        contains_stop = single_model.labeling.contains_label('stop')
        for state in single_model.states:
            state_name = single_model.state_valuations.get_string(state.id)
            num_actions = single_model.get_nr_available_actions(state.id)
            option_labels = []
            if num_actions > 1:
                if contains_stop:
                    assert not single_model.labeling.has_state_label('stop', state)
                for offset in range(num_actions):
                    choice = single_model.get_choice_index(state, offset)
                    label = single_model.choice_labeling.get_labels_of_choice(choice)
                    option_labels.append(str(label))
                state_holes = {}
                for sched_name in self.sched_quant_dict.keys():
                    hole_name = f"{state_name}({sched_name})"
                    hole_index = family.num_holes
                    family.add_hole(hole_name, option_labels)
                    state_holes[sched_name] = hole_index
                for index, (_, sched_name) in enumerate(self.state_quant_dict.values()):
                    self.state_to_hole_indexes[index][state.id] = state_holes[sched_name]
        return family

    def build_self_composition(self, single_model, want_to_export, nr_replicas):
        contains_stop = single_model.labeling.contains_label('stop')
        logger.warning(f"Assuming \"stop\" is a special label to mark deadlock states, "
                       f"and collapsing all deadlock states in a single one. "
                       f"Please change it to another label if this is not the intended meaning.")

        # generate the state set of the self-composition
        # preprocess to avoid thousands of sink states, sc = self-composition
        nr_states = single_model.nr_states
        state_permutations = list(product(range(nr_states), repeat=nr_replicas))
        deadlock_state = None  # the only one we are allowing
        sc_state_to_sc_index = {}
        fresh_id = 0
        for states_tuple in state_permutations:
            states = list(map(lambda id: single_model.states[id], states_tuple))
            is_deadlock = contains_stop and any(
                map(lambda state: single_model.labeling.has_state_label('stop', state), states))
            if not is_deadlock or deadlock_state is None:
                # register this state
                sc_state_to_sc_index[states_tuple] = fresh_id
                if is_deadlock:
                    deadlock_state = fresh_id
                fresh_id += 1

        # generate the transition matrix of the self-composition
        logger.info(f"generating the transition system of the self-composition")
        builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                              has_custom_row_grouping=True, row_groups=0)
        sc_row_counter = 0
        for states_tuple in state_permutations:
            # generate the state in the cross product associated with this tuple
            states = list(map(lambda id: single_model.states[id], states_tuple))
            sc_state = sc_state_to_sc_index.get(states_tuple, None)
            if sc_state is not None:
                self.product_id_to_state_tuple.append(states_tuple)
                if sc_state == deadlock_state:  # the deadlock state goes to itself
                    builder.new_row_group(sc_row_counter)
                    self.choice_to_hole_options.append([])
                    if want_to_export:
                        self.choice_to_action_tuple.append(tuple([0 for _ in states_tuple]))
                    builder.add_next_value(sc_row_counter, deadlock_state, 1)
                    sc_row_counter += 1

                else:
                    actions_lists = list(map(lambda id: range(single_model.get_nr_available_actions(id)), states_tuple))
                    actions_tuples = product(*actions_lists) # all the available tuples of actions of the tuple of states
                    builder.new_row_group(sc_row_counter)
                    for actions_tuple in actions_tuples:
                        self.choice_to_hole_options.append([])
                        self.choice_to_action_tuple.append(actions_tuple)
                        transitions_lists = [states[index].actions[id].transitions for index, id in
                                             enumerate(actions_tuple)]
                        for (index, action_id) in enumerate(actions_tuple):
                            hole_id = self.state_to_hole_indexes[index].get(states_tuple[index], None)
                            if hole_id is not None:  # if it is None, then there is no hole associated with this state
                                self.choice_to_hole_options[sc_row_counter].append((hole_id, action_id))
                        transitions_tuples = product(*transitions_lists)
                        deadlock_prob = 0
                        for transitions_tuple in transitions_tuples:
                            destination_tuple = tuple(map(lambda transition: transition.column, transitions_tuple))
                            destination = sc_state_to_sc_index.get(destination_tuple, None)
                            value = prod(map(lambda transition: transition.value(), transitions_tuple))
                            if destination is None or destination == deadlock_state:
                                deadlock_prob += value
                            else:
                                builder.add_next_value(sc_row_counter, destination, value)
                        if deadlock_prob > 0:
                            builder.add_next_value(sc_row_counter, deadlock_state, deadlock_prob)
                        sc_row_counter += 1

        # generate the labelings of the self-composition
        logger.info("Generating the labels of the self-composition")
        nr_sc_states = fresh_id
        sc_state_labeling = stormpy.storage.StateLabeling(nr_sc_states)
        for label in single_model.labeling.get_labels():
            assert not label == 'soi', "Soi is a reserved label. Please change it to another label."
            if label == "deadlock":
                pass
            elif label == 'stop':
                for index, state_variable in enumerate(self.state_quant_dict):
                    sc_label = label + state_variable
                    sc_state_labeling.add_label(sc_label)
                    sc_state_labeling.set_states(sc_label,
                                                    stormpy.BitVector(nr_sc_states, [deadlock_state]))

            # reserved label for initial states
            elif label == 'init':
                states = list(single_model.labeling.get_states(label))
                affected_states_tuples = list(product(states, repeat=nr_replicas))
                filtered_state_tuples = list(filter(
                    lambda tup: all(map(lambda t: t[1] in single_model.labeling.get_labels_of_state(tup[t[0]]),
                                        list(self.state_quant_restrictions.items()))), affected_states_tuples))
                affected_states = list(
                    map(lambda tup: sc_state_to_sc_index[tup], filtered_state_tuples))
                sc_state_labeling.add_label(label)
                sc_state_labeling.set_states(label,
                                                stormpy.BitVector(nr_sc_states, affected_states))
            else:
                affected_states = list(single_model.labeling.get_states(label))
                for index, state_variable in enumerate(self.state_quant_dict):
                    sc_label = label + state_variable
                    sc_state_labeling.add_label(sc_label)
                    sc_state_labeling.set_states(sc_label,
                                                    stormpy.BitVector(nr_sc_states,
                                                                      [sc_state_to_sc_index[tup] for tup in
                                                                       sc_state_to_sc_index if
                                                                       tup[index] in affected_states and
                                                                       sc_state_to_sc_index[tup] != deadlock_state]))

        sc_transition_matrix = builder.build(overridden_column_count=nr_sc_states)
        components = stormpy.SparseModelComponents(transition_matrix=sc_transition_matrix,
                                                   state_labeling=sc_state_labeling,
                                                   rate_transitions=False)

        # build the self-composition
        self.composed_model = stormpy.storage.SparseMdp(components)
        logger.info(f"Number of states of the self-composition: {self.composed_model.nr_states}")

    # constructing the cross-product(s) with the automata for the formulae
    def build_cross_product(self, single_model, want_to_export, single_property):
        for index, property in enumerate(self.specification.stormpy_properties()):
            if (not property.raw_formula.subformula.is_eventually_formula) or want_to_export:
                formula = property.raw_formula

                logger.info(f"Generating explicit cross-product for formula: {formula}")
                if formula.is_reward_operator:
                    logger.info(f"Refactoring current formula to give it to STORM to build the DRA cross-product.")
                    rf = str(formula)
                    formula_re = re.compile(r'^(.*)\[(.*)\]')
                    match = formula_re.search(rf)
                    formula = stormpy.parse_properties_without_context(f"Pmax=?[{match.group(2)}]\n")[0]
                    logger.info(f"Using fictitious formula: {formula}")

                # generate the cross-product model
                product_rep = stormpy.build_product_model(self.composed_model, formula)
                cross_product = product_rep.product_model
                p_index_to_p_state = product_rep.product_index_to_product_state
                # add labels
                for label in self.composed_model.labeling.get_labels():
                    if not (label == 'soi'):
                        cross_product.labeling.add_label(label)
                logger.info(f"Number of states of the cross-product: {cross_product.nr_states}")

                # resetting the initial state
                assert cross_product.labeling.contains_label("soi")
                initial_states = list(cross_product.labeling.get_states("soi"))
                assert len(initial_states) == 1
                cross_product.labeling.add_label_to_state("init", initial_states[0])

                # generate the family and the choice_to_hole_option mapping
                logger.info("Regenerating choice-to-hole-options to adapt to cross-product - family does not change")
                new_choice_to_hole_options = []
                new_choice_to_actions_tuple = []  # to export to Inf-JESP
                new_product_id_to_state_tuple = []
                new_target_sets = {}

                for state in cross_product.states:
                    num_actions = cross_product.get_nr_available_actions(state.id)
                    (mdp_state, sA) = p_index_to_p_state[state.id]

                    # mark this tuple as target for previous formulae
                    if self.target_sets.get(mdp_state):
                        new_target_sets[state.id] = self.target_sets[mdp_state]

                    # readding labels for next cross products, each cross-product needs its own.
                    if not single_property:
                        labels = self.composed_model.labeling.get_labels_of_state(mdp_state)
                        for label in labels:
                            if not (label == 'init'):
                                cross_product.labeling.add_label_to_state(label, state.id)

                    old_state_tuple = self.product_id_to_state_tuple[mdp_state]
                    new_product_id_to_state_tuple.append(old_state_tuple + (sA,))

                    for offset in range(num_actions):
                        old_choice = self.composed_model.get_choice_index(mdp_state, offset)
                        old_hole_options = self.choice_to_hole_options[old_choice]
                        if want_to_export:
                            old_actions_tuple = self.choice_to_action_tuple[old_choice]
                            if num_actions > 1:  # this state has to be mapped to a hole
                                assert old_hole_options
                                new_choice_to_actions_tuple.append(old_actions_tuple + (0,))
                            else:
                                assert not old_hole_options
                                assert all([action == 0 for action in old_actions_tuple])
                                new_choice_to_actions_tuple.append(old_actions_tuple + (0,))
                        new_choice_to_hole_options.append(old_hole_options)

                # updating various information
                # adding target states of this property
                assert list(product_rep.accepting_states)
                for accepting_state in list(product_rep.accepting_states):
                    # the state is accepting also for this formula.
                    new_target_sets[accepting_state] = new_target_sets.get(accepting_state, []) + [index]
                self.target_sets = new_target_sets

                self.composed_model = cross_product
                self.choice_to_hole_options = new_choice_to_hole_options
                self.product_id_to_state_tuple = new_product_id_to_state_tuple
                if want_to_export:
                    self.choice_to_action_tuple = new_choice_to_actions_tuple

        # generate the reward models of the overall cross-product
        reward_models = {}
        for name, reward_model in single_model.reward_models.items():
            logger.info(f"Generating the reward model of the self-composition for reward model {name}")
            assert reward_model.has_state_rewards
            assert not reward_model.has_state_action_rewards
            assert not reward_model.has_transition_rewards
            for index, state_variable in enumerate(self.state_quant_dict):
                state_reward = [reward_model.get_state_reward(state_tuple[index]) for state_tuple in
                                self.product_id_to_state_tuple]
                reward_name = name + state_variable
                reward_models[reward_name] = stormpy.SparseRewardModel(state_reward)

        if reward_models:
            components = stormpy.SparseModelComponents(transition_matrix=self.composed_model.transition_matrix,
                                                       state_labeling=self.composed_model.labeling,
                                                       reward_models=reward_models,
                                                       rate_transitions=False)

            self.composed_model = stormpy.storage.SparseMdp(components)

    def refactor_specification(self, want_to_export):
        # refactor formulae to have only reachability probabilities
        self.lines = []
        for index, property in enumerate(self.specification.stormpy_properties()):

            # refactor the formula
            formula = property.raw_formula
            logger.info(f"Refactoring formula: {formula}")
            rf = str(formula)
            formula_re = re.compile(r'^(.*)\[(.*)\]')
            match = formula_re.search(rf)
            if match is None:
                raise Exception(f"Formula is not supported: {rf}!")
            if not property.raw_formula.subformula.is_eventually_formula or want_to_export:
                new_rf = f"{match.group(1)}[F \"target{index}\"]\n"
                target_states = [state for state, targetFormulas in self.target_sets.items() if index in targetFormulas]
                target_state_info = len(target_states) if len(target_states) > 20  else target_states
                logger.info(f"target states for target{index} (only length is reported if above 20 states): {target_state_info}")
                assert target_states
                self.composed_model.labeling.add_label(f"target{index}")
                self.composed_model.labeling.set_states(f"target{index}",
                                                        stormpy.BitVector(self.composed_model.nr_states, target_states))
            else:
                new_rf = rf
            self.lines.append(new_rf)

    def export_to_drn(self, single_model, sketch_path):
        # export the build model
        sketch_folder = sketch_path.replace("sketch.templ", "")

        # export the model
        file_name = f"{sketch_folder}/model.drn"
        stormpy.export_to_drn(self.composed_model, file_name)

        #zip the model
        zipped_file_name = f"{sketch_folder}/model.zip"
        with zipfile.ZipFile(zipped_file_name, 'w', zipfile.ZIP_DEFLATED) as myzip:
            myzip.write(file_name)

        # remove non-zipped model
        os.remove(file_name)

        # some assertions for safety
        assert self.composed_model.labeling.contains_label("target0")
        assert not self.composed_model.labeling.contains_label("target1")

        if single_model.is_partially_observable:
            # refactor the state labeling: each agent gets observations, not the full information of the current state
            # the agent is the DRA, which always gets the same observation.
            get_obs = lambda state_id: single_model.get_observation(state_id)
            self.product_id_to_state_tuple = list(map(lambda t: tuple(map(get_obs, t[:-1])) + (0,),
                                                      self.product_id_to_state_tuple))

        with open(f"{sketch_folder}/helpers.json", "w") as file:
            helpers = {"state labeling": self.product_id_to_state_tuple,
                       "target states": list(self.composed_model.labeling.get_states('target0')),
                       "choice labeling": self.choice_to_action_tuple}

            # accepting states of the full model
            logger.info(
                f"Setting {len(list(self.composed_model.labeling.get_states('target0')))} target states while exporting.")
            json.dump(helpers, file)

        logger.info("hyperExport OK, aborting...")
        exit(0)

    def read_prism(self, sketch_path, properties_path, relative_error, discount_factor, export):

        # loading the specification file
        logger.info(f"Loading properties from {properties_path} ...")
        self.load_specification(properties_path)

        # parsing the scheduler quantifications
        logger.info(f"Parsing scheduler quantifiers ...")
        self.parse_scheduler_quants()
        logger.info(f"Found the following scheduler quantifiers: {self.sched_quant_dict}")

        # parsing state quantifications
        logger.info("Parsing state quantifiers...")
        self.parse_state_quants()
        logger.info(f"Found the following state quantifiers: {self.state_quant_dict}")

        # parsing restrictions on the initial states
        logger.info(f"Parsing restrictions")
        self.parse_restrictions()
        logger.info(f"Found the following restrictions: {self.state_quant_restrictions}")

        # parse program
        logger.info(f"Loading sketch from {sketch_path}...")
        logger.info(f"Assuming a sketch in a PRISM format ...")
        prism = stormpy.parse_prism_program(sketch_path, prism_compat=True)

        # initializing the building options
        builder_options = stormpy.BuilderOptions()
        builder_options.set_build_with_choice_origins(True)
        builder_options.set_build_state_valuations(True)
        builder_options.set_add_overlapping_guards_label(True)
        builder_options.set_build_observation_valuations(True)
        builder_options.set_build_all_labels(True)
        builder_options.set_build_choice_labels(True)

        # building the given model as it is
        single_model = stormpy.build_sparse_model_with_options(prism, builder_options)
        nr_initial_states = len(single_model.initial_states)
        nr_states = single_model.nr_states
        nr_replicas = len(self.state_quant_dict)
        logger.info(
            f"The original (non self-composed) model has {nr_initial_states} initial states and {nr_states} states")
        if single_model.is_partially_observable:
            logger.info(f"The original (non self-composed) model is a POMDP!")

        # generate the family of holes (aka parameters, schedulers' choices...)
        logger.info(f"Generating the family...")
        if single_model.is_partially_observable:
            family = self.generate_partially_observable_family(single_model, nr_replicas)
        else:
            family = self.generate_locally_fully_observable_family(single_model, nr_replicas)
        logger.info(f"Current family has {family.num_holes} holes")

        # check if export is needed
        assert (export is None) or (export == "drn"), "can export hypermodels only in zipped drn format for the moment"
        want_to_export = export is not None

        # build the self-composition between multiple replicase
        self.build_self_composition(single_model, want_to_export, nr_replicas)

        # actual parsing of the properties
        logger.info("Checking that we have a single initial state...")
        assert len(self.composed_model.initial_states) == 1, \
            f"The self-composed model has {len(self.composed_model.initial_states)} initial states"
        logger.info("We have a single initial state now, so no instantiation of state quantifiers will be done")
        self.parse_specification(relative_error, discount_factor)

        single_property = len(self.specification.stormpy_properties()) == 1
        # call STORM/SPOT construction of DRA
        self.build_cross_product(single_model, want_to_export, single_property)

        self.refactor_specification(want_to_export)
        self.parse_specification(relative_error, discount_factor)

        # export the model, if required
        if want_to_export:
            assert single_property, "Cannot export a multi-property model, since it is not a Dec-POMDP."
            logging.info("Exporting model...")
            self.export_to_drn(single_model, sketch_path)

        # generating the coloring
        logger.info("Generating the coloring")
        coloring = payntbind.synthesis.Coloring(family.family, self.composed_model.nondeterministic_choice_indices,
                                                self.choice_to_hole_options)

        return None, self.composed_model, self.specification, family, coloring, None, None

