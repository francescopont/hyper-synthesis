import stormpy
import payntbind

import paynt.family.family
import paynt.verification.property
import paynt.parser.jani

from itertools import product
from math import prod

import os
import re

import logging

logger = logging.getLogger(__name__)

class PrismHyperParser:

    def __init__(self):
        self.sched_quant_dict = {}
        self.state_quant_dict = {}
        self.state_quant_restrictions = {}

        # parsed lines of the property
        self.lines = []

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

            # for now we support only straighforward synthesis specifications
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
            state_name = match.group(2)
            sched_name = match.group(3)

            # every scheduler variable must be quantified
            if sched_name not in list(self.sched_quant_dict.keys()):
                raise Exception(f"a scheduler variable occurs free in the state quantifications: {line}")

            # the implementation of HyperPaynt supports only specifications in conjunctive normal form
            if existential_quantifier and state_quant == "A":
                raise Exception(f"this nesting E*A* of state quantifications is not allowed: please use conjunctions of disjuctions (A*E*)")

            if state_name in list(self.state_quant_dict.keys()):
                raise Exception(f"two state variables cannot have the same name: {state_name}")

            if state_quant == "E":
                existential_quantifier = True

            # add the state quantifier to the dictionary
            self.state_quant_dict[state_name] = (state_quant, sched_name)

            # end of state quantifiers
            if match.group(5) == "":
                break

            # move to next state quantifier
            line = match.group(5)
            match = state_re.search(line)
            if match is None:
                raise Exception(f"the input formula is wrongly formatted: {line}")

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
            state_name = match.group(1)
            restriction_label = match.group(2)
            if state_name not in self.state_quant_dict:
                raise Exception(f"Trying to restrict a variable not in scope: {state_name}")
            replica_number = list(self.state_quant_dict).index(state_name)
            if replica_number in self.state_quant_restrictions:
                raise Exception(f"Trying to restrict two times the same variable: {state_name}")

            # storing this restriction
            self.state_quant_restrictions[replica_number] = restriction_label

            if match.group(4) == "":
                # no other restrictions found
                return

            # move to next restriction
            line = match.group(4)
            match = restriction_re.search(line)
            if match is None:
                raise Exception(f"Wrong formatted restrictions: {line}")

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

        specification = paynt.verification.property.Specification(properties)
        logger.info(f"found the following specification: {specification}")
        assert not specification.has_double_optimality, "did not expect double-optimality property"
        return specification

    def state_map(self, tup, state_count, acc = 0):
        if len(tup) == 1:
            return acc + tup[0]
        else:
            return self.state_map(tup[1:], state_count, acc + (tup[0] * (state_count ** (len(tup) -1))))
    def read_prism(self, sketch_path, properties_path, relative_error, discount_factor):

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
        logger.info(f"The original (non self-composed) model has {nr_initial_states} initial states and {nr_states} states")

        # generate the family of holes (aka parameters, schedulers' choices...)
        logger.info(f"Generating the family...")
        family = paynt.family.family.Family()
        nr_replicas = len(self.state_quant_dict)
        state_variables = list(self.state_quant_dict.keys())
        state_to_hole_index = [{} for _ in range(nr_replicas)]
        for state in single_model.states:
            state_name = single_model.state_valuations.get_string(state.id)
            num_actions = single_model.get_nr_available_actions(state.id)
            option_labels = []
            if num_actions > 1:
                for offset in range(num_actions):
                    choice = single_model.get_choice_index(state, offset)
                    label = single_model.choice_labeling.get_labels_of_choice(choice)
                    option_labels.append(str(label))
                state_holes = {}
                for sched_name in self.sched_quant_dict.keys():
                    replica_name = f"{state_name}(sched {sched_name})"
                    hole_index = family.num_holes
                    family.add_hole(replica_name, option_labels)
                    state_holes[sched_name] = hole_index
                for index, (_, sched_name) in enumerate(self.state_quant_dict.values()):
                    state_to_hole_index[index][state.id] = state_holes[sched_name]
        logger.info(f"Current family has {family.num_holes} holes")

        # generate the state set of the self-composition
        state_permutations = list(product(range(nr_states), repeat= nr_replicas))
        builder = stormpy.SparseMatrixBuilder(rows=0,columns=0, entries=0, force_dimensions=False,
                                              has_custom_row_grouping=True, row_groups=0)

        # generate the labelings of the self-composition
        logger.info("Generating the labels of the self-composition")
        cross_state_labeling = stormpy.storage.StateLabeling(len(state_permutations))
        for label in single_model.labeling.get_labels():
            if label == 'deadlock':
                states = list(single_model.labeling.get_states(label))
                affected_states = list(map(lambda tup: self.state_map(tup, nr_states), product(states, repeat=nr_replicas)))
                cross_state_labeling.add_label(label)
                cross_state_labeling.set_states(label,
                                                stormpy.BitVector(len(state_permutations), affected_states))
            if label == 'init':
                states = list(single_model.labeling.get_states(label))
                affected_states_tuples = list(product(states, repeat=nr_replicas))
                filtered_state_tuples = list(filter(lambda tup: all(map(lambda t: t[1] in single_model.labeling.get_labels_of_state(tup[t[0]]),
                            list(self.state_quant_restrictions.items()))), affected_states_tuples))
                affected_states = list(
                        map(lambda tup: self.state_map(tup, nr_states),filtered_state_tuples))
                cross_state_labeling.add_label(label)
                cross_state_labeling.set_states(label,
                                                stormpy.BitVector(len(state_permutations), affected_states))
            else:
                affected_states = list(single_model.labeling.get_states(label))
                for index, state_variable in enumerate(state_variables):
                    cross_label = label + state_variable
                    cross_state_labeling.add_label(cross_label)
                    cross_state_labeling.set_states(cross_label,
                                                   stormpy.BitVector(len(state_permutations),
                                                                     [i for i,tup in enumerate(state_permutations)
                                                                      if tup[index] in affected_states]))
        # generate the reward models of the self-composition
        logger.info(f"Generating the reward model of the self-composition")
        cross_reward_models = {}
        for name, reward_model in single_model.reward_models.items():
            assert reward_model.has_state_rewards
            assert not reward_model.has_state_action_rewards
            assert not reward_model.has_transition_rewards
            for index, state_variable in enumerate(state_variables):
                state_reward = [reward_model.get_state_reward(state_tuple[index]) for state_tuple in state_permutations]
                cross_name = name + state_variable
                cross_reward_models[cross_name] = stormpy.SparseRewardModel(state_reward)

        # generate the transition matrix of the self-composition
        logger.info(f"generating the transition system of the self-composition")
        choice_to_hole_options = []
        cross_product_row_counter = 0
        for states_tuple in state_permutations:
            # generate the state in the cross product associated with this tuple
            states = list(map(lambda id: single_model.states[id], states_tuple))
            actions_lists = list(map(lambda id: range(single_model.get_nr_available_actions(id)), states_tuple))
            actions_tuples = product(*actions_lists) # all the actions of the tuple of states
            builder.new_row_group(cross_product_row_counter)
            for actions_tuple in actions_tuples:
                choice_to_hole_options.append([])
                transitions_lists = [states[index].actions[id].transitions for index, id in enumerate(actions_tuple)]
                for (index, action_id) in enumerate(actions_tuple):
                    hole_id = state_to_hole_index[index].get(states_tuple[index], None)
                    if hole_id is not None: # if it is None, then there is no hole associated with this state
                        choice_to_hole_options[cross_product_row_counter].append((hole_id,action_id))
                transitions_tuples = product(*transitions_lists)
                for transitions_tuple in transitions_tuples:
                    destination_tuple = list(map(lambda transition: transition.column, transitions_tuple))
                    destination = self.state_map(destination_tuple, nr_states)
                    value = prod(map(lambda transition: transition.value(), transitions_tuple))
                    builder.add_next_value(cross_product_row_counter, destination, value)
                cross_product_row_counter += 1



        product_transition_matrix = builder.build()
        components = stormpy.SparseModelComponents(transition_matrix=product_transition_matrix,
                                                   state_labeling=cross_state_labeling,
                                                   reward_models=cross_reward_models,
                                                   rate_transitions=False)

        # build the self-composition (which is the quotient mdp)
        quotient_mdp = stormpy.storage.SparseMdp(components)
        logger.info(f"Number of states of the self-composition: {quotient_mdp.nr_states}")

        # actual parsing of the properties
        logger.info("Checking that we have a single initial state...")
        assert len(quotient_mdp.initial_states) == 1, f"The self-composed model has {len(quotient_mdp.initial_states)} initial states"
        logger.info("We have a single initial state now, so no instantiation of the state quantifications will be done")
        specification = self.parse_specification(relative_error, discount_factor)

        if specification.is_single_property:
            single_property = specification.stormpy_properties()[0]
            single_formula = single_property.raw_formula
            if single_formula.subformula.is_complex_path_formula:
                logger.info("Generating explicit cross-product due to presence of a complex formula")
                #generate the cross-product model
                product_rep  = stormpy.build_product_model(quotient_mdp, single_formula)
                new_quotient_mdp = product_rep.product_model
                logger.info(f"Number of states of the cross-product: {new_quotient_mdp.nr_states}")
                new_quotient_mdp.labeling.add_label("target")
                new_quotient_mdp.labeling.set_states("target", product_rep.accepting_states)

                new_initial = None
                assert len(quotient_mdp.initial_states) == 1
                for ((MDP_state, _), index) in product_rep.product_state_to_product_index.items():
                    if MDP_state == quotient_mdp.initial_states[0]:
                        if new_initial is not None:
                            raise Exception("the cross product has multiple initial states, it is not deterministic")
                        new_initial = index

                new_quotient_mdp.labeling.add_label("init")
                new_quotient_mdp.labeling.add_label_to_state("init", new_initial)

                # generate the family and the choice_to_hole_option mapping
                logger.info("Regenerating the family and the choice-to-hole-options to adapt to cross-product")
                new_choice_to_hole_options = []

                product_hole_to_hole_index = {} # this keeps track of whether the new hole with unfolded memory value has been created

                p_index_to_p_state = product_rep.product_index_to_product_state
                for state in new_quotient_mdp.states:
                    num_actions = new_quotient_mdp.get_nr_available_actions(state.id)
                    if num_actions > 1:
                        # this state has to be mapped to a hole
                        (mdp_state, memory_value) = p_index_to_p_state[state.id]
                        for offset in range(num_actions):
                            old_choice = quotient_mdp.get_choice_index(mdp_state, offset)
                            old_choice_hole_options = choice_to_hole_options[old_choice]
                            hole_options = []
                            assert old_choice_hole_options
                            if memory_value == 0:
                                hole_options = old_choice_hole_options # keep old holes
                            else:
                                for (hole_id, action_id) in old_choice_hole_options:
                                    new_hole_id = product_hole_to_hole_index.get((hole_id, memory_value), None)
                                    if new_hole_id is None:
                                        # create a new hole
                                        new_name = f"{family.hole_name(hole_id)}(memory = {memory_value})"
                                        option_labels = family.hole_options_strings(hole_id).copy()
                                        new_hole_id = family.num_holes
                                        family.add_hole(new_name, option_labels)
                                        product_hole_to_hole_index[(hole_id, memory_value)] = new_hole_id
                                    hole_options.append((new_hole_id, action_id))
                            new_choice_to_hole_options.append(hole_options)
                    else: new_choice_to_hole_options.append([])
                logger.info(f"The updated family has {family.num_holes} holes")

                # refactor the formula
                logger.info("Refactoring the formula!")
                rf = str(single_formula)
                formula_re = re.compile(r'^(.*)\[(.*)\]')
                match = formula_re.search(rf)
                if match is None:
                    raise Exception(f"Formula is not supported: {rf}!")
                new_rf = f"{match.group(1)}[F \"target\"]\n"
                self.lines = [new_rf]
                new_specification = self.parse_specification(relative_error, discount_factor)

                # updating the variables
                quotient_mdp = new_quotient_mdp
                choice_to_hole_options = new_choice_to_hole_options
                specification = new_specification

        # generating the coloring
        logger.info("Generating the coloring")
        coloring = payntbind.synthesis.Coloring(family.family, quotient_mdp.nondeterministic_choice_indices,
                                                choice_to_hole_options)

        return None, quotient_mdp, specification, family, coloring, None, None
