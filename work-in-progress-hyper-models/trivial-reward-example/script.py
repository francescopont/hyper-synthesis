import stormpy

if __name__ == '__main__':
    path = "ex.pm"
    formula_str = "R=? [F \"target\"]"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)
    formula = formulas[0]
    builder_options = stormpy.BuilderOptions()
    builder_options.set_build_with_choice_origins(True)
    builder_options.set_build_state_valuations(True)
    builder_options.set_add_overlapping_guards_label(True)
    builder_options.set_build_observation_valuations(True)
    builder_options.set_build_all_labels(True)
    builder_options.set_build_choice_labels(True)

    model = stormpy.build_sparse_model_with_options(program, builder_options)
    initial_state = model.initial_states[0]
    assert initial_state == 0
    result = stormpy.model_checking(model, formula, extract_scheduler=True)
    print(f"{model.state_valuations}")
    for state in model.states:
        print(f"Result at {model.state_valuations.get_string(state.id)} = {result.at(state.id)}")