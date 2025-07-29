
def get_robot_and_state_id_from_symbol(sym):
    """
    Extract robot and state ID from a symbol string.
    Similar to getRobotAndStateIDFromSymbol in C++ code.
    """
    LANDMARK_SYMBOL = 'L'
    MAP_SYMBOL = 'M'
    FIRST_AGENT_SYMBOL = 'A'

    if len(sym) < 1:
        raise ValueError(f"Invalid symbol format: {sym}")

    if sym[0] == LANDMARK_SYMBOL:
        # Landmark symbol
        if len(sym) < 2:
            raise ValueError(f"Invalid landmark symbol format: {sym}")

        if sym[1].isupper():
            if sym[1] == MAP_SYMBOL:
                # Map landmark warning
                print(f"Warning: landmark symbol 'LM#' is associated with the map. ",
                      "Map landmark features should be formatted as 'L#'.")
            # Landmark associated with a robot
            robot_id = ord(sym[1]) - ord(FIRST_AGENT_SYMBOL)
            state_id = int(sym[2:])
        else:
            # Landmark associated with the map
            robot_id = ord(MAP_SYMBOL) - ord(FIRST_AGENT_SYMBOL)
            state_id = int(sym[1:])
    elif sym[0].isupper():
        # Pose symbol
        robot_id = ord(sym[0]) - ord(FIRST_AGENT_SYMBOL)
        state_id = int(sym[1:])
    else:
        raise ValueError(f"Error parsing symbol: {sym}!")

    return (robot_id, state_id)
