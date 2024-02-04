import numpy as np

action_names = ["left", "stay", "right"]
state_names = ["in charge", "s1", "s2", "s3", "s4", "garbage"]


def r(battery, garbage) -> np.ndarray:
    battery_count = 10 if battery < 50 else 0
    garbage_count = 1 if garbage else 0
    return np.array(
        [
            [0, battery_count, 0],
            [battery_count, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, garbage_count],
        ]
    )


def t(state, action):
    if action == 1:
        return state
    elif action == 0:
        if state == 0:
            return 0
        return state - 1
    elif action == 2:
        if state == len(state_names) - 1:
            return state
        return state + 1


def get_state_text(index):
    return state_names[index]


def max_next_state_q(q, state):
    next_state_value = -np.inf
    for i in range(len(q[state])):
        if q[state][i] > next_state_value:
            next_state_value = q[state][i]
    return next_state_value


def calculate_q(rewards, gama):
    q = np.zeros(rewards.shape)
    for _ in range(10):
        for i in range(len(rewards)):
            # We are in state i, now take action
            for j in range(len(rewards[i])):
                q[i][j] = rewards[i][j] + gama * max_next_state_q(q, t(i, j))
    return q


def make_decision(q, state):
    next_state_value = -np.inf
    index = -np.inf
    for i in range(len(q[state])):
        if q[state][i] > next_state_value:
            next_state_value = q[state][i]
            index = i
    if next_state_value == 0:
        index = 1
    return index


def main():
    battery = 100
    garbage = True
    gama = 0.5
    state = 2
    for _ in range(100):
        battery -= 10
        if garbage and state == len(state_names) - 1:
            garbage = not garbage
        if state == 0:
            battery = 100
            garbage = True
        rewards = r(battery, garbage)
        q = calculate_q(rewards, gama)
        prev_state = state
        state = t(prev_state, make_decision(q, prev_state))
        print(
            f"previous state = {get_state_text(prev_state)}\ncurrent state = {get_state_text(state)}\nInformation:\n\tBattery: {battery}%\n\tGarbage: {'Exists' if garbage else 'Grabbed'}"
        )
        input()


if __name__ == "__main__":
    main()
