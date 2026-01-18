import ext_plant
import ex3
import time
import numpy as np
# import ex3_random
import sys

# ANSI color codes for nicer terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
SEP = "\n" + ("=" * 60)


def draw_board(problem):
    """Draw a simple emoji board for the given problem dict."""
    rows, cols = problem.get("Size", (0, 0))
    walls = set(problem.get("Walls", []))
    taps = dict(problem.get("Taps", {}))
    plants = dict(problem.get("Plants", {}))
    robots = dict(problem.get("Robots", {}))

    # Build grid rows (r from 0..rows-1)
    grid_lines = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            pos = (r, c)
            if pos in walls:
                cell = "🧱 "
            elif pos in taps:
                # show tap emoji
                cell = "🚰 "
            elif pos in plants:
                cell = "🌱 "
            else:
                cell = "⬜ "

            # overlay robot if present (show last digit of id)
            robot_here = None
            for rid, (rr, cc, _carried, _cap) in robots.items():
                if (rr, cc) == pos:
                    robot_here = rid
                    break
            if robot_here is not None:
                cell = f"🤖{str(robot_here)[-1]}"

            row_cells.append(cell)
        grid_lines.append(" ".join(row_cells))

    print("\nBoard layout:")
    for line in grid_lines:
        print(line)
    # also print legends for taps/plants/robots (counts)
    if taps:
        taps_str = ", ".join([f"{pos}:{amt}" for pos, amt in taps.items()])
        print(f"Taps: {taps_str}")
    if plants:
        plants_str = ", ".join([f"{pos}:{need}" for pos, need in plants.items()])
        print(f"Plants: {plants_str}")
    if robots:
        robots_str = ", ".join(
            [f"{rid}:{(r,c)}" for rid, (r, c, _car, _cap) in robots.items()]
        )
        print(f"Robots: {robots_str}")
    # Show robot action success probabilities if provided
    robot_probs = problem.get("robot_chosen_action_prob", {})
    if robot_probs:
        probs_str = ", ".join(
            [f"{rid}:{prob:.2f}" for rid, prob in robot_probs.items()]
        )
        print(f"Robot success probs: {probs_str}")

    # Show plant reward distributions if provided
    plants_reward = problem.get("plants_reward", {})
    if plants_reward:
        pr_strs = []
        for pos, rewards in plants_reward.items():
            pr_strs.append(f"{pos}:{rewards}")
        print(f"Plant rewards: {', '.join(pr_strs)}")


def solve(game: ext_plant.Game, run_idx: int, controller_module):
    policy = controller_module.Controller(game)
    for i in range(game.get_max_steps()):
        game.submit_next_action(
            chosen_action=policy.choose_next_action(game.get_current_state())
        )
        if game.get_done():
            break

    r = game.get_current_reward()
    state = game.get_current_state()
    state_str = str(state)

    finished_txt = "SUCCESS" if state[-1] else "FAILED"
    color = GREEN if state[-1] else RED

    print(
        f"Run {run_idx:2d}: {BOLD}{YELLOW}Reward: {r:3d}{RESET} | Steps: {game.get_max_steps():2d} | {color}{finished_txt}{RESET} \nState: {state_str}"
    )
    return r


def solve2(game: ext_plant.Game):
    policy = ex3_random.Controller(game)
    for i in range(game.get_max_steps()):
        game.submit_next_action(
            chosen_action=policy.choose_next_action(game.get_current_state())
        )
        if game.get_done():
            break
    print(
        "Game result:",
        game.get_current_state(),
        "\n\tFinished in",
        game.get_max_steps(),
        "Steps.\n\tReward result->",
        game.get_current_reward(),
    )
    print(
        "Game finished ",
        "" if game.get_current_state()[-1] else "un",
        "successfully.",
        sep="",
    )
    game.show_history()
    return game.get_current_reward()


problem_pdf = {
    "Size": (3, 3),
    "Walls": {(0, 1), (2, 1)},
    "Taps": {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.9,
    },
    "goal_reward": 10,
    "plants_reward": {
        (0, 2): [1, 2, 3, 4],
        (2, 0): [1, 2, 3, 4],
    },
    "seed": 45,
    "horizon": 30,
}


problem_pdf2 = {
    "Size": (3, 3),
    "Walls": {(0, 1), (2, 1)},
    "Taps": {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob": {
        10: 0.9,
        11: 0.8,
    },
    "goal_reward": 12,
    "plants_reward": {
        (0, 2): [1, 3, 5, 7],
        (2, 0): [1, 2, 3, 4],
    },
    "seed": 45,
    "horizon": 35,
}

problem_pdf3 = {
    "Size": (3, 3),
    "Walls": {(0, 1), (2, 1)},
    "Taps": {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob": {
        10: 0.7,
        11: 0.6,
    },
    "goal_reward": 30,
    "plants_reward": {
        (0, 2): [1, 2, 3, 4],
        (2, 0): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new1_version1 = {
    "Size": (5, 6),
    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2),
        (1, 3),
        (3, 2),
        (3, 3),
    },
    "Taps": {
        (2, 2): 12,
    },
    "Plants": {
        (0, 1): 3,
        (4, 5): 6,
    },
    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
    "robot_chosen_action_prob": {
        10: 0.9,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5): [1, 2, 3, 4],
        (0, 1): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 30,
}
problem_new1_version2 = {
    "Size": (5, 6),
    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2),
        (1, 3),
        (3, 2),
        (3, 3),
    },
    "Taps": {
        (2, 2): 12,
    },
    "Plants": {
        (0, 1): 3,
        (4, 5): 6,
    },
    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
    "robot_chosen_action_prob": {
        10: 0.6,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5): [1, 2, 3, 4],
        (0, 1): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 70,
}
problem_new1_version3 = {
    "Size": (5, 6),
    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2),
        (1, 3),
        (3, 2),
        (3, 3),
    },
    "Taps": {
        (2, 2): 12,
    },
    "Plants": {
        (0, 1): 2,
        (4, 5): 6,
    },
    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
    "robot_chosen_action_prob": {
        10: 0.6,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5): [1, 2, 3, 4],
        (0, 1): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new2_version1 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0): [5, 7],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new2_version2 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0): [5, 7],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 70,
}
problem_new2_version3 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 20,
    "plants_reward": {
        (0, 0): [5, 7, 9],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 30,
}
problem_new2_version4 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.7,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0): [5, 7],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 40,
}


problem_new3_version1 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (4, 2),
        (4, 3),
        (6, 2),
        (6, 3),
    },
    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },
    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 10,  # upper-right corrido
        (9, 0): 10,
    },
    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),  # bottom-left area near the tap side
        11: (7, 0, 0, 20),  # bottom-left area near the tap side
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (9, 0): [1, 3],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new3_version2 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (4, 2),
        (4, 3),
        (6, 2),
        (6, 3),
    },
    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },
    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 10,  # upper-right corrido
        (9, 0): 10,
    },
    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),  # bottom-left area near the tap side
        11: (7, 0, 0, 20),  # bottom-left area near the tap side
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.8,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (9, 0): [1, 3],
    },
    "seed": 45,
    "horizon": 50,
}


problem_new3_version3 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (4, 2),
        (4, 3),
        (6, 2),
        (6, 3),
    },
    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },
    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 5,  # upper-right corrido
        (9, 0): 5,
    },
    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),  # bottom-left area near the tap side
        11: (7, 0, 0, 20),  # bottom-left area near the tap side
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.0001,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (9, 0): [1, 3],
    },
    "seed": 45,
    "horizon": 210,  # "horizon": 70,
}
# reset ?
problem_new4_version1 = {
    "Size": (10, 10),
    "Walls": set(),  # completely open grid
    "Taps": {
        (8, 8): 24,
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (0, 9): 5,  # top-right
        (9, 0): 5,  # bottom-left
        (9, 9): 5,  # bottom-right
        # total need = 20
    },
    "Robots": {
        10: (8, 9, 0, 5),
    },
    "robot_chosen_action_prob": {
        10: 0.95,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (0, 9): [1, 3],
        (9, 0): [1, 3],
        (9, 9): [1, 3],
    },
    "seed": 45,
    "horizon": 210,  # "horizon": 70,
}

# reset ?
problem_new4_version2 = {
    "Size": (10, 10),
    "Walls": set(),  # completely open grid
    "Taps": {
        (8, 8): 24,
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (0, 9): 5,  # top-right
        (9, 0): 5,  # bottom-left
        (9, 9): 5,  # bottom-right
        # total need = 20
    },
    "Robots": {
        10: (8, 9, 0, 5),
    },
    "robot_chosen_action_prob": {
        10: 0.85,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (0, 9): [1, 3],
        (9, 0): [1, 3],
        (9, 9): [1, 3],
    },
    "seed": 45,
    "horizon": 40,
}


def append_lines(path: str, lines: list[str]):
    # writes in the SAME place you run the program (current directory)
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    debug_mode = False
    n_runs = 30

    # Keep the same problem list as before
    problems = [
        ("problem_pdf", problem_pdf),
        ("problem_pdf2", problem_pdf2),
        ("problem_pdf3", problem_pdf3),
        # ("problem_new1_version1", problem_new1_version1),
        # ("problem_new1_version2", problem_new1_version2),
        # ("problem_new1_version3", problem_new1_version3),
        # ("problem_new2_version1", problem_new2_version1),
        # ("problem_new2_version2", problem_new2_version2),
        # ("problem_new2_version3", problem_new2_version3),
        # ("problem_new2_version4", problem_new2_version4),
        # ("problem_new3_version1", problem_new3_version1),
        # ("problem_new3_version2", problem_new3_version2),
        # ("problem_new3_version3", problem_new3_version3),
        # ("problem_new4_version1", problem_new4_version1),
        # ("problem_new4_version2", problem_new4_version2),
    ]

    # Estimated baseline values provided by user
    baseline_map = {
        "problem_pdf": (19.86889, 0.0),
        "problem_pdf2": (28.33333, 0.0),
        "problem_pdf3": (32.56667, 0.0),
        "problem_new1_version1": (59.325, 0.0),
        "problem_new1_version2": (81.87778, 0.0),
        "problem_new1_version3": (25.472, 0.0),
        "problem_new2_version1": (44.33333, 0.0),
        "problem_new2_version2": (123.715, 0.0),
        "problem_new2_version3": (65.5859, 0.0),
        "problem_new2_version4": (41.65, 0.0),
        "problem_new3_version1": (5.925, 0.0),
        "problem_new3_version2": (7.176667, 0.0),
        "problem_new3_version3": (8.922222, 0.0),
        "problem_new4_version1": (120.5333, 0.0),
        "problem_new4_version2": (20.76238, 0.0),
    }

    summaries = []

    # choose controller module: default `ex3`, or `ex3_random` when 'random' passed
    args = sys.argv[1:]
    if "random" in args:
        controller_module = ex3_random
    else:
        controller_module = ex3

    for idx, (pname, problem) in enumerate(problems, start=1):
        print()
        print(f"*** Problem: {pname} ({idx}) ***")
        draw_board(problem)
        print(
            f"\n{SEP}\n{BOLD}{MAGENTA}--- Running {pname} (problem index slice item {idx}) ---{RESET}"
        )

        total_reward = 0.0
        horizon = problem.get("horizon", 0)
        time_limit = 20 + 0.5 * horizon

        run_times = []
        for seed in range(n_runs):
            problem["seed"] = seed
            game = ext_plant.create_pressure_plate_game((problem, debug_mode))

            run_start = time.time()
            run_reward = solve(game, seed, controller_module)
            run_duration = time.time() - run_start
            run_times.append(run_duration)
            total_reward += run_reward

            limit_ok = run_duration <= time_limit
            status_color = GREEN if limit_ok else RED
            print(
                f"Run {seed:2d} time: {run_duration:.2f}s | Limit: {time_limit:.2f}s | {status_color}{'OK' if limit_ok else 'TIMEOUT'}{RESET}"
            )

        duration = sum(run_times)
        avg_time_per_run = duration / n_runs if n_runs else 0.0

        num_timeouts = sum(1 for t in run_times if t > time_limit)
        time_status = (
            f"{GREEN}PASS{RESET}"
            if num_timeouts == 0
            else f"{RED}TIMEOUT ({num_timeouts}/{n_runs} runs exceeded {time_limit:.1f}s){RESET}"
        )

        avg_reward = total_reward / n_runs if n_runs else 0.0
        summaries.append(
            (
                pname,
                avg_reward,
                duration,
                baseline_map.get(pname, (None, None))[0],
                baseline_map.get(pname, (None, None))[1],
                time_status,
            )
        )

        if pname in baseline_map:
            baseline_reward, baseline_time = baseline_map[pname]
            pct = (avg_reward / baseline_reward * 100) if baseline_reward > 0 else 0
            comp = (
                f"{GREEN}BETTER{RESET}"
                if avg_reward > baseline_reward
                else f"{RED}WORSE{RESET}"
            )
            print(
                f"\nAverage reward over {n_runs} runs: {avg_reward:.6f} ({pct:.1f}% of estimated baseline {baseline_reward:.6f}) | {comp} than estimated baseline"
            )
            print(
                f"Estimated baseline time: {baseline_time}s | Current time: {duration:.2f}s | Status: {time_status}"
            )
        else:
            print(f"\nAverage reward over {n_runs} runs: {avg_reward}")
            print(f"Time taken: {duration:.2f}s | Status: {time_status}")

    # Final summary across all problems run
    print(f"\n{SEP}\n{BOLD}{CYAN}=== Summary (per problem) ==={RESET}")
    for pname, avg, dur, baseline_avg, baseline_time, t_status in summaries:
        avg_col = f"{GREEN}{avg:.2f}{RESET}" if avg >= 0 else f"{RED}{avg:.2f}{RESET}"
        if baseline_avg:
            pct = (avg / baseline_avg * 100) if baseline_avg > 0 else 0
            comp = (
                f"{GREEN}BETTER{RESET}" if avg > baseline_avg else f"{RED}WORSE{RESET}"
            )
            baseline_str = f"baseline {YELLOW}{baseline_avg:.2f}{RESET} (time {YELLOW}{baseline_time:.2f}s{RESET})"
        else:
            pct = 0
            comp = ""
            baseline_str = "no baseline"
        print(
            f"{BOLD}{pname}{RESET}: average = {avg_col} ({pct:.1f}% of {baseline_str}) | time = {YELLOW}{dur:.2f}s{RESET} | {comp} {t_status}"
        )

    total_time = sum(d for (_, _, d, _, _, _) in summaries)
    print(f"{BOLD}Total time for all problems: {RESET}{YELLOW}{total_time:.2f}s{RESET}")
    if summaries:
        overall_avg = sum(avg for (_, avg, _, _, _, _) in summaries) / len(summaries)
        print(
            f"{BOLD}Overall average across {len(summaries)} problems: {RESET}{YELLOW}{overall_avg:.2f}{RESET}"
        )


if __name__ == "__main__":
    main()
