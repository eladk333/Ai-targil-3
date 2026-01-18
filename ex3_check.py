import ext_plant
import ex3
import time
import numpy as np

def solve(game: ext_plant.Game):
    # Initialize the Controller from ex3
    policy = ex3.Controller(game)
    
    # Run the simulation loop
    for i in range(game.get_max_steps()):
        # Submit the action chosen by your policy
        game.submit_next_action(chosen_action=policy.choose_next_action(game.get_current_state()))
        if game.get_done():
            break
            
    # Output results
    print('Game result:', game.get_current_state(), '\n\tFinished in', game.get_max_steps(),
         'Steps.\n\tReward result->', game.get_current_reward())
    print("Game finished ", "" if game.get_current_state()[-1] else "un", "successfully.", sep='')
    game.show_history()
    return game.get_current_reward()

# --- Problem Definitions ---

problem_pdf = {
    "Size":   (3, 3),
    "Walls":  {(0, 1), (2, 1)},
    "Taps":   {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob":{
        10: 0.95,
        11: 0.9,
    },
    "goal_reward": 10,
    "plants_reward": {
        (0, 2) : [1,2,3,4],
        (2, 0) : [1,2,3,4],
    },
    "seed": 45,
    "horizon": 30,
}

problem_pdf2 = {
    "Size":   (3, 3),
    "Walls":  {(0, 1), (2, 1)},
    "Taps":   {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob":{
        10: 0.9,
        11: 0.8,
    },
    "goal_reward": 12,
    "plants_reward": {
        (0, 2) : [1,3,5,7],
        (2, 0) : [1,2,3,4],
    },
    "seed": 45,
    "horizon": 35,
}

problem_pdf3 = {
    "Size":   (3, 3),
    "Walls":  {(0, 1), (2, 1)},
    "Taps":   {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob":{
        10: 0.7,
        11: 0.6,
    },
    "goal_reward": 30,
    "plants_reward": {
        (0, 2) : [1,2,3,4],
        (2, 0) : [10,11,12,13],
    },
    "seed": 45,
    "horizon": 30,
}

# --- Main Execution ---

def main():
    debug_mode = False
    n_runs = 30
    
    # List of problems to test
    problems = [
        ("problem_pdf", problem_pdf),
        # ("problem_pdf2", problem_pdf2),
        # ("problem_pdf3", problem_pdf3),
    ]

    out_file = "Solution_ex3.txt"
    
    # Prepare output file
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Averages per problem (n_runs = {n_runs})\n")
        f.write("=" * 50 + "\n\n")

    for name, problem in problems:
        total_reward = 0.0
        problem_start = time.perf_counter()

        print(f"\n--- Testing {name} ---")

        for seed in range(n_runs):
            run_start = time.perf_counter()

            # Set a different random seed each run for variability
            problem["seed"] = seed

            # Create a fresh game for this run
            # Note: ext_plant handles converting plants_reward to max_rewards for the agent
            game = ext_plant.create_pressure_plate_game((problem, debug_mode))

            # Solve and accumulate reward
            run_reward = solve(game)
            total_reward += run_reward

            run_end = time.perf_counter()
            run_time = run_end - run_start

            # Optional: Print every run
            # print(f"Run {seed}: reward = {run_reward} | time = {run_time:.4f}s")

        problem_end = time.perf_counter()
        total_time = problem_end - problem_start

        avg_reward = total_reward / n_runs
        avg_time = total_time / n_runs

        print(f"Results for {name}:")
        print(f"  Average Reward: {avg_reward}")
        print(f"  Average Time:   {avg_time:.4f}s")
        
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"{name}: reward_average={avg_reward:.6f} | time_average={avg_time:.6f}s\n")

if __name__ == "__main__":
    main()