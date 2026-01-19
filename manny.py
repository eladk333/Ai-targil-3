import tkinter as tk
from tkinter import messagebox
import ext_plant
import sys

# --- Problem Definition: problem_new3_version1 ---
problem_new3_version1 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (6, 1), (7, 1), (8, 1), (9, 1),
        (4, 2), (4, 3), (6, 2), (6, 3),
    },
    "Taps": {
        (5, 3): 20,
    },
    "Plants": {
        (0, 0): 10,  # upper-right corridor
        (9, 0): 10,
    },
    "Robots": {
        10: (2, 0, 0, 2),   # ID 10
        11: (7, 0, 0, 20),  # ID 11
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
    "horizon": 60,
}

TARGET_BASELINE = 71.0

# --- GUI Constants ---
CELL_SIZE = 60
COLOR_WALL = "#404040"
COLOR_EMPTY = "#FFFFFF"
COLOR_TAP = "#4FC3F7"   # Light Blue
COLOR_PLANT = "#81C784" # Green
COLOR_ROBOT_1 = "#FF7043" # Orange (Robot 10)
COLOR_ROBOT_2 = "#AB47BC" # Purple (Robot 11)

class PlantWateringGUI:
    def __init__(self, master, problem):
        self.master = master
        self.master.title("Plant Watering - Manual Play")
        
        # Initialize Game Logic
        self.game = ext_plant.Game(problem, debug=False)
        self.robot_ids = list(problem["Robots"].keys())
        self.selected_robot_var = tk.IntVar(value=self.robot_ids[0])
        
        # Dimensions
        self.rows, self.cols = problem["Size"]
        self.width = self.cols * CELL_SIZE
        self.height = self.rows * CELL_SIZE

        # --- Layout ---
        
        # Top Stats Bar
        self.stats_frame = tk.Frame(master, bg="#eee", pady=5)
        self.stats_frame.pack(fill=tk.X)
        
        self.lbl_steps = tk.Label(self.stats_frame, text="Steps: 0/60", font=("Arial", 12, "bold"))
        self.lbl_steps.pack(side=tk.LEFT, padx=10)
        
        self.lbl_reward = tk.Label(self.stats_frame, text="Reward: 0", font=("Arial", 12, "bold"), fg="blue")
        self.lbl_reward.pack(side=tk.LEFT, padx=10)
        
        self.lbl_target = tk.Label(self.stats_frame, text=f"Target: {TARGET_BASELINE}", font=("Arial", 10), fg="gray")
        self.lbl_target.pack(side=tk.LEFT, padx=10)

        # Canvas for Grid
        self.canvas = tk.Canvas(master, width=self.width, height=self.height, bg="white")
        self.canvas.pack(pady=10, padx=10)

        # Control Panel
        self.controls_frame = tk.Frame(master, pady=10)
        self.controls_frame.pack()

        # Robot Selector
        tk.Label(self.controls_frame, text="Select Robot:", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3)
        self.rb1 = tk.Radiobutton(self.controls_frame, text=f"Robot {self.robot_ids[0]}", variable=self.selected_robot_var, value=self.robot_ids[0], fg=COLOR_ROBOT_1)
        self.rb1.grid(row=1, column=0)
        if len(self.robot_ids) > 1:
            self.rb2 = tk.Radiobutton(self.controls_frame, text=f"Robot {self.robot_ids[1]}", variable=self.selected_robot_var, value=self.robot_ids[1], fg=COLOR_ROBOT_2)
            self.rb2.grid(row=1, column=2)

        # Action Buttons
        btn_width = 8
        tk.Button(self.controls_frame, text="UP", command=lambda: self.do_action("UP"), width=btn_width).grid(row=2, column=1, pady=5)
        tk.Button(self.controls_frame, text="LEFT", command=lambda: self.do_action("LEFT"), width=btn_width).grid(row=3, column=0)
        tk.Button(self.controls_frame, text="DOWN", command=lambda: self.do_action("DOWN"), width=btn_width).grid(row=3, column=1)
        tk.Button(self.controls_frame, text="RIGHT", command=lambda: self.do_action("RIGHT"), width=btn_width).grid(row=3, column=2)
        
        tk.Button(self.controls_frame, text="LOAD", command=lambda: self.do_action("LOAD"), bg="#E1F5FE", width=btn_width).grid(row=4, column=0, pady=10)
        tk.Button(self.controls_frame, text="POUR", command=lambda: self.do_action("POUR"), bg="#E8F5E9", width=btn_width).grid(row=4, column=2, pady=10)
        
        tk.Button(self.controls_frame, text="RESET GAME", command=self.reset_game, fg="red").grid(row=5, column=1, pady=5)

        # Log
        self.lbl_log = tk.Label(master, text="Welcome! Use buttons or Arrows/L/P keys.", fg="gray")
        self.lbl_log.pack()

        # Keyboard Bindings
        master.bind('<Up>', lambda e: self.do_action("UP"))
        master.bind('<Down>', lambda e: self.do_action("DOWN"))
        master.bind('<Left>', lambda e: self.do_action("LEFT"))
        master.bind('<Right>', lambda e: self.do_action("RIGHT"))
        master.bind('l', lambda e: self.do_action("LOAD"))
        master.bind('p', lambda e: self.do_action("POUR"))
        master.bind('<Tab>', self.toggle_robot)

        self.draw_board()

    def toggle_robot(self, event=None):
        current = self.selected_robot_var.get()
        if current == self.robot_ids[0] and len(self.robot_ids) > 1:
            self.selected_robot_var.set(self.robot_ids[1])
        else:
            self.selected_robot_var.set(self.robot_ids[0])

    def reset_game(self):
        # Sending RESET to engine
        try:
            self.game.submit_next_action("RESET")
            self.lbl_log.config(text="Game Reset.", fg="black")
            self.draw_board()
        except Exception as e:
            print(e)

    def do_action(self, action_type):
        if self.game.get_done():
            messagebox.showinfo("Game Over", f"Final Score: {self.game.get_current_reward()}")
            return

        rid = self.selected_robot_var.get()
        action_str = f"{action_type} ({rid})"
        
        try:
            prev_reward = self.game.get_current_reward()
            self.game.submit_next_action(action_str)
            curr_reward = self.game.get_current_reward()
            
            diff = curr_reward - prev_reward
            if diff > 0:
                self.lbl_log.config(text=f"Success! +{diff}", fg="green")
            else:
                self.lbl_log.config(text=f"{action_str} executed.", fg="black")
                
            self.draw_board()
            
        except ValueError as e:
            # Illegal move
            self.lbl_log.config(text=f"Illegal Move: {e}", fg="red")
        except Exception as e:
            self.lbl_log.config(text=f"Error: {e}", fg="red")

    def draw_board(self):
        self.canvas.delete("all")
        
        state = self.game.get_current_state()
        robots, plants, taps, total_need = state
        
        # Build dictionaries for easy lookup
        wall_set = self.game.get_problem().get("Walls", set())
        tap_map = {t[0]: t[1] for t in taps}
        plant_map = {p[0]: p[1] for p in plants}
        
        # Prepare robot data
        robot_positions = {} # pos -> list of robots
        for rid, pos, load in robots:
            if pos not in robot_positions: robot_positions[pos] = []
            robot_positions[pos].append((rid, load))

        # Update Stats
        steps = self.game.get_current_steps()
        max_steps = self.game.get_max_steps()
        self.lbl_steps.config(text=f"Steps: {steps}/{max_steps}")
        self.lbl_reward.config(text=f"Reward: {self.game.get_current_reward()}")

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                
                pos = (r, c)
                
                # Draw Background Cell
                fill_color = COLOR_EMPTY
                if pos in wall_set:
                    fill_color = COLOR_WALL
                elif pos in tap_map:
                    fill_color = COLOR_TAP
                elif pos in plant_map:
                    fill_color = COLOR_PLANT
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#ccc")

                # Draw Details (Tap amount / Plant Need)
                if pos in tap_map:
                    self.canvas.create_text(x1+10, y1+10, text="🚰", font=("Arial", 16))
                    self.canvas.create_text(x1+CELL_SIZE/2, y2-10, text=f"{tap_map[pos]}L", font=("Arial", 10, "bold"))
                
                if pos in plant_map:
                    self.canvas.create_text(x1+10, y1+10, text="🌱", font=("Arial", 16))
                    self.canvas.create_text(x1+CELL_SIZE/2, y2-10, text=f"Need: {plant_map[pos]}", font=("Arial", 10, "bold"), fill="white")

                # Draw Robots
                if pos in robot_positions:
                    # If multiple robots on one cell, offset them slightly
                    bots = robot_positions[pos]
                    for idx, (rid, load) in enumerate(bots):
                        offset = idx * 5
                        color = COLOR_ROBOT_1 if rid == self.robot_ids[0] else COLOR_ROBOT_2
                        
                        # Draw Robot Circle
                        pad = 15
                        self.canvas.create_oval(x1+pad+offset, y1+pad+offset, x2-pad+offset, y2-pad+offset, fill=color, outline="black")
                        
                        # Draw ID and Load
                        cap = self.game.get_capacities()[rid]
                        self.canvas.create_text(x1+CELL_SIZE/2+offset, y1+CELL_SIZE/2+offset, text=f"R{str(rid)[-1]}\n{load}/{cap}", font=("Arial", 8, "bold"), fill="white")

        if self.game.get_done():
            self.canvas.create_text(self.width/2, self.height/2, text="GAME OVER", font=("Arial", 30, "bold"), fill="red")


if __name__ == "__main__":
    root = tk.Tk()
    gui = PlantWateringGUI(root, problem_new3_version1)
    root.mainloop()