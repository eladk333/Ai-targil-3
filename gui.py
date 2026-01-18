import tkinter as tk
from tkinter import ttk, messagebox
import time
import ext_plant
import ex3
import copy

# --- Problem Definitions (Exact match to ex3_check.py) ---
PROBLEM_LIST = [
    ("Problem 1 (Default)", {
        "Size":   (3, 3),
        "Walls":  {(0, 1), (2, 1)},
        "Taps":   {(1, 1): 6},
        "Plants": {(2, 0): 2, (0, 2): 3},
        "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
        "robot_chosen_action_prob": {10: 0.95, 11: 0.9},
        "goal_reward": 10,
        "plants_reward": {(0, 2): [1, 2, 3, 4], (2, 0): [1, 2, 3, 4]},
        "seed": 45,
        "horizon": 30,
    }),
    ("Problem 2 (Higher Reward/Horizon)", {
        "Size":   (3, 3),
        "Walls":  {(0, 1), (2, 1)},  # Same layout as Problem 1
        "Taps":   {(1, 1): 6},
        "Plants": {(2, 0): 2, (0, 2): 3},
        "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
        "robot_chosen_action_prob": {10: 0.9, 11: 0.8}, # Lower success rates
        "goal_reward": 12,
        "plants_reward": {(0, 2): [1, 3, 5, 7], (2, 0): [1, 2, 3, 4]}, # Different rewards
        "seed": 45,
        "horizon": 35, # Longer horizon
    }),
    ("Problem 3 (High Goal Reward)", {
        "Size":   (3, 3),
        "Walls":  {(0, 1), (2, 1)}, # Same layout as Problem 1
        "Taps":   {(1, 1): 6},
        "Plants": {(2, 0): 2, (0, 2): 3},
        "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
        "robot_chosen_action_prob": {10: 0.7, 11: 0.6}, # Much lower success rates
        "goal_reward": 30, # High goal reward
        "plants_reward": {(0, 2): [1, 2, 3, 4], (2, 0): [10, 11, 12, 13]},
        "seed": 45,
        "horizon": 30,
    })
]

class PlantWateringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Plant Watering Visualizer")
        self.root.geometry("1000x700")

        # --- Game Data ---
        self.history = []  # List of (state, action_taken_string, reward, step_num)
        self.max_steps = 0
        self.current_step_idx = 0
        self.is_playing = False
        self.play_speed_ms = 300
        self.problem_data = None
        self.capacities = {}

        # --- Colors ---
        self.colors = {
            'wall':      '#2c3e50',  # Dark Blue-Grey
            'empty':     '#ecf0f1',  # Light Grey
            'tap_bg':    '#d6eaf8',  # Very Light Blue
            'tap_border':'#3498db',  # Blue
            'plant_bg':  '#d5f5e3',  # Very Light Green
            'plant_border':'#2ecc71',# Green
            'robot':     '#e74c3c',  # Red
            'robot_border':'#c0392b',# Darker Red
            'text':      '#2c3e50',  # Dark Text
            'highlight': '#f1c40f'   # Yellow
        }

        self._setup_ui()
        self.load_problem(0)

    def _setup_ui(self):
        # 1. Main Layout: Canvas Left, Controls Right
        main_frame = tk.Frame(self.root, padx=10, pady=10, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas Area
        self.canvas_frame = tk.Frame(main_frame, bg="white", relief=tk.SUNKEN, bd=2)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.draw_current_step)

        # Controls Area
        controls_frame = tk.Frame(main_frame, width=300, padx=15, bg="#f0f0f0")
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Header
        tk.Label(controls_frame, text="Control Panel", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 20))

        # Problem Selector
        tk.Label(controls_frame, text="Select Problem:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor="w")
        self.prob_combo = ttk.Combobox(controls_frame, state="readonly", values=[p[0] for p in PROBLEM_LIST])
        self.prob_combo.current(0)
        self.prob_combo.bind("<<ComboboxSelected>>", lambda e: self.load_problem(self.prob_combo.current()))
        self.prob_combo.pack(fill=tk.X, pady=(0, 20))

        # Timeline Slider
        tk.Label(controls_frame, text="Timeline (Step):", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor="w")
        self.slider_var = tk.IntVar()
        self.slider = tk.Scale(controls_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                               variable=self.slider_var, command=self.on_slider_move, length=250)
        self.slider.pack(fill=tk.X, pady=(0, 20))

        # Playback Controls
        btn_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.btn_prev = tk.Button(btn_frame, text="<< Step", command=self.step_back, width=10)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        
        self.btn_play = tk.Button(btn_frame, text="▶ Play", command=self.toggle_play, bg="#2ecc71", fg="white", font=("Arial", 10, "bold"), width=10)
        self.btn_play.pack(side=tk.LEFT, padx=2)

        self.btn_next = tk.Button(btn_frame, text="Step >>", command=self.step_fwd, width=10)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        # Speed Control
        tk.Label(controls_frame, text="Playback Speed:", bg="#f0f0f0").pack(anchor="w", pady=(10, 0))
        self.speed_scale = tk.Scale(controls_frame, from_=100, to=1000, orient=tk.HORIZONTAL, resolution=50)
        self.speed_scale.set(300)
        self.speed_scale.pack(fill=tk.X, pady=(0, 20))

        # Stats Box
        self.stats_frame = tk.LabelFrame(controls_frame, text="Current State Stats", padx=10, pady=10, bg="white", font=("Arial", 10, "bold"))
        self.stats_frame.pack(fill=tk.X, expand=False)
        
        self.lbl_step = tk.Label(self.stats_frame, text="Step: 0", bg="white", anchor="w")
        self.lbl_step.pack(fill=tk.X)
        self.lbl_action = tk.Label(self.stats_frame, text="Last Action: -", bg="white", fg="blue", anchor="w")
        self.lbl_action.pack(fill=tk.X)
        self.lbl_reward = tk.Label(self.stats_frame, text="Reward: 0", bg="white", anchor="w")
        self.lbl_reward.pack(fill=tk.X)
        self.lbl_need = tk.Label(self.stats_frame, text="Global Need: 0", bg="white", anchor="w")
        self.lbl_need.pack(fill=tk.X)

        # Legend
        self._draw_legend(controls_frame)

    def _draw_legend(self, parent):
        leg_frame = tk.LabelFrame(parent, text="Legend", padx=10, pady=10, bg="#f0f0f0")
        leg_frame.pack(fill=tk.X, pady=20)
        
        items = [
            ("Tap (Water Left)", self.colors['tap_border']),
            ("Plant (Need)", self.colors['plant_border']),
            ("Robot (Load/Cap)", self.colors['robot']),
            ("Wall", self.colors['wall'])
        ]
        for text, col in items:
            f = tk.Frame(leg_frame, bg="#f0f0f0")
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, bg=col, width=2).pack(side=tk.LEFT, padx=(0,5))
            tk.Label(f, text=text, bg="#f0f0f0").pack(side=tk.LEFT)

    def load_problem(self, index):
        self.is_playing = False
        self.btn_play.config(text="▶ Play", bg="#2ecc71")
        
        # Deep copy problem data
        _, raw_prob = PROBLEM_LIST[index]
        self.problem_data = copy.deepcopy(raw_prob)
        
        # Pre-calculate the entire game history
        self._precompute_game()
        
        # Reset UI
        self.max_steps = len(self.history) - 1
        self.slider.config(to=self.max_steps)
        self.current_step_idx = 0
        self.slider_var.set(0)
        
        self.draw_current_step()

    def _precompute_game(self):
        """Runs the game fully in the background to generate history."""
        game = ext_plant.create_pressure_plate_game((self.problem_data, False))
        controller = ex3.Controller(game)
        self.capacities = game.get_capacities()
        
        self.history = []
        
        # Store Initial State
        # Format: (state_tuple, action_that_led_here, current_reward, step_count)
        init_state = game.get_current_state()
        self.history.append((init_state, "Start", 0, 0))
        
        max_s = game.get_max_steps()
        
        for _ in range(max_s):
            if game.get_done():
                break
            
            try:
                state = game.get_current_state()
                action = controller.choose_next_action(state)
                game.submit_next_action(action)
                
                # Snapshot
                self.history.append((
                    game.get_current_state(),
                    action,
                    game.get_current_reward(),
                    game.get_current_steps()
                ))
            except Exception as e:
                print(f"Simulation Error: {e}")
                break

    # --- Playback Logic ---

    def on_slider_move(self, val):
        self.current_step_idx = int(val)
        self.draw_current_step()

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="▶ Play", bg="#2ecc71")
        else:
            if self.current_step_idx >= self.max_steps:
                self.current_step_idx = 0 # Restart if at end
            self.is_playing = True
            self.btn_play.config(text="⏸ Pause", bg="#e74c3c")
            self.run_animation()

    def run_animation(self):
        if not self.is_playing: return
        
        if self.current_step_idx < self.max_steps:
            self.current_step_idx += 1
            self.slider_var.set(self.current_step_idx)
            self.draw_current_step()
            ms = self.speed_scale.get()
            self.root.after(ms, self.run_animation)
        else:
            self.is_playing = False
            self.btn_play.config(text="▶ Play", bg="#2ecc71")

    def step_fwd(self):
        if self.current_step_idx < self.max_steps:
            self.current_step_idx += 1
            self.slider_var.set(self.current_step_idx)
            self.draw_current_step()

    def step_back(self):
        if self.current_step_idx > 0:
            self.current_step_idx -= 1
            self.slider_var.set(self.current_step_idx)
            self.draw_current_step()

    # --- Drawing Logic ---

    def draw_current_step(self, event=None):
        if not self.history: return
        self.canvas.delete("all")
        
        # Get data for this frame
        state, last_act, reward, step = self.history[self.current_step_idx]
        robots, plants, taps, need = state
        
        # Update Stats Panel
        self.lbl_step.config(text=f"Step: {step} / {self.max_steps}")
        self.lbl_action.config(text=f"Last Action: {last_act}")
        self.lbl_reward.config(text=f"Total Reward: {reward:.2f}")
        self.lbl_need.config(text=f"Remaining Need: {need}")

        # Parse State for easy lookup
        robot_map = {r[1]: (r[0], r[2]) for r in robots} # pos -> (id, load)
        plant_map = {p[0]: p[1] for p in plants}         # pos -> need
        tap_map = {t[0]: t[1] for t in taps}             # pos -> water
        walls = self.problem_data.get("Walls", set())

        # Geometry
        rows, cols = self.problem_data["Size"]
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # padding
        pad_x = 20
        pad_y = 20
        avail_w = w - (2 * pad_x)
        avail_h = h - (2 * pad_y)

        cell_size = min(avail_w / cols, avail_h / rows)
        
        grid_w = cell_size * cols
        grid_h = cell_size * rows
        
        start_x = (w - grid_w) / 2
        start_y = (h - grid_h) / 2

        # Drawing Loop
        for r in range(rows):
            for c in range(cols):
                x1 = start_x + c * cell_size
                y1 = start_y + r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                pos = (r, c)
                
                # 1. Base Layer (Walls, Taps, Plants, Empty)
                if pos in walls:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.colors['wall'], outline="white")
                
                elif pos in tap_map:
                    # Draw Tap Floor
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.colors['tap_bg'], outline=self.colors['tap_border'], width=2)
                    # Text Top Center
                    self.canvas.create_text((x1+x2)/2, y1 + (cell_size * 0.2), 
                                            text=f"Tap\nWater: {tap_map[pos]}", 
                                            fill="#000000", font=("Arial", int(cell_size/8), "bold"))
                    
                elif pos in plant_map:
                    # Draw Plant Floor
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.colors['plant_bg'], outline=self.colors['plant_border'], width=2)
                    # Text Top Center
                    self.canvas.create_text((x1+x2)/2, y1 + (cell_size * 0.2), 
                                            text=f"Plant\nNeed: {plant_map[pos]}", 
                                            fill="#000000", font=("Arial", int(cell_size/8), "bold"))
                else:
                    # Empty Floor
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.colors['empty'], outline="#bdc3c7")

                # 2. Object Layer (Robots)
                # Robot sits on TOP of Taps/Plants
                if pos in robot_map:
                    rid, load = robot_map[pos]
                    cap = self.capacities.get(rid, "?")
                    
                    # Robot is a circle in the center, taking up 60% of cell
                    margin = cell_size * 0.2
                    rx1, ry1 = x1 + margin, y1 + margin
                    rx2, ry2 = x2 - margin, y2 - margin
                    
                    # Shift down slightly if on a tap/plant so we don't cover the label
                    if pos in tap_map or pos in plant_map:
                        offset = cell_size * 0.15
                        rx1 += 0; rx2 += 0
                        ry1 += offset; ry2 += offset

                    self.canvas.create_oval(rx1, ry1, rx2, ry2, fill=self.colors['robot'], outline=self.colors['robot_border'], width=3)
                    
                    # Robot Stats Text inside the circle
                    self.canvas.create_text((rx1+rx2)/2, (ry1+ry2)/2,
                                            text=f"Bot {rid}\n{load}/{cap}",
                                            fill="white", font=("Arial", int(cell_size/9), "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantWateringGUI(root)
    root.mainloop()