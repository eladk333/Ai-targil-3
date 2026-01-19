import ext_plant
import collections
import time
import random
import math

# ID for submission
id = ["322587064"]

class Controller:
    """
    RL-Adaptive Controller for Assignment 3.
    Adapts Ex2 planning logic to handle hidden probabilities and rewards.
    Now includes dynamic probability estimation based on observed action outcomes.
    """

    def __init__(self, game: ext_plant.Game):
        """Initialize controller for given game model."""
        self.game_ref = game
        self.problem_config = game.get_problem()
        
        # --- 1. Problem Parsing ---
        self.grid_h, self.grid_w = self.problem_config["Size"]
        self.walls = set(self.problem_config.get("Walls", []))
        self.goal_bonus = self.problem_config["goal_reward"]
        self.time_limit = self.problem_config["horizon"]
        self.max_caps = game.get_capacities()

        # --- 2. Handling Hidden Information (Assignment 3 Changes) ---
        self.plant_values = game.get_plants_max_reward()
        
        # Initialize Robot Statistics for Probability Estimation
        # Structure: {robot_id: {'success': count, 'total': count}}
        # We start with a "Prior" belief. 
        # Being slightly optimistic (Beta(4, 1) -> 0.8) helps exploration early on.
        self.robot_ids = list(self.problem_config["Robots"].keys())
        self.robot_stats = {rid: {'success': 4, 'total': 5} for rid in self.robot_ids}
        
        # State tracking for learning
        self.last_state = None
        self.last_action = None

        # --- 3. Pre-computations ---
        self.targets = set(self.problem_config.get("Plants", {}).keys()) | \
                       set(self.problem_config.get("Taps", {}).keys())
        self.initial_state = self._parse_initial_state()
        
        # Add initial robot positions to targets for BFS
        for r_info in self.initial_state[0]:
            self.targets.add(r_info[1])

        self.dist_cache = {}
        for t in self.targets:
            self.dist_cache[t] = self._generate_flood_map(t)

        # --- 4. Strategy Initialization ---
        self.initial_needs = self.problem_config.get("Plants", {}).copy()
        self.memo_table = {}
        self.dynamic_initialized = False
        self.reset_threshold = 0.15

    def choose_next_action(self, state):
        """
        Decides the next action using Iterative Deepening Expectimax
        based on the dynamically estimated model.
        """
        # 0. Update Model based on previous action's result
        if self.last_state and self.last_action and self.last_action != "RESET":
            self._update_model(self.last_state, self.last_action, state)

        # 1. Calibration (Run once)
        if not self.dynamic_initialized:
            self._calibrate_thresholds()
            self.dynamic_initialized = True

        step = self.game_ref.get_current_steps()
        time_rem = self.time_limit - step
        
        # 2. Reset Logic
        if self._should_trigger_reset(state, time_rem):
            self.last_action = "RESET"
            self.last_state = state
            return "RESET"

        # 3. Algorithm Selection
        decision = self._search_efficiency_strategy(state, time_rem)
        
        # 4. Save state for next learning step
        self.last_action = decision
        self.last_state = state
        
        return decision

    # =========================================================================
    #       LEARNING & PROBABILITY ESTIMATION
    # =========================================================================

    def _get_estimated_prob(self, rid):
        """Returns the smoothed probability of success for a specific robot."""
        stats = self.robot_stats.get(rid, {'success': 4, 'total': 5})
        # Laplace smoothing / Beta mean
        return stats['success'] / stats['total']

    def _update_model(self, prev_state, action, curr_state):
        """
        Compares previous state and current state to determine if the action
        succeeded or failed, then updates the robot's statistics.
        """
        try:
            prev_robots, prev_plants, prev_taps, prev_total_need = prev_state
            curr_robots, curr_plants, curr_taps, curr_total_need = curr_state

            # --- FIX: DETECT AUTOMATIC RESET ---
            # If the total need INCREASED, the environment reset automatically.
            # We cannot use this transition to estimate probabilities because 
            # the state change wasn't caused by physics, but by the game rules.
            if curr_total_need > prev_total_need:
                return
            # -----------------------------------

            parts = action.split()
            atype = parts[0]
            rid = int(parts[1].strip("()"))
            
            # Helper to get specific robot/plant data
            def get_r(s_robots, r_id):
                return next(r for r in s_robots if r[0] == r_id)
            
            r_prev = get_r(prev_robots, rid)
            r_curr = get_r(curr_robots, rid)
            
            success = False
            
            if atype in ["UP", "DOWN", "LEFT", "RIGHT"]:
                # Success: Robot moved to the intended cell
                dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[atype]
                expected_pos = (r_prev[1][0] + dr, r_prev[1][1] + dc)
                if r_curr[1] == expected_pos:
                    success = True
                else:
                    success = False 
                    
            elif atype == "LOAD":
                # Success: Load increased
                if r_curr[2] > r_prev[2]:
                    success = True
                else:
                    success = False
                    
            elif atype == "POUR":
                # Success: Plant need decreased
                r_pos = r_prev[1]
                p_prev_need = next((p[1] for p in prev_plants if p[0] == r_pos), 0)
                p_curr_need = next((p[1] for p in curr_plants if p[0] == r_pos), 0)
                
                # If plant disappeared (need 0) or need reduced
                plant_reduced = p_curr_need < p_prev_need
                
                if plant_reduced:
                    success = True
                else:
                    success = False

            # Update Stats
            self.robot_stats[rid]['total'] += 1
            if success:
                self.robot_stats[rid]['success'] += 1
                
        except Exception:
            pass

    # =========================================================================
    #       CORE SEARCH ALGORITHMS
    # =========================================================================

    def _search_efficiency_strategy(self, state, time_left):
        """
        Iterative Deepening Search.
        """
        valid_moves = self._get_valid_moves(state)
        if not valid_moves: return "RESET"
        
        candidates = self._prune_moves(state, valid_moves, time_left)
        if not candidates: return "RESET"

        t_start = time.time()
        # Time Management: Allocate time proportional to horizon urgency
        t_limit = 0.5 + (20.0 / self.time_limit)
        
        best_acts = []
        
        try:
            # Iterative Deepening depth 1 to 20
            for depth in range(1, 20):
                if (time.time() - t_start) > t_limit: break
                
                curr_best = []
                curr_max = float('-inf')
                completed_level = True
                
                for action in candidates:
                    # Timeout check
                    if (time.time() - t_start) > (t_limit * 1.5) and depth > 2: 
                        completed_level = False
                        break
                    
                    # Call Evaluation
                    val = self._calculate_node_value(state, action, depth, time_left, t_start, t_limit)
                    
                    if val > curr_max:
                        curr_max = val
                        curr_best = [action]
                    elif val == curr_max:
                        curr_best.append(action)
                
                if completed_level and curr_best: 
                    best_acts = curr_best
                
                if not completed_level: break
                
        except TimeoutError:
            pass
            
        if not best_acts: return "RESET"
        
        pours = [a for a in best_acts if "POUR" in a]
        return random.choice(pours) if pours else random.choice(best_acts)

    def _calculate_node_value(self, state, action, depth, time_left, t_start, t_limit):
        """
        Calculates the Expectimax value of a specific action node.
        """
        if (time.time() - t_start) > (t_limit * 1.5): raise TimeoutError()

        # Shallow depth optimization: treat high-prob robots as deterministic for speed
        rid = -1
        try:
            if "RESET" not in action:
                rid = int(action.split()[1].strip("()"))
        except: pass

        prob = self._get_estimated_prob(rid) if rid != -1 else 1.0
        
        # If very confident or shallow, run deterministic simulation
        is_det = (depth <= 1) or (prob > 0.95 and depth < 4)
        
        transitions = self._simulate_transition(state, action, deterministic=is_det)
        avg_score = 0
        
        for next_s, p, r in transitions:
            recurse_val = self._recursive_expectimax(next_s, depth - 1, time_left - 1, t_start, t_limit)
            avg_score += p * (r + (0.999 * recurse_val))
            
        return avg_score

    def _recursive_expectimax(self, state, depth, time_left, t_start, t_limit):
        if (time.time() - t_start) > (t_limit * 1.2): raise TimeoutError()
        
        key = (state, depth, time_left)
        if key in self.memo_table: return self.memo_table[key]

        _, _, _, total_need = state
        
        if total_need == 0: return self.goal_bonus + 1000
        if time_left <= 0: return -1000
        
        if depth == 0: return self._heuristic_utility(state, time_left)

        valid = self._get_valid_moves(state)
        if not valid: return self._heuristic_utility(state, time_left)
        
        cands = self._prune_moves(state, valid, time_left)
        if not cands: return self._heuristic_utility(state, time_left)
        
        max_v = float('-inf')
        for act in cands:
            v = self._calculate_node_value(state, act, depth, time_left, t_start, t_limit)
            if v > max_v: max_v = v
            
        self.memo_table[key] = max_v
        return max_v

    # =========================================================================
    #       HEURISTICS & EVALUATION
    # =========================================================================

    def _heuristic_utility(self, state, time_left):
        robots, plants, taps, total_need = state
        score = 0
        
        active_taps = [pos for pos, amt in taps if amt > 0]
        max_dist_obs = 0
        dist_penalty = 0

        # --- 1. SEPARATION PENALTY ---
        robot_positions = [r[1] for r in robots]
        for i in range(len(robot_positions)):
            for j in range(i + 1, len(robot_positions)):
                r1 = robot_positions[i]
                r2 = robot_positions[j]
                d_bots = abs(r1[0] - r2[0]) + abs(r1[1] - r2[1])
                if d_bots <= 1: score -= 2000.0
                elif d_bots <= 2: score -= 500.0

        # --- 2. CORRIDOR & GATE BLOCKAGE DETECTION (FIXED) ---
        robot_map = {r[1]: r[0] for r in robots}
        
        for rid, r_pos, load in robots:
            # Only apply to the "Main Carrier"
            if load < 5: continue 

            # Find target
            target_plant = None
            min_d = 999
            for p_pos, need in plants:
                if need > 0:
                    d = self._get_dist(r_pos, p_pos)
                    if d < min_d:
                        min_d = d
                        target_plant = p_pos
            
            if target_plant:
                # A. STRICT LEFT CORRIDOR LOGIC (For New 3 Map)
                # If target is in Col 0 (The Tunnel), checks if ANYONE else is in Col 0 or Col 1.
                if target_plant[1] == 0:
                    tunnel_occupied = False
                    for other_r_pos in robot_map:
                        # If another robot is in Col 0 or Col 1
                        if other_r_pos != r_pos and other_r_pos[1] <= 1:
                            tunnel_occupied = True
                            break
                    
                    # If tunnel is busy, I MUST stay back in Col 2 or higher.
                    # Entering Col 1 (The Gate) or Col 0 is forbidden.
                    if tunnel_occupied:
                        if r_pos[1] <= 1:
                            score -= 5000.0 # GET OUT OF THE GATE!
                
                # B. Standard Line-of-Sight Blockage (For other maps)
                # Vertical
                if r_pos[1] == target_plant[1]: 
                    min_r, max_r = min(r_pos[0], target_plant[0]), max(r_pos[0], target_plant[0])
                    for r_check in range(min_r + 1, max_r):
                        if (r_check, r_pos[1]) in robot_map:
                            score -= 5000.0
                # Horizontal
                elif r_pos[0] == target_plant[0]: 
                    min_c, max_c = min(r_pos[1], target_plant[1]), max(r_pos[1], target_plant[1])
                    for c_check in range(min_c + 1, max_c):
                        if (r_pos[0], c_check) in robot_map:
                            score -= 5000.0
        # -------------------------------------------
        
        # --- 3. DISTANCE & EVACUATION ---
        for p_pos, need in plants:
            if need <= 0: continue
            p_val = self.plant_values.get(p_pos, 0)
            
            min_dist = 999
            for rid, r_pos, load in robots:
                # Evacuation Logic
                cap = self.max_caps.get(rid, 0)
                if total_need > 5 and cap < 3:
                    safe_row = 5
                    dist_to_safe = abs(r_pos[0] - safe_row)
                    score -= (dist_to_safe * 200.0) 
                    continue

                curr_dist = 999
                if load > 0:
                    curr_dist = self._get_dist(r_pos, p_pos)
                elif active_taps:
                    to_tap = 999
                    best_tap = None
                    for t in active_taps:
                        d = self._get_dist(r_pos, t)
                        if d < to_tap:
                            to_tap = d
                            best_tap = t
                    if best_tap:
                        curr_dist = to_tap + self._get_dist(best_tap, p_pos)
                
                if curr_dist < min_dist:
                    min_dist = curr_dist
            
            if min_dist >= 900: dist_penalty += 1000
            else:
                dist_penalty += min_dist
                if min_dist > max_dist_obs: max_dist_obs = min_dist
            
            score += (need * p_val)

        score -= (total_need * 50)
        metric = dist_penalty + max_dist_obs
        score -= (metric * 2.0)
        
        # --- 4. RESOURCE HEURISTIC ---
        hoard_val = 25.0
        target_need = total_need 
        if self.reset_threshold <= 0.05:
            hoard_val = 15.0
            if any(r[1] in [t[0] for t in taps] for r in robots):
                 target_need = max(target_need, 20) 

        for rid, r_pos, load in robots:
            prob = self._get_estimated_prob(rid)
            usable = min(load, time_left)
            usable = min(usable, target_need)
            score += usable * hoard_val * prob
            if load > target_need:
                score -= 50.0 * (load - target_need)

        return score

    # =========================================================================
    #       HELPER METHODS & INIT
    # =========================================================================

    def _calibrate_thresholds(self):
        """
        Determines farming mode using initial estimates.
        """
        # 1. Estimate General Mission Efficiency
        total_pot = sum(n * self.plant_values.get(p, 0) for p, n in self.initial_needs.items())
        fleet_cap = sum(self.max_caps.get(rid, 0) for rid in self.robot_ids) or 1
        avg_dist = (self.grid_h + self.grid_w) / 2.0
        tot_need = sum(self.initial_needs.values())
        
        steps_est = (tot_need / fleet_cap) * 2 * avg_dist + (tot_need * 2)
        est_miss_eff = (self.goal_bonus + total_pot) / (max(steps_est, 1) * 1.2)
        
        # 2. Estimate Farming Efficiency
        max_farm_eff = 0
        self.focus_need = 999
        self.focus_robot = -1
        
        for p_pos, need in self.initial_needs.items():
            val = self.plant_values.get(p_pos, 0)
            tap_dist = avg_dist
            if p_pos in self.dist_cache:
                dists = [self.dist_cache[p_pos].get(t[0], 999) for t in self.initial_state[2]]
                if dists: tap_dist = min(dists)

            for rid in self.robot_ids:
                rcap = self.max_caps.get(rid, 1) or 1
                cycles = max(need / rcap, 1)
                steps = 3 + (cycles * 2 * tap_dist) + (2 * need)
                
                # Use current estimate
                prob = self._get_estimated_prob(rid)
                exp_steps = steps / prob
                
                eff = (val * need) / exp_steps
                
                if eff > max_farm_eff:
                    max_farm_eff = eff
                    self.focus_need = need
                    self.focus_robot = rid

        if max_farm_eff > est_miss_eff:
            self.reset_threshold = 0.05
        else:
            self.reset_threshold = 0.15
            
        self.baseline_route_val = self._get_best_efficiency(self.initial_state, self.time_limit)[0]

    def _should_trigger_reset(self, state, time_left):
        min_safe_horizon = (self.grid_h + self.grid_w) * 1.5 
        if time_left < min_safe_horizon: 
            return False
        
        reset_eff, _ = self._get_best_efficiency(self.initial_state, time_left - 1)
        curr_eff, _ = self._get_best_efficiency(state, time_left)
        
        loaded = sum(r[2] for r in state[0])
        almost_done = loaded >= (sum(self.initial_needs.values()) * 0.5)
        
        better_reset = False
        if curr_eff == -1:
            better_reset = True
        elif reset_eff > (curr_eff + self.reset_threshold):
            better_reset = True
            
        dire = curr_eff < (reset_eff * 0.33)
        feasible = (time_left > 10) if self.reset_threshold > 0.1 else (time_left > 5)
        
        if self.reset_threshold > 0.1 and not dire and (curr_eff > reset_eff * 0.8):
            better_reset = False
            
        if better_reset and feasible:
            if not almost_done or curr_eff == -1:
                if state != self.initial_state:
                     return True
        return False

    def _get_best_efficiency(self, state, time_left):
        """
        Estimates the best possible efficiency (Reward / Step) achievable from
        the current state given the time remaining.
        """
        robots, plants, taps, _ = state
        active_taps = [p for p, a in taps if a > 0]
        
        best_e = -1.0
        best_a = None
        
        for rid, r_pos, load in robots:
            cap = self.max_caps.get(rid, 0)
            # Dynamic Prob
            prob = self._get_estimated_prob(rid)

            for p_pos, need in plants:
                if need <= 0: continue
                avg = self.plant_values.get(p_pos, 0)
                
                # Case 1: Deliver current load
                if load > 0:
                    dist = self._get_dist(r_pos, p_pos)
                    amt = min(need, load)
                    steps = dist + amt
                    # Adjust steps by probability to get expected time cost
                    eff_steps = steps / prob
                    
                    if steps <= time_left and steps < 900:
                        rew = amt * avg
                        # Goal reward approximation
                        if (state[3] - amt) <= 0: rew += self.goal_bonus
                        
                        eff = rew / (eff_steps + 2.2) # +2.2 is a heuristic base cost
                        if eff > best_e:
                            best_e = eff
                            best_a = f"POUR ({rid})" if dist == 0 else self._move_towards(r_pos, p_pos, rid)

                # Case 2: Fill at tap then Deliver
                if load < cap and active_taps:
                    min_t_dist = 999
                    best_t = None
                    for t in active_taps:
                        d = self._get_dist(r_pos, t)
                        if d < min_t_dist:
                            min_t_dist = d
                            best_t = t
                    
                    if best_t:
                        max_l = min(cap - load, 99)
                        d_plant = self._get_dist(best_t, p_pos)
                        steps_fix = min_t_dist + d_plant
                        
                        feasible = (time_left - steps_fix) // 2
                        if feasible >= 1:
                            amt_load = min(max_l, feasible)
                            final_l = load + amt_load
                            final_del = min(need, final_l)
                            
                            steps_tot = min_t_dist + amt_load + d_plant + final_del
                            eff_steps = steps_tot / prob
                            
                            if steps_tot <= time_left and steps_tot < 900:
                                rew = final_del * avg
                                if (state[3] - final_del) <= 0: rew += self.goal_bonus
                                
                                eff = rew / (eff_steps + 2.2)
                                if eff > best_e:
                                    best_e = eff
                                    best_a = f"LOAD ({rid})" if min_t_dist == 0 else self._move_towards(r_pos, best_t, rid)
        return best_e, best_a

    # =========================================================================
    #       TRANSITION & STATE LOGIC
    # =========================================================================

    def _simulate_transition(self, state, action, deterministic=False):
        if action == "RESET": return [(self.initial_state, 1.0, 0)]
        
        parts = action.split()
        atype = parts[0]
        rid = int(parts[1].strip("()"))
        
        # Use Learned Probability
        prob = self._get_estimated_prob(rid)
        
        outcomes = []
        
        # Success Case
        s_succ, r_succ = self._apply_state_change(state, rid, atype, True)
        if deterministic:
             return [(s_succ, 1.0, r_succ)]
        
        outcomes.append((s_succ, prob, r_succ))
        
        # Fail Case
        p_fail = 1.0 - prob
        if p_fail > 0:
            if atype in ["UP", "DOWN", "LEFT", "RIGHT"]:
                valid = self._get_robot_moves_only(state, rid)
                others = [m for m in valid if m != atype]
                others.append("STAY")
                
                sub_p = p_fail / len(others)
                for fa in others:
                    if fa == "STAY":
                        outcomes.append((state, sub_p, 0))
                    else:
                        sf, _ = self._apply_state_change(state, rid, fa, True)
                        outcomes.append((sf, sub_p, 0))
            elif atype == "POUR":
                sf, _ = self._apply_state_change(state, rid, atype, False)
                outcomes.append((sf, p_fail, 0))
            elif atype == "LOAD":
                outcomes.append((state, p_fail, 0))
                
        return outcomes

    def _apply_state_change(self, state, rid, atype, success):
        robots, plants, taps, tot_need = state
        rl = list(robots)
        pl = list(plants)
        tl = list(taps)
        
        idx = next(i for i, r in enumerate(rl) if r[0] == rid)
        _, (r, c), load = rl[idx]
        rew = 0
        
        if atype in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[atype]
            rl[idx] = (rid, (r+dr, c+dc), load)
            
        elif atype == "LOAD" and success:
            t_idx = next(i for i, t in enumerate(tl) if t[0] == (r,c))
            tp, tamt = tl[t_idx]
            if tamt - 1 <= 0: del tl[t_idx]
            else: tl[t_idx] = (tp, tamt - 1)
            rl[idx] = (rid, (r,c), load + 1)
            
        elif atype == "POUR":
            if success:
                pidx = -1
                for i, p in enumerate(pl):
                    if p[0] == (r, c):
                        pidx = i
                        break
                
                if pidx != -1:
                    ppos, pneed = pl[pidx]
                    val = self.plant_values.get(ppos, 0)
                    rew = val
                    
                    if pneed - 1 <= 0: del pl[pidx]
                    else: pl[pidx] = (ppos, pneed - 1)
                    tot_need -= 1
                    rl[idx] = (rid, (r,c), load - 1)
                else:
                    rl[idx] = (rid, (r,c), load - 1)
            else:
                rl[idx] = (rid, (r,c), load - 1)

        rl.sort(key=lambda x: x[0])
        pl.sort(key=lambda x: x[0])
        tl.sort(key=lambda x: x[0])
        
        return (tuple(rl), tuple(pl), tuple(tl), tot_need), rew

    # =========================================================================
    #       UTILITIES
    # =========================================================================

    def _parse_initial_state(self):
        r_list = []
        for rid, d in self.problem_config["Robots"].items():
            r_list.append((rid, (d[0], d[1]), d[2]))
        r_list.sort(key=lambda x: x[0])
        
        p_list = []
        for pos, n in self.problem_config["Plants"].items():
            p_list.append((pos, n))
        p_list.sort(key=lambda x: x[0])
        
        t_list = []
        for pos, w in self.problem_config["Taps"].items():
            t_list.append((pos, w))
        t_list.sort(key=lambda x: x[0])
        
        tot = sum(x[1] for x in p_list)
        return (tuple(r_list), tuple(p_list), tuple(t_list), tot)

    def _generate_flood_map(self, start):
        q = collections.deque([(start, 0)])
        visited = {start: 0}
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            (r, c), dist = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
                    if (nr, nc) not in self.walls and (nr, nc) not in visited:
                        visited[(nr, nc)] = dist + 1
                        q.append(((nr, nc), dist + 1))
        return visited

    def _get_dist(self, start, end):
        if end in self.dist_cache and start in self.dist_cache[end]:
            return self.dist_cache[end][start]
        return 999 

    def _move_towards(self, curr, target, rid):
        for act, (dr, dc) in { "UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1) }.items():
             nr, nc = curr[0]+dr, curr[1]+dc
             if self._get_dist((nr, nc), target) < self._get_dist(curr, target):
                 return f"{act} ({rid})"
        return None

    def _prune_moves(self, state, actions, time_left):
        """Filters out inefficient moves."""
        # Special farming filter
        if self.reset_threshold <= 0.05 and hasattr(self, 'focus_robot'):
            temp = []
            for a in actions:
                if a == "RESET": 
                    temp.append(a)
                    continue
                parts = a.split()
                if len(parts) > 1:
                    rid = int(parts[1].strip("()"))
                    if rid != self.focus_robot:
                        if parts[0] == "LOAD": continue
                    temp.append(a)
            if temp: actions = temp

        robots, plants, taps, _ = state
        rmap = {r[0]: (r[1], r[2]) for r in robots}
        
        grouped = {}
        active_p = [p for p, n in plants]
        
        # Calculate min dists
        r_dists = {}
        for rid, (pos, _) in rmap.items():
            md = 999
            for p in active_p:
                d = self._get_dist(pos, p)
                if d < md: md = d
            r_dists[rid] = md

        for a in actions:
            if a == "RESET": continue
            parts = a.split()
            atype = parts[0]
            rid = int(parts[1].strip("()"))
            
            if rid not in rmap: continue
            pos, load = rmap[rid]
            
            if rid not in grouped: grouped[rid] = []
            
            if atype == "LOAD":
                if load >= time_left: continue
                if (load + r_dists.get(rid, 999)) >= time_left: continue
                grouped[rid].append((a, -1))
                continue
            
            if atype == "POUR":
                grouped[rid].append((a, -1))
                continue
            
            targets = active_p if load > 0 else [t for t, amt in taps if amt > 0]
            if not targets:
                grouped[rid].append((a, 0))
                continue
                
            dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[atype]
            npos = (pos[0]+dr, pos[1]+dc)
            
            md = 999
            for t in targets:
                d = self._get_dist(npos, t)
                if d < md: md = d
            grouped[rid].append((a, md))

        final = []
        for rid, moves in grouped.items():
            moves.sort(key=lambda x: x[1])
            if moves:
                best = moves[0][1]
                for act, d in moves:
                    if d <= best: final.append(act)
        
        def rank(act):
            if act == "RESET": return -999
            try:
                rid = int(act.split()[1].strip("()"))
                if rid not in rmap: return 0
                _, l = rmap[rid]
                c = self.max_caps.get(rid, 1)
                # Use Learned Prob
                p = self._get_estimated_prob(rid)
                return l + (c * p)
            except: return 0

        final.sort(key=rank, reverse=True)
        return final

    def _get_valid_moves(self, state):
        robots, plants, taps, _ = state
        legal = []
        occ = {pos for _, pos, _ in robots}
        plocs = {pos for pos, _ in plants}
        tlocs = {pos for pos, _ in taps}
        
        for rid, (r, c), load in robots:
            # Moves
            for act, (dr, dc) in { "UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1) }.items():
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w and (nr, nc) not in self.walls:
                    if (nr, nc) not in occ:
                        legal.append(f"{act} ({rid})")
            
            # Load
            if (r, c) in tlocs and load < self.max_caps[rid]:
                legal.append(f"LOAD ({rid})")
            
            # Pour
            if (r, c) in plocs and load > 0:
                legal.append(f"POUR ({rid})")
        
        legal.append("RESET")
        return legal

    def _get_robot_moves_only(self, state, rid):
        robots, _, _, _ = state
        occ = {pos for r_id, pos, _ in robots if r_id != rid}
        curr = next(r for r in robots if r[0] == rid)
        r, c = curr[1]
        res = []
        for act, (dr, dc) in { "UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1) }.items():
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w and \
               (nr, nc) not in self.walls and (nr, nc) not in occ:
                res.append(act)
        return res