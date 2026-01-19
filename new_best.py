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
        # We cannot access 'robot_chosen_action_prob' or 'plants_reward' directly.
        
        # A. Reward Estimation: Use Max Reward as the utility proxy
        # The game provides a specific getter for this in Ex3.
        self.plant_values = game.get_plants_max_reward()
        
        # B. Probability Estimation: Assume a default model
        # Since we don't know who is broken, we assume a "Standard" success rate.
        # 0.85 is chosen as a balance between optimism (planning long paths) 
        # and caution (expecting some failure).
        self.assumed_prob = 0.85
        self.robot_ids = list(self.problem_config["Robots"].keys())
        
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
        based on the estimated model.
        """
        # 1. Calibration (Run once)
        if not self.dynamic_initialized:
            self._calibrate_thresholds()
            self.dynamic_initialized = True

        step = self.game_ref.get_current_steps()
        time_rem = self.time_limit - step
        
        # 2. Reset Logic
        # If we are performing significantly worse than the baseline efficiency, RESET.
        if self._should_trigger_reset(state, time_rem):
            return "RESET"

        # 3. Algorithm Selection
        # In Ex3, without knowing probs, we default to the Efficiency Strategy
        # which uses time-bound Iterative Deepening.
        return self._search_efficiency_strategy(state, time_rem)

    # =========================================================================
    #       CORE SEARCH ALGORITHMS
    # =========================================================================

    def _search_efficiency_strategy(self, state, time_left):
        """
        Iterative Deepening Search.
        Adapted for Ex3 to use assumed probabilities.
        """
        valid_moves = self._get_valid_moves(state)
        if not valid_moves: return "RESET"
        
        # Prune moves that are obviously bad (loading when full, etc)
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
                
                # Hard timeout
                if not completed_level: break
                
        except TimeoutError:
            pass
            
        if not best_acts: return "RESET"
        
        # Tie-breaker: Prefer POURing to immediate rewards
        pours = [a for a in best_acts if "POUR" in a]
        return random.choice(pours) if pours else random.choice(best_acts)

    def _calculate_node_value(self, state, action, depth, time_left, t_start, t_limit):
        """
        Calculates the Expectimax value of a specific action node.
        """
        # Timeout Check
        if (time.time() - t_start) > (t_limit * 1.5): raise TimeoutError()

        # Deterministic Shortcut:
        # If depth is shallow, treat as deterministic to save time.
        # In Ex3, since we assume 0.85, we can skip stochastic calc for depth 1.
        is_det = (depth <= 1)
        
        transitions = self._simulate_transition(state, action, deterministic=is_det)
        avg_score = 0
        
        for next_s, p, r in transitions:
            # Recursive step
            recurse_val = self._recursive_expectimax(next_s, depth - 1, time_left - 1, t_start, t_limit)
            
            # Bellman Equation with decay
            avg_score += p * (r + (0.999 * recurse_val))
            
        return avg_score

    def _recursive_expectimax(self, state, depth, time_left, t_start, t_limit):
        """
        Recursive helper for Expectimax.
        """
        if (time.time() - t_start) > (t_limit * 1.2): raise TimeoutError()
        
        # Memoization
        key = (state, depth, time_left)
        if key in self.memo_table: return self.memo_table[key]

        _, _, _, total_need = state
        
        # Terminal States
        if total_need == 0: return self.goal_bonus + 1000
        if time_left <= 0: return -1000
        
        # Leaf Node
        if depth == 0: return self._heuristic_utility(state, time_left)

        valid = self._get_valid_moves(state)
        if not valid: return self._heuristic_utility(state, time_left)
        
        # Pruning inside recursion
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
        """
        Estimates the value of a state based on distance and potential reward.
        Modified for Ex3 to use estimated Probabilities and Max Rewards.
        """
        robots, plants, taps, total_need = state
        score = 0
        
        active_taps = [pos for pos, amt in taps if amt > 0]
        
        dist_penalty = 0
        max_dist_obs = 0
        
        for p_pos, need in plants:
            if need <= 0: continue
            # Use MAX REWARD as the heuristic value
            p_val = self.plant_values.get(p_pos, 0)
            
            min_dist = 999
            
            for rid, r_pos, load in robots:
                curr_dist = 999
                if load > 0:
                    curr_dist = self._get_dist(r_pos, p_pos)
                elif active_taps:
                    # Dist to tap + Dist to plant
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
            
            # Penalize long distances heavily
            if min_dist >= 900:
                dist_penalty += 1000
            else:
                dist_penalty += min_dist
                if min_dist > max_dist_obs: max_dist_obs = min_dist
            
            score += (need * p_val)

        score -= (total_need * 50)
        
        # Apply distance costs
        metric = dist_penalty + max_dist_obs
        score -= (metric * 2.0)
        
        # Hoarding Logic (Encourage carrying water)
        hoard_val = 25.0
        target_need = 999
        
        # "Farming Mode" adjustment (focusing heavily on one plant)
        if self.reset_threshold <= 0.05:
            hoard_val = 15.0
            target_need = getattr(self, 'focus_need', 999)
            if target_need == 0:
                 target_need = max(self.initial_needs.values()) if self.initial_needs else 999

        for rid, r_pos, load in robots:
            # Use Assumed Probability (0.85)
            prob = self.assumed_prob
            
            usable = min(load, time_left)
            usable = min(usable, target_need)
            
            r_val = hoard_val
            if self.reset_threshold <= 0.05 and hasattr(self, 'focus_robot'):
                if rid != self.focus_robot:
                    r_val = 0.0
            
            score += usable * r_val * prob
            
            # Over-hoarding penalty
            if self.reset_threshold <= 0.05:
                if load > target_need:
                    score -= 5000.0

        return score

    # =========================================================================
    #       HELPER METHODS & INIT
    # =========================================================================

    def _calibrate_thresholds(self):
        """
        Determines if we should enter "Farming Mode" (Focus one robot on one plant).
        Uses assumed probabilities and max rewards.
        """
        # 1. Estimate General Mission Efficiency
        total_pot = sum(n * self.plant_values.get(p, 0) for p, n in self.initial_needs.items())
        denom = self.goal_bonus + total_pot + 0.1
        
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
            
            # Estimate distance to nearest tap
            tap_dist = avg_dist
            if p_pos in self.dist_cache:
                dists = [self.dist_cache[p_pos].get(t[0], 999) for t in self.initial_state[2]]
                if dists: tap_dist = min(dists)

            for rid in self.robot_ids:
                rcap = self.max_caps.get(rid, 1) or 1
                cycles = max(need / rcap, 1)
                
                steps = 3 + (cycles * 2 * tap_dist) + (2 * need)
                
                # Use Assumed Probability
                prob = self.assumed_prob
                exp_steps = steps / prob
                
                eff = (val * need) / exp_steps
                
                if eff > max_farm_eff:
                    max_farm_eff = eff
                    self.focus_need = need
                    self.focus_robot = rid

        if max_farm_eff > est_miss_eff:
            self.reset_threshold = 0.05  # Trigger Farming Mode
        else:
            self.reset_threshold = 0.15  # Normal Mode
            
        # Store baseline for Reset checks
        self.baseline_route_val = self._get_best_efficiency(self.initial_state, self.time_limit)[0]

    def _should_trigger_reset(self, state, time_left):
        """Decides if current performance is so bad compared to baseline that we should RESET."""
        if time_left < 5: return False
        
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
        """Calculates (reward / steps) for best possible single delivery."""
        robots, plants, taps, _ = state
        active_taps = [p for p, a in taps if a > 0]
        
        best_e = -1.0
        best_a = None
        
        for rid, r_pos, load in robots:
            cap = self.max_caps.get(rid, 0)
            
            for p_pos, need in plants:
                if need <= 0: continue
                
                # Use MAX reward
                avg = self.plant_values.get(p_pos, 0)
                
                # Case 1: Deliver current load
                if load > 0:
                    dist = self._get_dist(r_pos, p_pos)
                    amt = min(need, load)
                    steps = dist + amt
                    
                    if steps <= time_left and steps < 900:
                        rew = amt * avg
                        if (state[3] - amt) <= 0: rew += self.goal_bonus
                        
                        eff = rew / (steps + 2.2)
                        if eff > best_e:
                            best_e = eff
                            best_a = f"POUR ({rid})" if dist == 0 else self._move_towards(r_pos, p_pos, rid)

                # Case 2: Fill then Deliver
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
                            if steps_tot <= time_left and steps_tot < 900:
                                rew = final_del * avg
                                if (state[3] - final_del) <= 0: rew += self.goal_bonus
                                
                                eff = rew / (steps_tot + 2.2)
                                if eff > best_e:
                                    best_e = eff
                                    best_a = f"LOAD ({rid})" if min_t_dist == 0 else self._move_towards(r_pos, best_t, rid)
        return best_e, best_a

    # =========================================================================
    #       TRANSITION & STATE LOGIC
    # =========================================================================

    def _simulate_transition(self, state, action, deterministic=False):
        """
        Simulates state transition.
        CRITICAL EX3 CHANGE: Uses self.assumed_prob instead of reading unknown props.
        """
        if action == "RESET": return [(self.initial_state, 1.0, 0)]
        
        parts = action.split()
        atype = parts[0]
        rid = int(parts[1].strip("()"))
        
        # Use Assumed Model Probability
        prob = self.assumed_prob
        
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
                # Find plant
                pidx = -1
                for i, p in enumerate(pl):
                    if p[0] == (r, c):
                        pidx = i
                        break
                
                if pidx != -1:
                    ppos, pneed = pl[pidx]
                    # We only know MAX reward in Ex3, so we use that for planning
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
    #       UTILITIES (Parsing, Pruning, BFS)
    # =========================================================================

    def _parse_initial_state(self):
        """Converts problem dict to state tuple."""
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
        """BFS Flood fill."""
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
        
        # Sort by load + expectation
        def rank(act):
            if act == "RESET": return -999
            try:
                rid = int(act.split()[1].strip("()"))
                if rid not in rmap: return 0
                _, l = rmap[rid]
                c = self.max_caps.get(rid, 1)
                # Use Assumed Prob
                p = self.assumed_prob
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
            # Note: We check if it's a valid plant location, even if reward list is hidden
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