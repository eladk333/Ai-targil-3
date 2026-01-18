import ext_plant
import collections
import time
import random

id = ["322587064"]

class ModelLearner:
    def __init__(self, robots_config, plants_config, plant_max_rewards):        
        self.robot_stats = {rid: {"success": 1, "total": 1} for rid in robots_config}
        self.plant_stats = {}
        for pos in plants_config:
            max_r = plant_max_rewards.get(pos, 0)
            self.plant_stats[pos] = {"sum": max_r, "count": 1}

    def get_robot_prob(self, rid):
        stats = self.robot_stats.get(rid, {"success": 1, "total": 1})
        return stats["success"] / stats["total"]

    def get_expected_reward(self, plant_pos):
        stats = self.plant_stats.get(plant_pos, {"sum": 0, "count": 0})
        if stats["count"] == 0: return 0
        return stats["sum"] / stats["count"]

    def update(self, prev_state, action_str, curr_state, reward_gained):
        if action_str == "RESET": return
        parts = action_str.split()
        act_type = parts[0]
        try: rid = int(parts[1].strip("()"))
        except: return 

        prev_robots = {r[0]: (r[1], r[2]) for r in prev_state[0]}
        curr_robots = {r[0]: (r[1], r[2]) for r in curr_state[0]}
        
        if rid not in prev_robots or rid not in curr_robots: return
        prev_pos, prev_load = prev_robots[rid]
        curr_pos, curr_load = curr_robots[rid]
        success = False
        
        if act_type in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[act_type]
            expected_pos = (prev_pos[0]+dr, prev_pos[1]+dc)
            if curr_pos == expected_pos: success = True
        elif act_type == "LOAD":
            if curr_load > prev_load: success = True
        elif act_type == "POUR":
            prev_plants = {p[0]: p[1] for p in prev_state[1]}
            curr_plants = {p[0]: p[1] for p in curr_state[1]}
            if prev_pos in prev_plants:
                prev_need = prev_plants[prev_pos]
                curr_need = curr_plants.get(prev_pos, 0)
                if curr_need < prev_need:
                    success = True
                    if reward_gained > 0:
                        if prev_pos not in self.plant_stats:
                             self.plant_stats[prev_pos] = {"sum": 0, "count": 0}
                        self.plant_stats[prev_pos]["sum"] += reward_gained
                        self.plant_stats[prev_pos]["count"] += 1

        self.robot_stats[rid]["total"] += 1
        if success: self.robot_stats[rid]["success"] += 1

class Controller:
    def __init__(self, game: ext_plant.Game):
        self.game_ref = game
        self.problem_config = game.get_problem()
        self.grid_h, self.grid_w = self.problem_config["Size"]
        self.walls = set(self.problem_config.get("Walls", []))
        self.goal_bonus = self.problem_config["goal_reward"]
        self.time_limit = self.problem_config["horizon"]
        self.max_caps = game.get_capacities()
        self.plant_max_rewards = game.get_plants_max_reward()

        self.learner = ModelLearner(
            self.problem_config["Robots"],
            self.problem_config["Plants"],
            self.plant_max_rewards
        )
        
        self.last_state = self.game_ref.get_current_state()
        self.last_action = None

        self.targets = set(self.problem_config.get("Plants", {}).keys()) | \
                       set(self.problem_config.get("Taps", {}).keys())
        self.initial_state = self._parse_initial_state()
        
        for r_info in self.initial_state[0]: self.targets.add(r_info[1])
        
        self.dist_cache = {}
        for t in self.targets: self.dist_cache[t] = self._generate_flood_map(t)
        self.memo_table = {}

    def choose_next_action(self, state):
        if self.last_action is not None:
            reward = self.game_ref.get_last_gained_reward()
            self.learner.update(self.last_state, self.last_action, state, reward)

        step = self.game_ref.get_current_steps()
        time_rem = self.time_limit - step
        
        next_action = self._search_efficiency_strategy(state, time_rem)

        self.last_state = state
        self.last_action = next_action
        return next_action

    def _search_efficiency_strategy(self, state, time_left):
        valid_moves = self._get_valid_moves(state)
        if not valid_moves: return "RESET"
        
        candidates = self._prune_moves(state, valid_moves, time_left)
        if not candidates: candidates = valid_moves

        def quick_score(action):
            rid = int(action.split()[1].strip("()"))
            atype = action.split()[0]
            next_s, _ = self._apply_state_change(state, rid, atype, True)
            return self._evaluate_state_utility(next_s, time_left - 1)
        
        candidates.sort(key=quick_score, reverse=True)

        t_start = time.time()
        t_limit = 0.4 + (20.0 / self.time_limit)
        
        best_acts = [candidates[0]]
        
        try:
            for depth in range(1, 4):
                if (time.time() - t_start) > t_limit: break
                
                curr_best = []
                curr_max = float('-inf')
                
                for action in candidates:
                    if (time.time() - t_start) > t_limit: raise TimeoutError
                    
                    val = self._calculate_node_value(state, action, depth, time_left)
                    if val > curr_max:
                        curr_max = val
                        curr_best = [action]
                    elif val == curr_max:
                        curr_best.append(action)
                
                if curr_best: best_acts = curr_best

        except TimeoutError:
            pass 
            
        if not best_acts: return random.choice(candidates)
        
        preferred = [a for a in best_acts if "POUR" in a or "LOAD" in a]
        return random.choice(preferred) if preferred else random.choice(best_acts)

    def _calculate_node_value(self, state, action, depth, time_left):
        outcomes = self._simulate_transition(state, action)
        avg_score = 0
        for next_s, prob, reward in outcomes:
            future_val = self._recursive_expectimax(next_s, depth - 1, time_left - 1)
            avg_score += prob * (reward + (0.95 * future_val))
        return avg_score

    def _recursive_expectimax(self, state, depth, time_left):
        if time_left <= 0: return -1000
        _, _, _, total_need = state
        if total_need == 0: return self.goal_bonus + 1000
        if depth == 0: return self._evaluate_state_utility(state, time_left)
        
        key = (state, depth, time_left)
        if key in self.memo_table: return self.memo_table[key]
        
        return self._evaluate_state_utility(state, time_left)

    def _evaluate_state_utility(self, state, time_left):
        robots, plants, taps, total_need = state
        score = -(total_need * 500.0)
        
        active_taps = [pos for pos, amt in taps if amt > 0]
        plant_target_counts = collections.defaultdict(int)

        for rid, r_pos, load in robots:
            prob = self.learner.get_robot_prob(rid)
            
            # --- CASE 1: Empty Robot (Needs Water) ---
            if load == 0:
                if not active_taps: 
                    score -= 1000 
                    continue
                
                min_dist = 999
                best_tap = None
                for t_pos in active_taps:
                    d = self._get_dist(r_pos, t_pos)
                    if d < min_dist: 
                        min_dist = d
                        best_tap = t_pos

                # --- REDUNDANCY CHECK ---
                # Before we encourage going to the tap, check if I am actually needed.
                # If another robot is strictly closer to satisfying the plants, I should yield.
                is_needed = False
                if best_tap:
                    for p_pos, p_need in plants:
                        my_total_dist = min_dist + self._get_dist(best_tap, p_pos)
                        
                        i_am_closest = True
                        for oid, opos, oload in robots:
                            if oid == rid: continue
                            
                            # Estimate other robot's distance to this plant
                            if oload > 0:
                                o_dist = self._get_dist(opos, p_pos)
                            else:
                                # They also need to go to tap
                                od_tap = 999
                                for ot in active_taps:
                                    dx = self._get_dist(opos, ot)
                                    if dx < od_tap: od_tap = dx
                                o_dist = od_tap + self._get_dist(best_tap, p_pos) # approx

                            # If they are strictly faster, I am redundant for this plant
                            if o_dist < my_total_dist:
                                i_am_closest = False
                                break
                        
                        if i_am_closest:
                            is_needed = True
                            break
                
                # If I'm redundant for ALL plants, don't reward movement
                if not is_needed:
                    continue 

                # If needed, encourage getting water
                score -= (min_dist * 8.0) 
            
            # --- CASE 2: Loaded Robot (Needs Plant) ---
            else:
                best_plant_val = -float('inf')
                target_plant = None
                
                for p_pos, need in plants:
                    if need <= 0: continue
                    dist = self._get_dist(r_pos, p_pos)
                    reward_est = self.learner.get_expected_reward(p_pos)
                    if reward_est == 0: reward_est = self.plant_max_rewards.get(p_pos, 5)
                    
                    val = (reward_est * 10.0) - (dist * 8.0)

                    # --- TERRITORY CLAIMING ---
                    # Check if another robot is STRICTLY closer to this plant
                    am_i_closest = True
                    for other_rid, other_pos, _ in robots:
                        if other_rid == rid: continue
                        d_other = self._get_dist(other_pos, p_pos)
                        if d_other < dist:
                            am_i_closest = False
                            break
                    
                    if not am_i_closest:
                        val -= 100.0

                    if val > best_plant_val:
                        best_plant_val = val
                        target_plant = p_pos

                if target_plant:
                    score += (best_plant_val * prob)
                    plant_target_counts[target_plant] += 1
                score += 20.0

        for p_pos, count in plant_target_counts.items():
            if count > 1:
                score -= 15.0

        return score

    def _prune_moves(self, state, actions, time_left):
        robots, plants, taps, total_need = state
        rmap = {r[0]: (r[1], r[2]) for r in robots}
        active_p = [p for p, n in plants]
        active_t = [t for t, amt in taps if amt > 0]
        
        grouped = collections.defaultdict(list)
        
        for a in actions:
            if a == "RESET": continue
            parts = a.split()
            atype = parts[0]
            rid = int(parts[1].strip("()"))
            if rid not in rmap: continue
            pos, load = rmap[rid]
            
            if atype == "LOAD":
                if load >= self.max_caps[rid]: continue
                if load >= total_need: continue
                
                my_valid_needs = []
                for p_pos, p_need in plants:
                    d_me = self._get_dist(pos, p_pos)
                    is_closest = True
                    for other_rid, other_r_pos, _ in robots:
                        if other_rid == rid: continue
                        d_other = self._get_dist(other_r_pos, p_pos)
                        if d_other < d_me:
                            is_closest = False
                            break
                    if is_closest:
                        my_valid_needs.append(p_need)
                
                if my_valid_needs:
                    max_req = max(my_valid_needs)
                    if load >= max_req: continue
                elif load > 0:
                    continue

                grouped[rid].append((a, -999)) 
                continue
            
            if atype == "POUR":
                grouped[rid].append((a, -999))
                continue
            
            targets = active_t if load == 0 else active_p
            if not targets:
                grouped[rid].append((a, 0))
                continue
            
            dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[atype]
            npos = (pos[0]+dr, pos[1]+dc)
            
            min_dist = 999
            for t in targets:
                d = self._get_dist(npos, t)
                if d < min_dist: min_dist = d
            grouped[rid].append((a, min_dist))

        final = []
        for rid, moves in grouped.items():
            if not moves: continue
            moves.sort(key=lambda x: x[1])
            best_dist = moves[0][1]
            for act, d in moves:
                if d <= best_dist: 
                    final.append(act)
                
        return final

    def _get_valid_moves(self, state):
        robots, plants, taps, _ = state
        legal = []
        occ = {pos for _, pos, _ in robots}
        plocs = {pos for pos, _ in plants}
        tlocs = {pos for pos, _ in taps}
        
        for rid, (r, c), load in robots:
            for act, (dr, dc) in { "UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1) }.items():
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w and (nr, nc) not in self.walls:
                    if (nr, nc) not in occ: legal.append(f"{act} ({rid})")
            if (r, c) in tlocs and load < self.max_caps[rid]: legal.append(f"LOAD ({rid})")
            if (r, c) in plocs and load > 0: legal.append(f"POUR ({rid})")
        legal.append("RESET")
        return legal

    def _simulate_transition(self, state, action):
        if action == "RESET": return [(self.initial_state, 1.0, 0)]
        parts = action.split()
        atype = parts[0]
        rid = int(parts[1].strip("()"))
        prob = self.learner.get_robot_prob(rid)
        
        outcomes = []
        s_succ, r_succ = self._apply_state_change(state, rid, atype, True)
        outcomes.append((s_succ, prob, r_succ))
        
        p_fail = 1.0 - prob
        if p_fail > 0:
            if atype in ["UP", "DOWN", "LEFT", "RIGHT"]:
                valid = self._get_robot_moves_only(state, rid)
                others = [m for m in valid if m != atype] + ["STAY"]
                sub_p = p_fail / len(others)
                for fa in others:
                    if fa == "STAY": outcomes.append((state, sub_p, 0))
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
            t_idx = -1
            for i, t in enumerate(tl):
                if t[0] == (r,c): t_idx = i; break
            if t_idx != -1:
                tp, tamt = tl[t_idx]
                if tamt - 1 <= 0: del tl[t_idx]
                else: tl[t_idx] = (tp, tamt - 1)
                rl[idx] = (rid, (r,c), load + 1)
                
        elif atype == "POUR":
            if success:
                pidx = -1
                for i, p in enumerate(pl):
                    if p[0] == (r, c): pidx = i; break
                if pidx != -1:
                    ppos, pneed = pl[pidx]
                    rew = self.learner.get_expected_reward(ppos)
                    if pneed - 1 <= 0: del pl[pidx]
                    else: pl[pidx] = (ppos, pneed - 1)
                    tot_need -= 1
                    rl[idx] = (rid, (r,c), load - 1)
                else: rl[idx] = (rid, (r,c), load - 1)
            else:
                rl[idx] = (rid, (r,c), load - 1)

        rl.sort(key=lambda x: x[0])
        pl.sort(key=lambda x: x[0])
        tl.sort(key=lambda x: x[0])
        return (tuple(rl), tuple(pl), tuple(tl), tot_need), rew

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

    def _parse_initial_state(self):
        r_list = sorted([(rid, (d[0], d[1]), d[2]) for rid, d in self.problem_config["Robots"].items()])
        p_list = sorted([(pos, n) for pos, n in self.problem_config["Plants"].items()])
        t_list = sorted([(pos, w) for pos, w in self.problem_config["Taps"].items()])
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