import heapq
import copy
import time

# ==========================================
# 1. 配置与数据输入 (Configuration & Data Input)
# ==========================================

class GameConfig:
    def __init__(self):
        print("=== Robocon 2026 决策辅助系统 (垂直九宫格简化版) ===")
        
        # --- 资源库存 (请在此修改) ---
        self.my_r1_kfs = 5       
        self.my_r2_kfs = 5       
        self.my_weapons = 3      
        
        # --- 时间消耗 (秒) ---
        self.t_place_bottom = 5.0
        self.t_place_middle = 5.0
        self.t_place_top = 8.0
        self.t_attack = 6.0       
        
        # --- 失败概率 (0.0 - 1.0) ---
        self.p_fail_place = 0.1  
        self.p_fail_attack = 0.2 
        
        # --- 比赛状态 ---
        self.time_remaining = 180.0 
        
        # --- 初始盘面录入 (3x3 棋盘) ---
        # 简化结构: [归属(-1/0/1), KFS类型(1=R1/2=R2)]
        # 0,0 代表空； 1,2 代表我方 R2KFS； -1,1 代表敌方 R1KFS
        # **行 0 (R0) 对应顶层区域**
        # **行 1 (R1) 对应中层区域**
        # **行 2 (R2) 对应底层区域**
        
        self.initial_board = [
            # 行 0 (顶层区域)
            [{'owner': 0, 'type': 0}, {'owner': 0, 'type': 0}, {'owner': 0, 'type': 0}],
            # 行 1 (中层区域)
            [{'owner': 0, 'type': 0}, {'owner': 0, 'type': 0}, {'owner': 0, 'type': 0}],
            # 行 2 (底层区域)
            [{'owner': 0, 'type': 0}, {'owner': 0, 'type': 0}, {'owner': 0, 'type': 0}],
        ]
        
        print("\n--- 配置载入完毕，无需输入 ---\n")

# ==========================================
# 2. 核心数据结构 (Data Structures)
# ==========================================

class GameState:
    def __init__(self, board, r1_kfs, r2_kfs, weapons, time_elapsed, parent=None, action_desc=""):
        self.board = board
        self.r1_kfs = r1_kfs
        self.r2_kfs = r2_kfs
        self.weapons = weapons
        self.time_elapsed = time_elapsed
        self.parent = parent
        self.action_desc = action_desc
        
        self.score = self.calculate_score()
        self.board_hash = self._get_hash()

    def _get_hash(self):
        s = ""
        for r in self.board:
            for c in r:
                s += f"{c['owner']}{c['type']}|"
        return s

    def calculate_score(self):
        score = 0
        for r in range(3):
            for c in range(3):
                cell = self.board[r][c]
                if cell['owner'] == 1:
                    kfs_type = cell['type']
                    
                    if r == 0: # 顶层区域
                        score += 90 if kfs_type == 1 else 100
                    elif r == 1: # 中层区域
                        score += 60 if kfs_type == 1 else 70
                    elif r == 2: # 底层区域
                        score += 40 if kfs_type == 1 else 50
        return score

    def check_grand_master(self):
        lines = []
        for c in range(3): 
            lines.append([self.board[r][c] for r in range(3)])
        lines.append([self.board[i][i] for i in range(3)])
        lines.append([self.board[i][2-i] for i in range(3)])
        
        for line in lines:
            owners = [cell['owner'] for cell in line]
            types = [cell['type'] for cell in line]
            
            if all(o == 1 for o in owners):
                if 2 in types:
                    return True
        return False

    def __lt__(self, other):
        return self.time_elapsed < other.time_elapsed

# ==========================================
# 3. A* 算法逻辑 (A* Solver)
# ==========================================

class AStarSolver:
    def __init__(self, config):
        self.cfg = config
        
    def heuristic(self, state):
        if state.check_grand_master(): 
            return 0
            
        min_steps_needed = 3 
        
        indices = []
        for c in range(3): 
            indices.append([(r, c) for r in range(3)])
        indices.append([(i, i) for i in range(3)])
        indices.append([(i, 2-i) for i in range(3)])
        
        for line_idx in indices:
            my_count, enemy_count = 0, 0
            for r, c in line_idx:
                cell = state.board[r][c]
                if cell['owner'] == 1: 
                    my_count += 1
                elif cell['owner'] == -1: 
                    enemy_count += 1
            
            steps = 0
            if enemy_count > 0:
                if state.weapons >= enemy_count: 
                    steps += enemy_count
                else: 
                    continue
            
            steps += (3 - my_count - enemy_count)
            if steps < min_steps_needed: 
                min_steps_needed = steps
                
        return min_steps_needed * self.cfg.t_place_bottom

    def solve(self):
        start_time = time.time()
        initial_state = GameState(
            self.cfg.initial_board, 
            self.cfg.my_r1_kfs, 
            self.cfg.my_r2_kfs, 
            self.cfg.my_weapons, 
            0.0
        )
        
        pq = []
        start_h = self.heuristic(initial_state)
        heapq.heappush(pq, (start_h, initial_state))
        visited = set()
        visited.add(initial_state.board_hash)
        best_score_state = initial_state
        
        while pq:
            f, current_state = heapq.heappop(pq)
            if current_state.time_elapsed > self.cfg.time_remaining: 
                continue
            if current_state.score > best_score_state.score: 
                best_score_state = current_state

            if current_state.check_grand_master():
                print(f"\n*** 找到致胜策略! 耗时: {time.time() - start_time:.4f}s ***")
                return self.reconstruct_path(current_state)
            
            next_states = self.get_next_states(current_state)
            
            for next_s in next_states:
                if next_s.board_hash in visited: 
                    continue
                visited.add(next_s.board_hash)
                h = self.heuristic(next_s)
                score_bias = next_s.score * 0.01 
                f_new = next_s.time_elapsed + h - score_bias
                heapq.heappush(pq, (f_new, next_s))
        
        print(f"\n--- 无法在剩余时间内达成绝对胜利，推荐最高分策略 ---")
        return self.reconstruct_path(best_score_state)

    def get_next_states(self, state):
        successors = []
        
        for r in range(3):
            for c in range(3):
                cell = state.board[r][c]
                
                if cell['owner'] == 0:
                    if r == 2: # 底层区域 (R1放)
                        cost_time = self.cfg.t_place_bottom
                        if state.r1_kfs > 0:
                            new_board = copy.deepcopy(state.board)
                            new_board[r][c] = {'owner': 1, 'type': 1}
                            cost = cost_time / (1.0 - self.cfg.p_fail_place)
                            successors.append(GameState(
                                new_board, state.r1_kfs - 1, state.r2_kfs, state.weapons,
                                state.time_elapsed + cost, state,
                                f"R1 放置 R1-KFS 到 ({r},{c}) [底层]"
                            ))
                        if state.r2_kfs > 0:
                            new_board = copy.deepcopy(state.board)
                            new_board[r][c] = {'owner': 1, 'type': 2}
                            cost = cost_time / (1.0 - self.cfg.p_fail_place)
                            successors.append(GameState(
                                new_board, state.r1_kfs, state.r2_kfs - 1, state.weapons,
                                state.time_elapsed + cost, state,
                                f"R1 放置 R2-KFS 到 ({r},{c}) [底层]"
                            ))

                    elif r == 1: # 中层区域 (R2放)
                        cost_time = self.cfg.t_place_middle
                        if state.r1_kfs > 0:
                            new_board = copy.deepcopy(state.board)
                            new_board[r][c] = {'owner': 1, 'type': 1}
                            cost = cost_time / (1.0 - self.cfg.p_fail_place)
                            successors.append(GameState(
                                new_board, state.r1_kfs - 1, state.r2_kfs, state.weapons,
                                state.time_elapsed + cost, state,
                                f"R2 放置 R1-KFS 到 ({r},{c}) [中层]"
                            ))
                        if state.r2_kfs > 0:
                            new_board = copy.deepcopy(state.board)
                            new_board[r][c] = {'owner': 1, 'type': 2}
                            cost = cost_time / (1.0 - self.cfg.p_fail_place)
                            successors.append(GameState(
                                new_board, state.r1_kfs, state.r2_kfs - 1, state.weapons,
                                state.time_elapsed + cost, state,
                                f"R2 放置 R2-KFS 到 ({r},{c}) [中层]"
                            ))

                    elif r == 0: # 顶层区域 (R1+R2协作)
                        cost_time = self.cfg.t_place_top
                        if state.r1_kfs > 0:
                            new_board = copy.deepcopy(state.board)
                            new_board[r][c] = {'owner': 1, 'type': 1}
                            cost = cost_time / (1.0 - self.cfg.p_fail_place)
                            successors.append(GameState(
                                new_board, state.r1_kfs - 1, state.r2_kfs, state.weapons,
                                state.time_elapsed + cost, state,
                                f"R1+R2 放置 R1-KFS 到 ({r},{c}) [顶层]"
                            ))
                        if state.r2_kfs > 0:
                            new_board = copy.deepcopy(state.board)
                            new_board[r][c] = {'owner': 1, 'type': 2}
                            cost = cost_time / (1.0 - self.cfg.p_fail_place)
                            successors.append(GameState(
                                new_board, state.r1_kfs, state.r2_kfs - 1, state.weapons,
                                state.time_elapsed + cost, state,
                                f"R1+R2 放置 R2-KFS 到 ({r},{c}) [顶层]"
                            ))

                elif cell['owner'] == -1:
                    if state.weapons > 0:
                        new_board = copy.deepcopy(state.board)
                        new_board[r][c] = {'owner': 0, 'type': 0}
                        cost = self.cfg.t_attack / (1.0 - self.cfg.p_fail_attack)
                        successors.append(GameState(
                            new_board, state.r1_kfs, state.r2_kfs, state.weapons - 1,
                            state.time_elapsed + cost, state,
                            f"R1 使用兵器移除 ({r},{c}) 的敌方方块"
                        ))
        
        return successors

    def reconstruct_path(self, state):
        path = []
        current = state
        while current.parent:
            path.append(current)
            current = current.parent
        return path[::-1]

# ==========================================
# 4. 主程序入口 (Main Execution)
# ==========================================

if __name__ == "__main__":
    config = GameConfig()
    solver = AStarSolver(config)
    print("正在计算最优决策路径...\n")
    path = solver.solve()
    
    if not path:
        print("未找到有效操作步骤。")
    else:
        print("="*40)
        print(f"推荐策略 (预计总耗时: {path[-1].time_elapsed:.2f}s, 最终得分: {path[-1].score})")
        print("="*40)
        
        for i, step in enumerate(path):
            print(f"步骤 {i+1}: {step.action_desc}")
            print(f"       -> 预计耗时节点: {step.time_elapsed:.2f}s")
            print("       当前盘面示意:")
            for row in step.board:
                line_str = "       |"
                for cell in row:
                    symbol = " "
                    if cell['owner'] == 1: symbol = "O"
                    elif cell['owner'] == -1: symbol = "X"
                    line_str += f" {symbol} |"
                print(line_str)
            print("-" * 20)
        
        print("\n提示: 'O' 代表我方, 'X' 代表敌方")
        print("注意: 耗时包含基于失败概率的风险溢价，实际操作可能更快。")
