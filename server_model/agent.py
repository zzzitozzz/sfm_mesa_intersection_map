import os
import copy

import mesa
import numpy as np
import pandas as pd
import math

class SharedParams:
    "_shared: Common human-related parameters shared between Human and ForcefulHuman instances."
    def __init__(self, in_dest_d, vision, dt):
        self.in_dest_d = in_dest_d
        self.vision = vision
        self.dt = dt

class HumanSpecs:
    "_hspecs: Human-related common specs set used in Human (and ForcefulHuman)"
    def __init__(self, r, m, tau, k, kappa, repul_h, repul_m): 
        self.r = r
        self.m = m
        self.tau = tau
        self.k = k
        self.kappa = kappa
        self.repul_h = repul_h
        self.repul_m = repul_m


class Human(mesa.Agent):
    STUCK_WINDOW = 10
    STUCK_DIST = 0.2
    REROUTE_COOLDOWN = 30  # 再探索後、30ステップは再探索しない

    def __init__(self, unique_id, model,
                 pos, velocity,
                 dest, route,
                 tmp_div, shared,
                 human_var_inst,
                 space, add_file_name,
                 route_idx=0,
                 tmp_pos=(0., 0.), pos_array=[],
                 in_goal=False,elapsed_time=0.,  # 経過時間
                 ):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.velocity = velocity
        self._shared = shared
        self._hspecs = human_var_inst
        self.tmp_div = tmp_div #特定の人同士の反発力の大きさを除算もしくは乗算する値
        self.dest = dest #避難所の座標
        self.route = route
        self.space = space #エージェントが動き回る空間を管理するモジュール
        self.add_file_name = add_file_name #保存するファイル名(の基礎.最終的には絶対パスまたは相対パスができる)
        self.route_idx = route_idx #経路のインデックス
        self.tmp_pos = np.array((0., 0.)) #一時的に計算した結果の位置を保存する値(将来的には壁を乗り越えるなどのありえない挙動をした時に元の位置に戻すために一旦計算した位置を保存している)
        self.in_goal = in_goal #目的地に到着したか判定するboolean型の変数
        self.pos_array = [] #自分の位置をステップごとに記録する配列
        self.pos_array.append(self.pos)
        self.elapsed_time = elapsed_time #経過時間
        self.last_reroute_tick = -10**9  # 最後に再探索した時間を保存する変数
        self.rng = model.make_agent_rng(unique_id) #(将来的に)ランダムな要素を入れるためかもしれないため設定

    @property
    def hspecs(self):
        return self._hspecs
    
    @property
    def cur_dest(self):
        return self.model.dests[self.route[self.route_idx]]
    
    def step(self):  # 次の位置を特定するための計算式を書く
        self._calculate()
        dest_dis = self.space.get_distance(self.pos, self.cur_dest)
        self.goal_check(dest_dis)
        self.tmp_pos[0] = self.pos[0] + \
            self.velocity[0] * self._shared.dt  # 仮の位置を計算
        self.tmp_pos[1] = self.pos[1] + self.velocity[1] * self._shared.dt
        return None

    def advance(self):
        self.pos = copy.deepcopy(self.tmp_pos)
        self.pos_array.append(self.pos)
        self.elapsed_time += self._shared.dt
        self.re_route()
        if (self.in_goal):  # goalした場合
            path = self.add_file_name
            self.make_dir(path)
            self.write_record(path)
            self.model.space.remove_agent(self)
            self.model.schedule.remove(self)
            return None
        else:
            self.model.space.move_agent(self, self.pos)  # goalしていない場合
        return None

    def goal_check(self, dest_dis):
        if dest_dis < 1.5:
            if len(self.route) == self.route_idx + 1:
                self.in_goal = True
                self.velocity = [0.0, 0.0]
        # if dest_dis < 0.5:
        #     if len(self.route) == self.route_idx + 1:
        #         self.in_goal = True
        #         self.velocity = [0.0, 0.0]
            else:
                self.route_idx += 1
                return None
            return None
        
    def re_route(self):
        WINDOW = self.STUCK_WINDOW
        D_MIN = self.STUCK_DIST
        COOLDOWN = self.REROUTE_COOLDOWN

        # クールダウン中なら何もしない
        if self.elapsed_time - self.last_reroute_tick < COOLDOWN * self._shared.dt:
            return None
        
        pos_num = len(self.pos_array)
        if pos_num < WINDOW + 1:
            return None
        
        cur_pos = self.pos_array[-1]
        past_pos = self.pos_array[-(WINDOW + 1)]
        moved = math.dist(cur_pos, past_pos)
        if moved < D_MIN:
            self.route, self.dest = self.model.select_first_subgoal(self.pos)
            self.route_idx = 0
            self.last_re_route_tick = self.elapsed_time
        return None

    def make_dir(self, path):
        os.makedirs(f"{path}/Data", exist_ok=True)
        if self.model.csv_plot:
            os.makedirs(f"{path}/csv", exist_ok=True)
        return None

    def write_record(self, path):
        if self.model.csv_plot:
            np.savetxt(f"{path}/csv/id{self.unique_id}_normal"
                       f".csv", self.pos_array, delimiter=",")
        if self.in_goal:
            with open(f"{self.add_file_name}/Data/"
                      f"normal.dat", "a") as f:
                f.write(f"{self.elapsed_time} \n")

    def _sincos(self, x2):
        r_0 = np.sqrt((x2[0] - self.pos[0]) ** 2 + (x2[1] - self.pos[1]) ** 2)
        sin = (x2[1] - self.pos[1]) / r_0
        cos = (x2[0] - self.pos[0]) / r_0
        return cos, sin

    def _force(self, dest):
        fx, fy = 0., 0.
        theta = self._sincos(dest)
        neighbors = self.model.space.get_neighbors(
            self.pos, self._shared.vision, False)
        fx, fy = self.force_from_goal(theta)
        for neighbor in neighbors:
            if self.unique_id == neighbor.unique_id:
                continue
            if type(neighbor) is Human or type(neighbor) is ForcefulHuman:
                tmp_fx, tmp_fy = self.force_from_human(neighbor)
                fx += tmp_fx
                fy += tmp_fy
            elif type(neighbor) is Wall:
                None
        tmp_fx, tmp_fy = 0., 0.
        tmp_fx, tmp_fy = self.force_from_wall()
        fx += tmp_fx
        fy += tmp_fy
        fx /= self.hspecs.m
        fy /= self.hspecs.m
        return fx, fy

    def force_from_goal(self, theta):
        fx = self.hspecs.m * (0.8 * theta[0] - self.velocity[0]) / self.hspecs.tau
        fy = self.hspecs.m * (0.8 * theta[1] - self.velocity[1]) / self.hspecs.tau
        return fx, fy

    def force_from_human(self, neighbor):
        fx, fy = 0., 0.
        n_ij = (self.pos - neighbor.pos) / \
            self.space.get_distance(self.pos, neighbor.pos)
        t_ij = [-n_ij[1], n_ij[0]]
        dis = (self.hspecs.r + neighbor.hspecs.r) - \
            self.space.get_distance(self.pos, neighbor.pos)
        if dis >= 0:
            fx += (self.hspecs.repul_h[0] * (math.e ** (dis / self.hspecs.repul_h[1])) + self.hspecs.k * dis) * \
                n_ij[0] + self.hspecs.kappa * dis * \
                np.dot(
                (neighbor.velocity - self.velocity), t_ij)*t_ij[0]
            fy += (self.hspecs.repul_h[0] * (math.e ** (dis / self.hspecs.repul_h[1])) + self.hspecs.k * dis) * \
                n_ij[1] + self.hspecs.kappa * dis * \
                np.dot(
                    (neighbor.velocity - self.velocity), t_ij)*t_ij[1]
        else:
            fx += self.hspecs.repul_h[0] * (math.e **
                                     (dis / self.hspecs.repul_h[1])) * n_ij[0]
            fy += self.hspecs.repul_h[0] * (math.e **
                                     (dis / self.hspecs.repul_h[1])) * n_ij[1]
        return fx, fy

    def force_from_wall(self):
        fx, fy = 0., 0.
        for i in range(len(self.model.wall_ab)):
            dis, n_iw = self.distance_point_to_segment(i)
            if dis < self._shared.vision:
                t_iw = np.array([-n_iw[1], n_iw[0]])
                tmp_fx, tmp_fy = self.wall_force_core(dis, n_iw, t_iw)
                fx += tmp_fx
                fy += tmp_fy
        return fx, fy

    def distance_point_to_segment(self, i):
        a = self.model.wall_a[i][:2]
        ab = self.model.wall_ab[i][:2]
        ap = self.pos - a
        ab_len2 = self.model.wall_ab_len2[i]

        if ab_len2 == 0: #壁の両端の座標が同じ場合
            vec = self.pos - a
            dis = np.linalg.norm(vec)
            n_iw = vec / dis if dis > 1e-8 else np.array([0., 0.])
            return dis, n_iw
        
        t = np.dot(ap, ab) / ab_len2 
        if t < 0.0:
            closest = self.model.wall_a[i][:2]
        elif t > 1.0:
            closest = self.model.wall_b[i][:2]
        else:
            closest = self.model.wall_a[i][:2] + t * ab

        vec = self.pos - closest
        dis = np.linalg.norm(vec)
        if dis > 1e-8:
            n_iw = vec / dis
        else:
            n_iw = np.array([0., 0.])
        return dis, n_iw
    
    def wall_force_core(self, dis, n_iw, t_iw):
        fx, fy = 0., 0.
        if dis >= 0:
            fx += (self.hspecs.repul_m[0] * (math.e ** (dis / self.hspecs.repul_m[1])) + self.hspecs.k *
                    dis) * n_iw[0] - self.hspecs.kappa * dis * np.dot(self.velocity, t_iw) * t_iw[0]
            fy += (self.hspecs.repul_m[0] * (math.e ** (dis / self.hspecs.repul_m[1])) + self.hspecs.k *
                    dis) * n_iw[1] - self.hspecs.kappa * dis * np.dot(self.velocity, t_iw) * t_iw[1]
        else:
            fx += (self.hspecs.repul_m[0] * (math.e **
                    (dis / self.hspecs.repul_m[1]))) * n_iw[0]
            fy += (self.hspecs.repul_m[0] * (math.e **
                    (dis / self.hspecs.repul_m[1]))) * n_iw[1]
        return fx, fy
    
    def _calculate(self):
        fx, fy = self._force(self.cur_dest)
        self.velocity[0] += fx * self._shared.dt
        self.velocity[1] += fy * self._shared.dt
        if (np.linalg.norm(self.velocity, 2) > 1.):  # review
            v = copy.deepcopy(self.velocity)
            vn = np.linalg.norm(v)
            self.velocity = v / vn
        # if self.unique_id == 2:
        #     print(f"{self.pos=},{self.velocity=}\n{fx=},{fy=}")
        return None
    
    def pos_check(self):
        # area = [[4., 26 + self.hspecs.r], [54., 40. - self.hspecs.r],
        #         [16. + self.hspecs.r, 4.], [22. - self.hspecs.r, 40. - self.hspecs.r]]       
        area = [[2., 26 + self.hspecs.r], [54., 40. - self.hspecs.r],
                [16. + self.hspecs.r, 4.], [22. - self.hspecs.r, 40. - self.hspecs.r]]     
        area_check = False
        i = 0
        while 1:
            if i >= len(area):
                break
            if area[i][0] <= self.tmp_pos[0] <= area[i + 1][0] and area[i][1] <= self.tmp_pos[1] <= area[i + 1][1]:
                area_check = True
                break
            else:
                i += 2
        if area_check:
            return True
        else:
            if self.tmp_pos[0] < 4. and 26. < self.tmp_pos[1] < 40.:
                print(f"!!!!!")
            else:
                print(f"{self.pos=},{self.tmp_pos=}")
            self.tmp_pos = copy.deepcopy(self.pos)
            return False


class ForcefulHumanSpecs:
    "_fhspecs:ForcefulHuman-related specs set used in ForcefulHuman"
    def __init__(self, f_r, f_m, f_tau, f_k, f_kappa, f_repul_h, f_repul_m):
        self.r = f_r
        self.m = f_m
        self.tau = f_tau
        self.k = f_k
        self.kappa = f_kappa
        self.repul_h = f_repul_h
        self.repul_m = f_repul_m


class ForcefulHuman(Human):
    def __init__(self, unique_id, model,
                 pos, velocity,
                 dest, route,
                 tmp_div, shared,
                 human_var_inst,
                 space, add_file_name,
                 forceful_human_var_inst,
                 route_idx=0, tmp_pos=(0., 0.),
                 in_goal=False, pos_array=[],
                 elapsed_time=0.,  # 経過時間
                 _force_mode = False,
                 ):
        super().__init__(unique_id, model, pos,
                         velocity, dest,
                         route, tmp_div, shared,
                         human_var_inst,
                         space, add_file_name,
                         route_idx,
                         tmp_pos, in_goal,  pos_array,
                         elapsed_time,
                         )
        self._fhspecs = forceful_human_var_inst
        self._force_mode = True

    @property
    def hspecs(self):
        return self._fhspecs if self._force_mode else self._hspecs
    
    @property
    def fmode(self):
        return self._force_mode
    
    @fmode.setter
    def fmode(self, val: bool):
        self._force_mode = val

    def step(self):  # 次の位置を特定するための計算式を書く
        self._calculate()
        dest_dis = self.space.get_distance(self.pos, self.cur_dest)
        self.goal_check(dest_dis)
        self.tmp_pos[0] = self.pos[0] + \
            self.velocity[0] * self._shared.dt  # 仮の位置を計算
        self.tmp_pos[1] = self.pos[1] + self.velocity[1] * self._shared.dt
        return None

    def write_record(self, path):
        if self.model.csv_plot:
            np.savetxt(f"{path}/csv/id{self.unique_id}_"
                       f"forceful.csv", self.pos_array, delimiter=",")
        if self.in_goal:
            with open(f"{self.add_file_name}/Data/"
                      f"forceful.dat", "a") as f:
                f.write(f"{self.elapsed_time} \n")
            path = path.replace(f"/seed_{self.model.seed}", "")
            df = pd.DataFrame({"m": [self.hspecs.m], "nol_pop": [self.model.population], "seed": [
                self.model.seed], "id": [self.unique_id], "elapsed_time": [self.elapsed_time]})
            df.to_csv(f"{path}/forceful_time.csv",
                      mode="a", header=False, index = False)
        return None
    

class Obstacle(mesa.Agent):
    def __init__(self, unique_id, model, pos, dir):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.dir = dir

    def step(self):
        return None


class Wall(mesa.Agent):
    def __init__(self, unique_id, model, pos, wall_r, dir):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.wall_r = wall_r
        self.dir = dir

    def step(self):
        return None


class Goal(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)

    def step(self):
        return None

