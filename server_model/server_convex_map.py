import sys
from model import MoveAgent

import numpy as np
from dataclasses import dataclass
from agent import SharedParams, Human, HumanSpecs, ForcefulHuman, ForcefulHumanSpecs, Wall

@dataclass
class InitPosFuncs:
    f_r_use: bool = False # True: 常に強引な避難者の半径を使用 False: 普段は強引な避難者も通常の避難者と同じ半径を使用

    def decide_position(self, r, f_r, human_array):
        while 1:
            x = np.random.randint(4, 34) + np.random.rand()
            y = np.random.randint(26, 40) + np.random.rand() #初期配置(未確定)
            if 4. + r * 2 <= x <= 34. - r * 2 and 26. + r * 2 <= y <= 40. - r * 2:
                tmp_pos = np.array((x, y))
                if self.human_pos_check(r, f_r, tmp_pos, human_array): #ボジションチェック(既存のエージェントの位置と被っていないか)
                    pos = tmp_pos
                    break
        return pos

    def decide_forceful_position(self, r, f_r, human_array):
        len_sq = 3 # 初期エリア：ただし長方形の一辺の長さはlen_sq*2
        while 1:
            x = np.random.randint(19. - len_sq, 19. +
                                    len_sq) + np.random.rand()
            y = np.random.randint(32.5 - len_sq, 32.5 +
                                    len_sq) + np.random.rand()
            if 19.- len_sq + r <= x <= 19.+ len_sq - r and 32.5- len_sq + r <= y <= 32.5+ len_sq - r:
                tmp_pos = np.array((x, y))
                if self.forceful_human_pos_check(r, f_r, tmp_pos, human_array):
                    pos = tmp_pos
                    break
        return pos

    def human_pos_check(self, r, f_r, tmp_pos, human_array):
        for hu in human_array:
            dis = self.get_distance(tmp_pos, hu.pos)
            if type(hu) == Human:
                if dis < r + r:
                    return False
            elif type(hu) == ForcefulHuman:
                if self.f_r_use:
                    if dis < r + f_r:
                        return False
                else:
                    if dis < r + r:
                        return False
        return True

    def forceful_human_pos_check(self, r, f_r, tmp_pos, human_array):
        for hu in human_array:
            dis = self.get_distance(tmp_pos, hu.pos)
            if type(hu) == Human:
                if self.f_r_use:
                    if dis < f_r + r:
                        return False
                else:
                    if dis < r + r:
                        return False
            elif type(hu) == ForcefulHuman:
                if self.f_r_use:
                    if dis < f_r + f_r:
                        return False
                else:
                    if dis < r + r:
                        return False
                    
        return True
    
    def get_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)


def make_new_model_instance(human_var, forceful_human_var, wall_arr, pop_num, for_pop, dests, edges,
                            goal_arr, tmp_seed,len_sq, f_r, f_tau, pos_func, csv_plot):
    ex_num = 1 # force_tau
    # if csv_plot:
    #     file_name_array = [
    #         f"/local_home/keito/simple_convex_map/agst_dir/goal_up_forceful_tau/ex{ex_num}_for_{for_pop}_len_{int(len_sq)}_csv/tau_{int(f_tau*100)}/"]
    # else:
    #     file_name_array = [
    #         f"/local_home/keito/simple_convex_map/agst_dir/goal_up_forceful_tau/ex{ex_num}_for_{for_pop}_len_{int(len_sq)}/tau_{int(f_tau*100)}/"]
    if csv_plot:
        file_name_array = [f"./tmp_data/tau_{int(f_tau*100)}/"]
    else:
        file_name_array = [f"./tmp_data/tau_{int(f_tau*100)}/"]

    m = MoveAgent(
        population=pop_num,
        for_population=for_pop,
        dests=dests,
        edges=edges,
        goal_arr=goal_arr,
        v_arg=[1., 1.],
        wall_arr=wall_arr,
        seed=tmp_seed,  # 乱数生成用
        r=0.5,  # 避難者の大きさ
        wall_r=1.0,  # うそ壁の大きさß
        human_var=human_var,
        forceful_human_var=forceful_human_var,
        width=60,  # 見かけの大きさ(マップ)
        height=60,  # 見かけの大きさ(マップ)
        dt=0.3,
        in_dest_d=3,
        vision=1.5,  # 10
        time_step=0,
        add_file_name="",
        add_file_name_arr=file_name_array,
        len_sq=len_sq,
        f_r=f_r,
        pos_func=pos_func,
        csv_plot=csv_plot)
    return m


if __name__ == '__main__':
    pop_num = int(sys.argv[1])  # 通常の人数
    f_tau = float(sys.argv[2])  # 変更する変数の値
    tmp_seed = int(sys.argv[3])  # seed値
    f_r = 0.5 # 強引な避難者の大きさ
    for_pop = 0  # 強引な避難者の人数 #tmp
    csv_plot = True  # csvファイル(各エージェントの動きの軌跡)を出力するかどうか
    len_sq = 3  # 長方形の一辺の長さはlen_sq*2
    # max_f_r = 1.01
    human_var = {"m": 80., "tau": 0.5, "k": 120000., "kappa": 240000.,
                 "repul_h": [2000., 0.08], "repul_m": [2000., 0.08]}
    forceful_human_var = {"f_m": 80., "f_tau": f_tau, "f_k": 120000.,
                          "f_kappa": 240000., "f_repul_h": [2000., 0.08], "f_repul_m": [2000., 0.08]}
    pos_func = InitPosFuncs()
    wall_arr = np.array([[[2., 40.], [54., 40.]],
                [[2., 26.], [16., 26.]],
                [[22., 26.], [54., 26.]],
                [[16., 4.], [16., 26.]],
                [[22., 4.], [22., 26.]]])

    dests = [[9, 33], [19, 33], [19, 4], [54, 33]]
    edges = {0: [1], 1: [0, 2, 3], 2: [1], 3: [1]} # ノードの接続情報
    goal_arr = [3, 2] # ゴールのインデックス(通常，強引)
    while 1:
        m = make_new_model_instance(
            human_var, forceful_human_var, wall_arr, pop_num, for_pop, dests, edges, goal_arr, tmp_seed, len_sq, f_r, f_tau, pos_func, csv_plot)
        m.running = True
        while m.running:
            m.step()
        sys.exit()

