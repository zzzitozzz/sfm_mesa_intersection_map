import subprocess
import time
import statistics
import shutil
import os

# ======== 設定 ========
TARGET_SCRIPTS = [
    ("旧版", "../old_server_model/goal_up_forceful_tau/server.py"),
    ("新版", "./server.py"),
]
PRE_RUN = 0  # 事前実行回数（キャッシュ生成用）
COLD_RUNS = 5  # 各モード(cold/warm)での繰り返し回数
WARM_RUNS = 10  # 各モード(cold/warm)での繰り返し回数
SIM_RUNS = 100  # 計算機一台あたりのシミュレーション回数

# Numbaキャッシュディレクトリ（環境に合わせて変更可能）
NUMBA_CACHE_DIR = os.path.expanduser("~/.numba")

# ======== 関数 ========

def clear_numba_cache():
    """Numbaキャッシュを削除"""
    if os.path.exists(NUMBA_CACHE_DIR):
        shutil.rmtree(NUMBA_CACHE_DIR)
        print(f"🧹 Numbaキャッシュを削除しました: {NUMBA_CACHE_DIR}")

def clear_pycache():
    """__pycache__ ディレクトリを再帰的に削除"""
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))

def run_script(script, use_cache=True, num_seed=0):
    """Pythonスクリプトを実行し、実行時間を返す"""
    cmd = ["python3.11"]
    if not use_cache:
        cmd.append("-B")  # キャッシュ無効化
    cmd.extend([script, "200", "0.2", str(num_seed)])

    start = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time()

    return end - start

    # cmd.append(script_path, "200", "0.2", str(num_seed))


def benchmark(script, label):
    """cold run / warm run を自動実行"""
    print(f"\n=== {label} ({script}) のベンチマーク ===")
    # --- Cold run ---
    print("\n--- 事前実行（OSキャッシュ生成） ---")
    t = run_script(script, use_cache=False, num_seed=PRE_RUN)  # 事前実行でキャッシュ生成
    print(f"{label} cold 実行(集計外) seed値:{PRE_RUN}: {t:.3f} 秒")
    print("\n--- Cold run（キャッシュなし） ---")
    cold_times = []
    for c_seed in range(COLD_RUNS):
        clear_pycache()
        clear_numba_cache()
        t = run_script(script, use_cache=False, num_seed=c_seed)
        cold_times.append(t)
        print(f"{label} cold 実行 seed値:{c_seed}: {t:.3f} 秒")

    # --- Warm run ---
    print("\n--- Warm run（キャッシュあり） ---")
    warm_times = []
    for w_seed in range(COLD_RUNS*10, COLD_RUNS*10 + WARM_RUNS):
        t = run_script(script, use_cache=True, num_seed=w_seed)
        warm_times.append(t)
        print(f"{label} warm 実行 seed値:{w_seed}: {t:.3f} 秒")

    # --- 結果 ---
    print("\n=== 結果 ===")
    print(f"cold 平均: {statistics.mean(cold_times):.3f} 秒, 標準偏差: {statistics.pstdev(cold_times):.3f}")
    print(f"warm 平均: {statistics.mean(warm_times):.3f} 秒, 標準偏差: {statistics.pstdev(warm_times):.3f}")

    return {
        "cold_mean": statistics.mean(cold_times),
        "warm_mean": statistics.mean(warm_times)
    }


# ======== 実行部分 ========

if __name__ == "__main__":
    results = {}
    for label, script in TARGET_SCRIPTS:
        results[label] = benchmark(script, label)

    # print("\n=== 改善率（warm run基準） ===")
    # old = results["旧版"]["warm_mean"]
    # new = results["新版"]["warm_mean"]
    # improvement = (old - new) / old * 100
    # print(f"改善率: {improvement:.2f}% （旧版→新版）")
    print(f"\n=== 改善率（cold run1回, warm run{SIM_RUNS}回） ===")
    total_time_old = results["旧版"]["cold_mean"] + results["旧版"]["warm_mean"] * SIM_RUNS
    total_time_new = results["新版"]["cold_mean"] + results["新版"]["warm_mean"] * SIM_RUNS
    improvement = (total_time_old - total_time_new) / total_time_old * 100
    print(f"改善率: {improvement:.2f}% （旧版→新版）")
