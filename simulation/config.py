import numpy as np
from pathlib import Path

# 区間数と状態数
num_sections = 10
n_states = num_sections * 2

# タイムステップ
T = 70

# ノイズ設定
q_disturbance = 0.03
r_phy_density = 0.05            
r_phy_velocity = 0.04           
r_human_den_perception = 0.01
r_human_vel_perception = 0.01

# 状態範囲
x_min_density = 0.0
x_max_density = 2.0
x_min_velocity = 0.0
x_max_velocity = 1.8

# 人センサ設定（5クラス)
num_human_categories = 5
cat_den_max = 2.0
category_thresholds = [i * (cat_den_max / num_human_categories) for i in range(num_human_categories + 1)]
category_means = [(category_thresholds[i] + category_thresholds[i+1]) / 2 for i in range(num_human_categories)]
r_human_quantization = ((category_thresholds[1] - category_thresholds[0])**2) / 12

num_vel_categories = 5
cat_vel_max = 1.8
vel_thresholds = [i * (cat_vel_max / num_vel_categories) for i in range(num_vel_categories + 1)]
vel_means = [(vel_thresholds[i] + vel_thresholds[i+1]) / 2 for i in range(num_vel_categories)]
r_vel_quantization = ((vel_thresholds[1] - vel_thresholds[0])**2) / 12

# シミュレーション実行設定
NUM_TRIALS = 100

# 物理センサ配置場所
PHY_SENSORS = [0, 4] 

# 人センサ配置場所と観測頻度
HUMAN_CONFIGS = [
    {'section': s, 'interval': 1, 'offset': 0}
    for s in range(num_sections)
    for _ in range(5) # 各区間5人
]

# 仮想入力
inflow_main = 0.1   # u1: 区間1への流入
inflow_u2   = 0.3   # u2: 区間4への流入
inflow_u3   = 0.2   # u3: 区間7への流入
inflow_v1   = 0.35   # v1: 区間1の速度入力(クリッピング範囲に入るように調整する)
inflow_v2   = 0.03  # v2: 区間4の速度入力
inflow_v3   = 0.05  # v3: 区間7の速度入力 


# システムモデル設定(model.py)
A_MATRIX_CSV = "A_matrix_synthetic.csv"
dim_u = 6
INPUT_STATE_INDICES = [0, 3, 6, num_sections + 0, num_sections + 3, num_sections + 6]

q_density_base  = 0.002
q_velocity_base = 0.001
DISTURBANCE_STATE_INDICES = [3, 6, num_sections + 3, num_sections + 6]

# 外乱の平均(simulation.py)
W_MEAN_DENSITY  = {3: 0.0, 6: 0.0}
W_MEAN_VELOCITY = {3: 0.0, 6: 0.0}

# 初期値(simulation.py)
x0_density  = 0.8    # 真値の初期密度(全区間)
x0_velocity = 0.9    # 真値の初期速度(全区間)
P0_scale    = 10.0   # 推定誤差共分散の初期値

# 評価実験設定(main.py)
ERROR_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # テキスト選択ミス率
NOISE_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # 無関係テキスト(noisyプール)の割合
CALIB_N_BINS = 10                               # キャリブレーション図のビン数

# 2ヘッドモデルのcheckpoint
ECC_DIR = Path(__file__).resolve().parents[1]
DUAL_HEAD_DIR = ECC_DIR / "training" / "dual_head"
DENSITY_CHECKPOINT_PATH  = DUAL_HEAD_DIR / "checkpoints" / "model_epoch10.pt"
VELOCITY_CHECKPOINT_PATH = DUAL_HEAD_DIR / "checkpoints_velocity" / "model_epoch10.pt"
