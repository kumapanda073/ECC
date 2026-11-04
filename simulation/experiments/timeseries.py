"""1試行だけ回して、区間ごとの時系列図を出力する (デモ用)。"""
from config import num_sections, PHY_SENSORS, HUMAN_CONFIGS
from simulation import run_kalman_simulation_affine
from plotting.timeseries import plot_timeseries


def run_timeseries_demo(A_mat, B_mat, Q_mat, U_sim, nlp_bridge):
    print("\n" + "-" * 60)
    print(" Demo Plot (Single Trial)")
    print("-" * 60)

    x_true_demo, _, _, _ = run_kalman_simulation_affine(
        A_mat, B_mat, U_sim, Q_mat,
        use_human_sensor=False, active_phy_sensors=[]
    )
    _, xe_phy_demo, _, _ = run_kalman_simulation_affine(
        A_mat, B_mat, U_sim, Q_mat,
        use_human_sensor=False,
        active_phy_sensors=PHY_SENSORS,
        x_true=x_true_demo
    )
    _, xe_nlp_demo, _, _ = run_kalman_simulation_affine(
        A_mat, B_mat, U_sim, Q_mat,
        use_human_sensor=True,
        human_sensor_configs=HUMAN_CONFIGS,
        active_phy_sensors=PHY_SENSORS,
        x_true=x_true_demo,
        use_nlp=True,
        nlp_bridge=nlp_bridge
    )

    for sec in range(num_sections):
        plot_timeseries(sec, x_true_demo, xe_phy_demo, xe_nlp_demo)
