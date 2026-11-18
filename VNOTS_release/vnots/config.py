"""Experiment defaults and original case/file labels."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    trajectories: int
    updates: int
    dtype: str
    decades: float
    data_file: str
    labels: tuple
    weight_prefix: str
    result_prefix: str
    colors: tuple


EXPERIMENTS = {
    "piston": Experiment(
        5000,
        100002,
        "float32",
        2.0,
        "Double_piston_G_vars.npy",
        (
            "G_unk_F1_kn_F2_kn",
            "G_kn_F1_unk_F2_unk",
            "G_kn_F1_kn_F2_kn",
            "G_unk_F1_unk_F2_unk",
            "G_kn_F1_kn_F2_unk",
            "G_kn_F1_unk_F2_kn",
        ),
        "weights_",
        "Double_piston_",
        ("tab:purple", "tab:green", "tab:blue", "tab:red", "tab:orange", "tab:brown"),
    ),
    "so3": Experiment(
        500,
        50002,
        "float64",
        1.0,
        "Dissipative_Rigid_Body_Data_SO3_generalized.npz",
        ("G_unk_F_kn", "G_kn_F_unk", "G_kn_F_kn", "G_unk_F_unk"),
        "weights_SO3_",
        "Solution_",
        ("red", "green", "black", "orange"),
    ),
}


def case_flags(system, case):
    if case not in range(len(EXPERIMENTS[system].labels)):
        raise ValueError(f"Invalid {system} case: {case}")
    if system == "piston":
        return case in (1, 2, 4, 5), case in (0, 2, 5), case in (0, 2, 4)
    return case in (1, 2), case in (0, 2)
