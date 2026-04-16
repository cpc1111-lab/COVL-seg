from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    use_hciba: bool = True
    use_spectral_ogp: bool = True
    use_ctr: bool = True
    use_replay: bool = True
    joint_training: bool = False
    use_ewc: bool = False
    use_mib_loss: bool = False
    use_plop_loss: bool = False
    use_zscl_distill: bool = False
