from .common import BaselineConfig


def build() -> BaselineConfig:
    return BaselineConfig(
        name="oracle",
        joint_training=True,
        use_hciba=True,
        use_spectral_ogp=False,
        use_ctr=True,
        use_replay=False,
    )
