from .common import BaselineConfig


def build() -> BaselineConfig:
    return BaselineConfig(
        name="mib",
        use_hciba=False,
        use_spectral_ogp=False,
        use_ctr=False,
        use_replay=True,
        use_mib_loss=True,
    )
