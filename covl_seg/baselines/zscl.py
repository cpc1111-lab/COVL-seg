from .common import BaselineConfig


def build() -> BaselineConfig:
    return BaselineConfig(
        name="zscl",
        use_hciba=False,
        use_spectral_ogp=False,
        use_ctr=False,
        use_replay=False,
        use_zscl_distill=True,
    )
