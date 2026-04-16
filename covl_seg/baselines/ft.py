from .common import BaselineConfig


def build() -> BaselineConfig:
    return BaselineConfig(
        name="ft",
        use_hciba=False,
        use_spectral_ogp=False,
        use_ctr=False,
        use_replay=False,
    )
