from metrics import HumanEyeColorMetric, SSIMMetric

# 0. Configuration
METRICS_CONFIG = [
    ("Oklab", HumanEyeColorMetric),
    ("SSIM", SSIMMetric),
]
