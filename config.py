from metrics import CosineMetric, HumanEyeColorMetric, SSIMMetric

# 0. Configuration
METRICS_CONFIG = [
    ("Oklab", HumanEyeColorMetric),
    ("SSIM", SSIMMetric),
    ("Cosine", CosineMetric),
]
