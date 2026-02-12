from metrics import CosineMetric, OklabMetric, SSIMMetric

# 0. Configuration
METRICS_CONFIG = [
    ("Oklab", OklabMetric),
    ("SSIM", SSIMMetric),
    ("Cosine", CosineMetric),
]
