from metrics import CIELABMetric, CosineMetric, OklabMetric, SSIMMetric

# 0. Configuration
METRICS_CONFIG = [
    ("Oklab", OklabMetric),
    ("SSIM", SSIMMetric),
    ("CIELAB", CIELABMetric),
    ("Cosine", CosineMetric),
]
