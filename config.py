from metrics import CIELABMetric, OklabMetric, SSIMMetric

# 0. Configuration
METRICS_CONFIG = [
    ("Oklab", OklabMetric),
    ("SSIM", SSIMMetric),
    ("CIELAB", CIELABMetric),
]
