from metrics import GradientColorMetric, HumanEyeColorMetric

# 0. Configuration
METRICS_CONFIG = [
    ("Oklab", HumanEyeColorMetric),
    ("Gradient & Color (Lines)", GradientColorMetric),
]
