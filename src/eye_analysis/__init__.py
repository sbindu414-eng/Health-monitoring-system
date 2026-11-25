"""
AI-Based Eye Image Analysis & Diabetes Risk Screening Package.

Separates distinct medical tasks:
1. Retinal fundus diabetic retinopathy severity grading (deep learning CNN).
2. Anterior eye / iris segmentation & CV quality/feature analysis (experimental CV).
3. Clinical / tabular diabetes risk screening on Pima Indians Diabetes dataset.
"""

__all__ = [
    "dataset_prep",
    "preprocessing",
    "iris_pipeline",
    "train_retinal",
    "train_tabular",
    "evaluate",
    "predict",
]
