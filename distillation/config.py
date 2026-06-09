"""
Configuration for Knowledge Distillation
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DistillationConfig:
    """
    Configuration for knowledge distillation with logit standardization
    
    Based on: Logit Standardization in Knowledge Distillation (CVPR 2024)
    """
    
    # Temperature parameters
    TEMPERATURE: float = 2.0  # Base temperature τ
    LOGIT_STAND: bool = True  # Enable logit standardization
    
    # Loss weights
    CE_WEIGHT: float = 0.1  # Cross-entropy loss weight
    KD_WEIGHT: float = 9.0  # Knowledge distillation loss weight
    DISTILL_METHOD: str = "kd"  # kd, dkd, mlkd, ctkd
    DKD_ALPHA: float = 1.0  # Target-class KD weight for DKD
    DKD_BETA: float = 8.0  # Non-target-class KD weight for DKD
    MLKD_TEMPERATURES: tuple = (2.0, 3.0, 4.0)  # Extra temperatures used by MLKD
    MLKD_CC_WEIGHT: float = 1.0  # Class-correlation term weight
    MLKD_BC_WEIGHT: float = 1.0  # Batch-correlation term weight
    CTKD_MIN_TEMP: float = 1.0  # Lower bound for learnable CTKD temperature
    CTKD_MAX_TEMP: float = 10.0  # Upper bound for learnable CTKD temperature
    
    # Model paths
    TEACHER_PATH: str = ""  # Path to teacher checkpoint
    STUDENT_PATH: Optional[str] = None  # Path to student checkpoint (for resume)
    
    # Training parameters
    WARMUP_EPOCHS: int = 20  # Warmup epochs for distillation
    
    # Dataset
    DATASET_NAME: str = "Blue"  # Dataset name (Blue, HFUT, PolyU, TJC)
    NUM_CLASSES: int = 250  # Number of classes
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        return cls(
            TEMPERATURE=args.temperature,
            LOGIT_STAND=args.logit_stand,
            CE_WEIGHT=args.ce_weight,
            KD_WEIGHT=args.kd_weight,
            DISTILL_METHOD=getattr(args, "distill_method", "kd"),
            DKD_ALPHA=getattr(args, "dkd_alpha", 1.0),
            DKD_BETA=getattr(args, "dkd_beta", 8.0),
            MLKD_TEMPERATURES=tuple(getattr(args, "mlkd_temperatures", [2.0, 3.0, 4.0])),
            MLKD_CC_WEIGHT=getattr(args, "mlkd_cc_weight", 1.0),
            MLKD_BC_WEIGHT=getattr(args, "mlkd_bc_weight", 1.0),
            CTKD_MIN_TEMP=getattr(args, "ctkd_min_temp", 1.0),
            CTKD_MAX_TEMP=getattr(args, "ctkd_max_temp", 10.0),
            TEACHER_PATH=args.teacher_path,
            STUDENT_PATH=args.student_path,
            WARMUP_EPOCHS=args.warmup_epochs,
            DATASET_NAME=args.dataset,
            NUM_CLASSES=args.num_classes,
        )
