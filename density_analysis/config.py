from typing import Optional

class Config:
    """
    Central configuration class to control all aspects of the analysis.
    All adjustable parameters are consolidated here.
    """
    # --- Path and Frame Settings ---
    INPUT_PATH: str = "Messungen/Autoexposure/BMA15_#13.cine"
    OUTPUT_BASE_DIR: str = "Output"
    FIRST_FRAME: Optional[int] = 3388
    LAST_FRAME: Optional[int] = 3500  # None processes all frames

    # --- Debugging and Output Control ---
    VISUAL_DEBUG_MODE: bool = True  # Creates detailed debug images for each frame
    NUM_CORES: Optional[int] = None  # None uses all available CPU cores

    # --- Edge and Contour Detection Parameters ---
    EXCLUSION_ZONE_PX: int = 2
    HULL_PROXIMITY_THRESHOLD_PX: int = 0

    # --- Mathematical Model Parameters ---
    NUMBER_OF_LEGENDRE_COEFFS: int = 7  # Must be >= 7 for c0-c6 output

    # --- Differential Evolution Optimizer Parameters ---
    # Adjusting these affects accuracy and runtime.
    DE_POPSIZE: int = 15
    DE_MAXITER: int = 200
    DE_TOL: float = 0.01

    # --- Bounds for Optimization Parameters ---
    DECLINATION_LIMIT_DEG: float = 45.0
    CENTROID_SHIFT_FACTOR_R: float = 0.5