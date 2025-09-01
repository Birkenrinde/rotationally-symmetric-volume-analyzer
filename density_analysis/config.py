from typing import Optional

class Config:
    """
    Central configuration class to control all aspects of the analysis.
    All adjustable parameters are consolidated here.
    """
    # --- Path and Frame Settings
    INPUT_PATH: str = "path/to/your/file.cine"
    OUTPUT_BASE_DIR: str = "analysis_results"
    FIRST_FRAME: Optional[int] = 0
    LAST_FRAME: Optional[int] = None  # None processes all frames

    # --- Debugging and Output Control ---
    VISUAL_DEBUG_MODE: bool = False  # if True it creates detailed debug images for each frame
    NUM_CORES: Optional[int] = None  # None uses all available CPU cores

    # --- Edge and Contour Detection Parameters ---
    EXCLUSION_ZONE_PX: int = 2 # Border around the image where no edge is allowed 
    HULL_PROXIMITY_THRESHOLD_PX: int = 0 # Includes all edgepoint in a here defined radius around a hullpoint

    # --- Mathematical Model Parameters ---
    NUMBER_OF_LEGENDRE_COEFFS: int = 7  

    # --- Differential Evolution Optimizer Parameters ---
    # Adjusting these affects accuracy and runtime.
    DE_POPSIZE: int = 15
    DE_MAXITER: int = 200
    DE_TOL: float = 0.01

    # --- Bounds for Optimization Parameters ---
    DECLINATION_LIMIT_DEG: float = 45.0

    CENTROID_SHIFT_FACTOR_R: float = 0.5
