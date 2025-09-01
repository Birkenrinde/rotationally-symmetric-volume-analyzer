import os
import csv
import pims
import numpy as np
import cv2
import multiprocessing
from tqdm import tqdm
from typing import Tuple, Optional, List, Dict, Any

from config import Config
from models import LegendreVolumeModel
from utils import (
    Image, Coords, Centroid, apply_canny, find_convex_hull, calculate_centroid
)

class FrameAnalyzer:
    def __init__(self, file_path: str, frame_number: int, config: Config):
        self.file_path = file_path
        self.frame_number = frame_number
        self.config = config
        self.model = LegendreVolumeModel(self.config)
        self.original_image: Optional[Image] = None
        self.hull_points: Optional[Coords] = None
        self.initial_centroid: Optional[Centroid] = None

    def run_analysis(self) -> Dict[str, Any]:
        if not self._load_and_preprocess():
            return {'frame': self.frame_number, 'volume': 0.0, 'error': "Preprocessing failed"}
        
        return self.model.run_model_analysis(
            self.hull_points, self.initial_centroid, self.original_image, self.frame_number
        )
    
    def _load_and_preprocess(self) -> bool:
        try:
            with pims.open(self.file_path) as frames:
                self.original_image = frames[self.frame_number]
        except Exception:
            return False

        proc_img = self.original_image.copy()
        if proc_img.dtype != np.uint8:
            max_val = np.max(proc_img)
            proc_img = (proc_img / (max_val if max_val > 0 else 1) * 255).astype(np.uint8)
        
        otsu_threshold, _ = cv2.threshold(proc_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower, upper = int(otsu_threshold * 0.5), int(otsu_threshold)
        
        edge_coords = apply_canny(proc_img, lower, upper)
        if edge_coords.size == 0: return False
            
        h, w = proc_img.shape[:2]; ez = self.config.EXCLUSION_ZONE_PX
        border_violation = np.any(edge_coords[:, 0] < ez) or np.any(edge_coords[:, 0] >= w - ez) or \
                           np.any(edge_coords[:, 1] < ez) or np.any(edge_coords[:, 1] >= h - ez)
        if border_violation:
            return False
            
        self.hull_points = find_convex_hull(edge_coords, self.config.HULL_PROXIMITY_THRESHOLD_PX)
        if self.hull_points is None or self.hull_points.shape[0] < 3: return False
        
        self.initial_centroid = calculate_centroid(self.hull_points)
        return self.initial_centroid is not None

def _analyze_single_frame_task(args: Tuple[str, int, Config]) -> Optional[Dict[str, Any]]:
    file_path, frame_number, config = args
    analyzer = FrameAnalyzer(file_path, frame_number, config)
    return analyzer.run_analysis()

def process_single_video(video_path: str, cfg: Config) -> None:
    print("\n" + "="*80 + f"\nStarting processing for: {os.path.basename(video_path)}\n" + "="*80)

    try:
        with pims.open(video_path) as frames: video_length = len(frames)
    except Exception as e:
        print(f"ERROR: Could not open video file '{video_path}'. Skipping. Reason: {e}")
        return
    
    print(f"-> Model: Legendre ({cfg.NUMBER_OF_LEGENDRE_COEFFS} coefficients, Optimizer: DifferentialEvolution)")
    
    start_frame = cfg.FIRST_FRAME if cfg.FIRST_FRAME is not None else 0
    end_frame = min(cfg.LAST_FRAME, video_length) if cfg.LAST_FRAME is not None else video_length
    frame_indices = range(start_frame, end_frame)
    
    if not frame_indices:
        print("WARNING: No frames found in the specified range. Skipping video.")
        return

    tasks = [(video_path, i, cfg) for i in frame_indices]
    num_cores = cfg.NUM_CORES if cfg.NUM_CORES is not None else multiprocessing.cpu_count()
    print(f"-> Analyzing {len(tasks)} frames (from {start_frame} to {end_frame-1}) on {num_cores} cores...")
    
    results_list = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap_unordered(_analyze_single_frame_task, tasks), total=len(tasks), desc="Processing frames"):
            if result: results_list.append(result)
    
    if not results_list:
        print("-> Analysis complete, but no valid results found to save.")
        return

    results_list.sort(key=lambda r: r['frame'])
    _write_results_to_csv(results_list, video_path, cfg)

def _write_results_to_csv(results_list: List[Dict], video_path: str, cfg: Config) -> None:
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    terms_suffix = f"_c{cfg.NUMBER_OF_LEGENDRE_COEFFS}"
    optimizer_suffix = "_DE"
    filename = f"Legendre_{video_basename}{terms_suffix}{optimizer_suffix}.csv"
    csv_path = os.path.join(cfg.OUTPUT_BASE_DIR, filename)
    
    fieldnames = ['frame', 'volume', 'rmse_final'] + [f'c{i}' for i in range(cfg.NUMBER_OF_LEGENDRE_COEFFS)]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', extrasaction='ignore')
        writer.writeheader()
        for res_item in results_list:
             # Standard formatting with '.' for decimals
            writer.writerow({k: v for k, v in res_item.items() if k in fieldnames})
            
    print(f"-> Analysis complete. Results saved to: {csv_path}")

def main() -> None:
    cfg = Config()
    os.makedirs(cfg.OUTPUT_BASE_DIR, exist_ok=True)
    
    if os.path.isfile(cfg.INPUT_PATH):
        print(f"Single file mode for: {cfg.INPUT_PATH}")
        process_single_video(cfg.INPUT_PATH, cfg)
    else:
        print(f"ERROR: The specified path does not exist: {cfg.INPUT_PATH}")
        return
        
    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
