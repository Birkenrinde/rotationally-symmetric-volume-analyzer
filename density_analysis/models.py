import os
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import quad
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from typing import Tuple, Optional, List, Dict, Any

from config import Config
from utils import (
    Coords, Centroid, FitCoeffs, OptParams, Image,
    polar_transform, get_shifted_centroid, legendre_polynomial_series,
    format_radians_as_pi, core_compute_radius_over_theta
)

class LegendreVolumeModel:
    def __init__(self, config: Config):
        self.config = config

    def run_model_analysis(self, hull_points: Coords, initial_centroid: Centroid,
                           original_image: Image, frame_number: int) -> Dict[str, Any]:
        _, initial_radii = polar_transform(hull_points, initial_centroid)
        if initial_radii.size == 0:
            return self._build_error_result(frame_number, "No initial radii found.")

        opt_params, final_cost = self._find_optimal_parameters(hull_points, initial_centroid, initial_radii, frame_number)
        if opt_params is None:
            return self._build_error_result(frame_number, "Optimization (Differential Evolution) failed.")

        final_thetas, final_radii = self._transform_points_to_fit_coords(opt_params, hull_points, initial_centroid)
        if final_thetas is None or final_radii is None:
            return self._build_error_result(frame_number, "Coordinate transformation failed.")

        final_coeffs, success = self._fit_legendre_polynomials(final_thetas, final_radii)
        if not success:
            return self._build_error_result(frame_number, "Final Legendre fit failed.")

        volume = self._compute_volume(final_coeffs)
        model_results = self._build_success_result(frame_number, volume, opt_params, final_cost, final_coeffs)

        if self.config.VISUAL_DEBUG_MODE:
            self._create_all_debug_plots(hull_points, initial_centroid,
                                        final_thetas, final_radii, final_coeffs,
                                        opt_params, model_results, original_image, frame_number)
        return model_results

    def _find_optimal_parameters(self, hull_points: Coords, centroid: Centroid, radii: np.ndarray, seed: int) -> Tuple[Optional[OptParams], float]:
        bounds = self._get_bounds(radii)
        result = differential_evolution(self._global_cost_function, bounds, args=(hull_points, centroid),
                                        popsize=self.config.DE_POPSIZE, maxiter=self.config.DE_MAXITER,
                                        tol=self.config.DE_TOL, seed=seed)
        return (list(result.x), result.fun) if result.success else (None, np.inf)

    def _global_cost_function(self, params: OptParams, hull_points: Coords, initial_centroid: Centroid) -> float:
        thetas, radii = self._transform_points_to_fit_coords(params, hull_points, initial_centroid)
        if thetas is None or len(radii) == 0: return np.inf

        coeffs, success = self._fit_legendre_polynomials(thetas, radii)
        if not success: return np.inf

        fitted_radii = legendre_polynomial_series(thetas, *coeffs)
        rmse = np.sqrt(np.mean((radii - fitted_radii)**2))
        return rmse if not np.isnan(rmse) else np.inf

    def _fit_legendre_polynomials(self, angles: np.ndarray, radii: np.ndarray) -> Tuple[Optional[FitCoeffs], bool]:
        """Performs a fast Legendre fit using linear algebra (np.linalg.lstsq)."""
        num_coeffs = self.config.NUMBER_OF_LEGENDRE_COEFFS
        if radii.size < num_coeffs: return None, False
        
        X = legvander(np.cos(angles), num_coeffs - 1)
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, radii, rcond=None)
            return (coeffs, True) if not np.any(np.isnan(coeffs)) else (None, False)
        except (np.linalg.LinAlgError, ValueError):
            return None, False

    def _transform_points_to_fit_coords(self, params: OptParams, hull_points: Coords, 
                                        initial_centroid: Centroid) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        angle_rad, perp_shift, para_shift = params
        final_centroid = get_shifted_centroid(initial_centroid, angle_rad, perp_shift, para_shift)
        thetas_orig_frame, radii = polar_transform(hull_points, final_centroid)
        if thetas_orig_frame.size == 0: return None, None
        thetas_fit_frame = np.arctan2(np.sin(thetas_orig_frame - angle_rad), np.cos(thetas_orig_frame - angle_rad))
        return thetas_fit_frame, radii
    
    def _compute_volume(self, coeffs: FitCoeffs) -> float:
        def integrand(theta_: float) -> float:
            r = legendre_polynomial_series(np.array(theta_), *coeffs)
            return (r**3) * np.sin(theta_)
        try:
            integral_val, _ = quad(integrand, 0, np.pi)
            volume = (2 * np.pi / 3) * integral_val
            return 0.0 if np.isnan(volume) or np.isinf(volume) else volume
        except Exception:
            return 0.0

    def _get_bounds(self, initial_radii: np.ndarray) -> List[Tuple[float, float]]:
        avg_r = np.mean(initial_radii) if initial_radii.size > 0 else 100.0
        shift_limit = self.config.CENTROID_SHIFT_FACTOR_R * avg_r
        angle_limit = np.deg2rad(self.config.DECLINATION_LIMIT_DEG)
        return [(-angle_limit, angle_limit), (-shift_limit, shift_limit), (-shift_limit, shift_limit)]

    def _build_error_result(self, fn: int, msg: str) -> Dict[str, Any]: 
        return {'frame': fn, 'volume': 0.0, 'error': msg}

    def _build_success_result(self, fn: int, vol: float, p: OptParams, cost: float, c: FitCoeffs) -> Dict[str, Any]:
        results = {
            'frame': fn, 
            'volume': vol, 
            'error': '',
            'rmse_final': cost,
            'optimizer_used': 'DifferentialEvolution',
            'optimal_angle_deg': np.rad2deg(p[0]),
            'optimal_perp_shift_px': p[1],
            'optimal_para_shift_px': p[2],
            **{f'c{i}': coeff for i, coeff in enumerate(c)}
        }
        return results
    
    # --- PLOTTING FUNCTIONS ---

    def _create_all_debug_plots(self, hull_points: Coords, initial_centroid: Centroid,
                                final_thetas: np.ndarray, final_radii: np.ndarray, final_coeffs: FitCoeffs,
                                opt_params: OptParams, model_results: Dict[str, Any],
                                original_image: Image, frame_number: int):
        initial_thetas, initial_radii_plot = core_compute_radius_over_theta(hull_points, initial_centroid)
        self._plot_radius_fit(initial_thetas, initial_radii_plot, "initial", frame_number)
        
        self._plot_radius_fit(final_thetas, final_radii, "final", frame_number, plot_coeffs=final_coeffs)
        
        self._create_summary_plot(hull_points, initial_centroid, final_coeffs,
                                  opt_params, model_results, original_image, frame_number)

    def _create_summary_plot(self, hull_points: Coords, initial_centroid: Centroid, 
                             final_coeffs: Optional[FitCoeffs], opt_params: OptParams, model_results: Dict[str, Any], 
                             original_image: Image, frame_number: int) -> None:
        COLOR_HULL, COLOR_INITIAL, COLOR_FINAL_FIT, COLOR_FINAL_GEOM = 'gray', 'cyan', 'lime', 'red'
        opt_angle, opt_perp, opt_para = opt_params
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(original_image, cmap='gray')

        ax.plot(hull_points[:, 0], hull_points[:, 1], 'o', color=COLOR_HULL, ms=2, alpha=0.6, label='Hüllenpunkte')

        init_thetas, init_radii_plot = core_compute_radius_over_theta(hull_points, initial_centroid)
        init_coeffs, _ = self._fit_legendre_polynomials(init_thetas, init_radii_plot)
        angles_plot = np.linspace(-np.pi, np.pi, 360)
        if init_coeffs is not None:
            radii_init_vals = legendre_polynomial_series(angles_plot, *init_coeffs)
            x_init = initial_centroid[0] + radii_init_vals * np.sin(angles_plot)
            y_init = initial_centroid[1] - radii_init_vals * np.cos(angles_plot)
            ax.plot(x_init, y_init, '--', color=COLOR_INITIAL, lw=1.5, label='Initialer Fit')

        final_centroid = get_shifted_centroid(initial_centroid, opt_angle, opt_perp, opt_para)
        if final_coeffs is not None:
            radii_final_vals = legendre_polynomial_series(angles_plot, *final_coeffs)
            x_final = final_centroid[0] + radii_final_vals * np.sin(angles_plot + opt_angle)
            y_final = final_centroid[1] - radii_final_vals * np.cos(angles_plot + opt_angle)
            ax.plot(x_final, y_final, '-', color=COLOR_FINAL_FIT, lw=2.5, label='Finaler Fit (optimiert)')

        ax.plot(initial_centroid[0], initial_centroid[1], 'x', color=COLOR_INITIAL, ms=10, mew=2)
        ax.plot(final_centroid[0], final_centroid[1], '+', color=COLOR_FINAL_GEOM, ms=12, mew=2)
        
        axis_len = np.max(init_radii_plot) * 1.2 if init_radii_plot.size > 0 else 200
        dx_axis, dy_axis = axis_len * np.sin(opt_angle), axis_len * np.cos(opt_angle)
        ax.plot([final_centroid[0] - dx_axis, final_centroid[0] + dx_axis],
                [final_centroid[1] + dy_axis, final_centroid[1] - dy_axis], '-.', color=COLOR_FINAL_GEOM, lw=1)

        res_text = (f"Frame: {frame_number}\nVolumen: {model_results.get('volume', 0.0):.2f} px³\n\n"
                    f"Optimizer: {model_results.get('optimizer_used', 'N/A')}\n"
                    f"Winkel (φ): {model_results.get('optimal_angle_deg', 0.0):.2f}°\n"
                    f"ds_perp: {model_results.get('optimal_perp_shift_px', 0.0):.2f} px\n"
                    f"ds_para: {model_results.get('optimal_para_shift_px', 0.0):.2f} px\n"
                    f"Finaler RMSE: {model_results.get('rmse_final', 0.0):.4f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax.text(0.05, 0.95, res_text, transform=ax.transAxes, fontsize=10, va='top', bbox=props)
        
        handles, _ = ax.get_legend_handles_labels()
        custom_handles = [
            handles[0], handles[1], handles[2],
            Line2D([0],[0], marker='x', color=COLOR_INITIAL, ls='None', ms=10, mew=2, label='Initialer Schwerpunkt'),
            Line2D([0],[0], marker='+', color=COLOR_FINAL_GEOM, ls='None', ms=12, mew=2, label='Finaler Schwerpunkt'),
            Line2D([0],[0], ls='-.', color=COLOR_FINAL_GEOM, lw=1, label='Symmetrieachse (optimiert)')
        ]
        ax.legend(handles=custom_handles, loc='lower right', fontsize='small')
        
        ax.set_title(f'Legendre-Analyse: Frame {frame_number} (Differential Evolution)')
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        
        save_dir = os.path.join(self.config.OUTPUT_BASE_DIR, "debug_images", "legendre_summary")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"summary_{frame_number:04d}.png"), dpi=200)
        plt.close(fig)

    def _plot_radius_fit(self, angles: np.ndarray, radii: np.ndarray, state: str, 
                         frame_number: int, plot_coeffs: Optional[FitCoeffs] = None):
        coeffs_to_use = plot_coeffs
        if plot_coeffs is None:
            coeffs_to_use, success = self._fit_legendre_polynomials(angles, radii)
            if not success: return

        save_dir = os.path.join(self.config.OUTPUT_BASE_DIR, "debug_images", "legendre_radius_fits")
        save_path = os.path.join(save_dir, f"fit_{frame_number:04d}_{state}.png")
        
        self._create_combined_plot(
            angles_rad=angles, radii=radii, plot_coeffs=coeffs_to_use,
            fit_function=legendre_polynomial_series, save_path=save_path,
            cartesian_xlabel=f"Theta (rad) - {state.title()}"
        )

    def _create_combined_plot(self, angles_rad: np.ndarray, radii: np.ndarray, 
                              plot_coeffs: Optional[FitCoeffs], fit_function: callable, 
                              save_path: str, cartesian_xlabel: str):
        if plot_coeffs is None: return
        COLOR_DATA, COLOR_FIT = '#0066cc', '#009933'
        FIG_WIDTH_CM, FIG_HEIGHT_CM = 16, 8
        fig = plt.figure(figsize=(FIG_WIDTH_CM / 2.54, FIG_HEIGHT_CM / 2.54), dpi=200)
        ax_polar = fig.add_subplot(1, 2, 1, projection='polar') 
        ax_cartesian = fig.add_subplot(1, 2, 2)                 
        
        domain = np.linspace(-np.pi, np.pi, 400)
        fit_radii = fit_function(domain, *plot_coeffs)
        fit_radii_clipped = np.maximum(0, fit_radii) 

        ax_polar.plot(angles_rad, radii, 'o', color=COLOR_DATA, markersize=2.5, alpha=0.8, label='Datenpunkte')
        ax_polar.plot(domain, fit_radii_clipped, '-', color=COLOR_FIT, linewidth=1.5, label='Legendre-Fit')
        ax_polar.set_theta_zero_location('N'); ax_polar.set_theta_direction(-1)
        ax_polar.tick_params(labelsize=9, pad=5)

        ax_cartesian.plot(angles_rad, radii, 'o', color=COLOR_DATA, markersize=2.5, alpha=0.8)
        ax_cartesian.plot(domain, fit_radii, '-', color=COLOR_FIT, linewidth=1.5)
        ax_cartesian.set_xlabel(cartesian_xlabel, fontsize=11)
        ax_cartesian.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
        ax_cartesian.xaxis.set_major_locator(mticker.MultipleLocator(base=np.pi / 2))
        ax_cartesian.xaxis.set_major_formatter(mticker.FuncFormatter(format_radians_as_pi))
        ax_cartesian.set_xlim(-np.pi, np.pi)

        if radii.size > 0:
            min_r, max_r = np.min(radii), np.max(radii)
            padding = (max_r - min_r) * 0.1
            ax_polar.set_ylim(0, max_r + padding)
            ax_cartesian.set_ylim(min_r - padding, max_r + padding)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)