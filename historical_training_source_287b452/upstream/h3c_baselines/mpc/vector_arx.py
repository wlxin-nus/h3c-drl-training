"""One shared vector-ARX structure for all three BOPTEST cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ArxLayout:
    zones: tuple[str, ...]
    disturbance_names: tuple[str, ...]
    lag_count: int = 4
    horizon_steps: int = 4

    @property
    def output_dimension(self) -> int:
        return len(self.zones) + 1

    @property
    def control_dimension(self) -> int:
        return len(self.zones)

    @property
    def feature_dimension(self) -> int:
        return self.lag_count * (self.output_dimension + self.control_dimension) + len(
            self.disturbance_names
        )


@dataclass(frozen=True)
class Scaling:
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    output_mean: NDArray[np.float64]
    output_scale: NDArray[np.float64]


@dataclass(frozen=True)
class FittedArxModel:
    layout: ArxLayout
    intercept: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    scaling: Scaling
    ridge_alpha: float
    identity: str
    pmv_robust_margin: float = 0.0

    def predict_next(
        self,
        output_history: NDArray[np.float64],
        control_history: NDArray[np.float64],
        disturbance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        feature = feature_row(self.layout, output_history, control_history, disturbance)
        standardized = (feature - self.scaling.feature_mean) / self.scaling.feature_scale
        prediction = self.intercept + standardized @ self.coefficients
        return prediction * self.scaling.output_scale + self.scaling.output_mean

    def rollout(
        self,
        output_history: NDArray[np.float64],
        control_history: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        disturbances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], int]:
        if future_controls.shape != (self.layout.horizon_steps, self.layout.control_dimension):
            raise ValueError("future control matrix has the wrong shape")
        if disturbances.shape != (self.layout.horizon_steps, len(self.layout.disturbance_names)):
            raise ValueError("future disturbance matrix has the wrong shape")
        outputs = np.asarray(output_history, dtype=np.float64).copy()
        controls = np.asarray(control_history, dtype=np.float64).copy()
        predictions: list[NDArray[np.float64]] = []
        negative_power_count = 0
        for horizon in range(self.layout.horizon_steps):
            candidate_controls = controls.copy()
            candidate_controls[0] = future_controls[horizon]
            predicted = self.predict_next(outputs, candidate_controls, disturbances[horizon])
            if predicted[-1] < 0:
                negative_power_count += 1
                predicted = predicted.copy()
                predicted[-1] = 0.0
            predictions.append(predicted)
            outputs = np.vstack((predicted, outputs[:-1]))
            controls = np.vstack((future_controls[horizon], controls[:-1]))
        return np.vstack(predictions), negative_power_count

    def rollout_unclipped(
        self,
        output_history: NDArray[np.float64],
        control_history: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        disturbances: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Roll out the affine ARX dynamics without altering predicted power."""
        if future_controls.shape != (self.layout.horizon_steps, self.layout.control_dimension):
            raise ValueError("future control matrix has the wrong shape")
        if disturbances.shape != (self.layout.horizon_steps, len(self.layout.disturbance_names)):
            raise ValueError("future disturbance matrix has the wrong shape")
        outputs = np.asarray(output_history, dtype=np.float64).copy()
        controls = np.asarray(control_history, dtype=np.float64).copy()
        predictions: list[NDArray[np.float64]] = []
        for horizon in range(self.layout.horizon_steps):
            candidate_controls = controls.copy()
            candidate_controls[0] = future_controls[horizon]
            predicted = self.predict_next(outputs, candidate_controls, disturbances[horizon])
            predictions.append(predicted)
            outputs = np.vstack((predicted, outputs[:-1]))
            controls = np.vstack((future_controls[horizon], controls[:-1]))
        return np.vstack(predictions)

    def affine_rollout(
        self,
        output_history: NDArray[np.float64],
        control_history: NDArray[np.float64],
        disturbances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``vec(y_future) = offset + response @ vec(u_future)``."""
        horizon = self.layout.horizon_steps
        controls = self.layout.control_dimension
        baseline = np.zeros((horizon, controls), dtype=np.float64)
        offset = self.rollout_unclipped(
            output_history, control_history, baseline, disturbances
        ).reshape(-1)
        response = np.empty((offset.size, horizon * controls), dtype=np.float64)
        for column in range(horizon * controls):
            basis = baseline.copy().reshape(-1)
            basis[column] = 1.0
            prediction = self.rollout_unclipped(
                output_history,
                control_history,
                basis.reshape(horizon, controls),
                disturbances,
            ).reshape(-1)
            response[:, column] = prediction - offset
        return offset, response

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            zones=np.asarray(self.layout.zones),
            disturbance_names=np.asarray(self.layout.disturbance_names),
            lag_count=self.layout.lag_count,
            horizon_steps=self.layout.horizon_steps,
            intercept=self.intercept,
            coefficients=self.coefficients,
            feature_mean=self.scaling.feature_mean,
            feature_scale=self.scaling.feature_scale,
            output_mean=self.scaling.output_mean,
            output_scale=self.scaling.output_scale,
            ridge_alpha=self.ridge_alpha,
            identity=self.identity,
            pmv_robust_margin=self.pmv_robust_margin,
        )

    @classmethod
    def load(cls, path: Path) -> FittedArxModel:
        with np.load(path, allow_pickle=False) as source:
            layout = ArxLayout(
                tuple(str(value) for value in source["zones"].tolist()),
                tuple(str(value) for value in source["disturbance_names"].tolist()),
                int(source["lag_count"]),
                int(source["horizon_steps"]),
            )
            return cls(
                layout,
                np.asarray(source["intercept"], dtype=np.float64),
                np.asarray(source["coefficients"], dtype=np.float64),
                Scaling(
                    np.asarray(source["feature_mean"], dtype=np.float64),
                    np.asarray(source["feature_scale"], dtype=np.float64),
                    np.asarray(source["output_mean"], dtype=np.float64),
                    np.asarray(source["output_scale"], dtype=np.float64),
                ),
                float(source["ridge_alpha"]),
                str(source["identity"]),
                (
                    float(source["pmv_robust_margin"])
                    if "pmv_robust_margin" in source.files
                    else 0.0
                ),
            )


def feature_row(
    layout: ArxLayout,
    output_history: NDArray[np.float64],
    control_history: NDArray[np.float64],
    disturbance: NDArray[np.float64],
) -> NDArray[np.float64]:
    if output_history.shape != (layout.lag_count, layout.output_dimension):
        raise ValueError("ARX output history has the wrong shape")
    if control_history.shape != (layout.lag_count, layout.control_dimension):
        raise ValueError("ARX control history has the wrong shape")
    if disturbance.shape != (len(layout.disturbance_names),):
        raise ValueError("ARX disturbance has the wrong shape")
    row = np.concatenate((output_history.reshape(-1), control_history.reshape(-1), disturbance))
    if row.shape != (layout.feature_dimension,) or np.any(~np.isfinite(row)):
        raise ValueError("ARX feature row is invalid")
    return row


def build_dataset(
    layout: ArxLayout,
    times: NDArray[np.int64],
    outputs: NDArray[np.float64],
    controls: NDArray[np.float64],
    disturbances: NDArray[np.float64],
    *,
    step_seconds: int = 900,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    row_count = len(times)
    if (
        outputs.shape != (row_count, layout.output_dimension)
        or controls.shape != (row_count, layout.control_dimension)
        or disturbances.shape != (row_count, len(layout.disturbance_names))
    ):
        raise ValueError("ARX source arrays are not aligned")
    if np.any(np.diff(times) != step_seconds):
        raise ValueError("ARX source timeline is not continuous")
    if any(np.any(~np.isfinite(array)) for array in (outputs, controls, disturbances)):
        raise ValueError("ARX source data contains non-finite values")
    features: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    target_times: list[int] = []
    for current in range(layout.lag_count - 1, row_count - 1):
        output_history = np.vstack([outputs[current - lag] for lag in range(layout.lag_count)])
        control_history = np.vstack([controls[current - lag] for lag in range(layout.lag_count)])
        features.append(feature_row(layout, output_history, control_history, disturbances[current]))
        targets.append(outputs[current + 1])
        target_times.append(int(times[current + 1]))
    return np.vstack(features), np.vstack(targets), np.asarray(target_times, dtype=np.int64)


def _scaling(features: NDArray[np.float64], outputs: NDArray[np.float64]) -> Scaling:
    feature_scale = features.std(axis=0)
    output_scale = outputs.std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    output_scale[output_scale < 1e-12] = 1.0
    return Scaling(features.mean(axis=0), feature_scale, outputs.mean(axis=0), output_scale)


def _fit(
    features: NDArray[np.float64], outputs: NDArray[np.float64], alpha: float, scaling: Scaling
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = (features - scaling.feature_mean) / scaling.feature_scale
    y = (outputs - scaling.output_mean) / scaling.output_scale
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return coefficients[0], coefficients[1:]


def _identity(
    layout: ArxLayout,
    intercept: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    scaling: Scaling,
    ridge_alpha: float,
    *,
    pmv_robust_margin: float = 0.0,
) -> str:
    identity_payload: dict[str, Any] = {
        "zones": layout.zones,
        "disturbance_names": layout.disturbance_names,
        "lag_count": layout.lag_count,
        "horizon_steps": layout.horizon_steps,
        "ridge_alpha": ridge_alpha,
        "intercept": intercept.tolist(),
        "coefficients": coefficients.tolist(),
        "scaling": {
            "feature_mean": scaling.feature_mean.tolist(),
            "feature_scale": scaling.feature_scale.tolist(),
            "output_mean": scaling.output_mean.tolist(),
            "output_scale": scaling.output_scale.tolist(),
        },
    }
    # Preserve every pre-refit model identity byte-for-byte.  A nonzero calibrated
    # margin is part of the new candidate identity because it changes optimization.
    if pmv_robust_margin != 0.0:
        identity_payload["pmv_robust_margin"] = pmv_robust_margin
    return hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def with_pmv_robust_margin(model: FittedArxModel, margin: float) -> FittedArxModel:
    value = float(margin)
    if not np.isfinite(value) or value < 0.0 or value >= 0.5:
        raise ValueError("PMV robust margin must be finite and within [0, 0.5)")
    identity = _identity(
        model.layout,
        model.intercept,
        model.coefficients,
        model.scaling,
        model.ridge_alpha,
        pmv_robust_margin=value,
    )
    return FittedArxModel(
        model.layout,
        model.intercept,
        model.coefficients,
        model.scaling,
        model.ridge_alpha,
        identity,
        value,
    )


def expected_model_identity(model: FittedArxModel) -> str:
    """Recompute the immutable bundle identity instead of trusting stored text."""
    return _identity(
        model.layout,
        model.intercept,
        model.coefficients,
        model.scaling,
        model.ridge_alpha,
        pmv_robust_margin=model.pmv_robust_margin,
    )


def fit_vector_arx(
    layout: ArxLayout,
    fit_features: NDArray[np.float64],
    fit_outputs: NDArray[np.float64],
    *,
    holdout_features: NDArray[np.float64],
    holdout_outputs: NDArray[np.float64],
    alpha_candidates: tuple[float, ...],
) -> tuple[FittedArxModel, dict[str, Any]]:
    if (
        len(fit_features) < 2
        or len(holdout_features) < 1
        or fit_features.shape[1:] != holdout_features.shape[1:]
        or fit_outputs.shape[1:] != holdout_outputs.shape[1:]
        or len(fit_features) != len(fit_outputs)
        or len(holdout_features) != len(holdout_outputs)
    ):
        raise ValueError("ARX fit and whole-episode holdout data are invalid")
    training_scaling = _scaling(fit_features, fit_outputs)
    scores: dict[float, float] = {}
    for alpha in alpha_candidates:
        if not math_is_valid_positive(alpha):
            raise ValueError("ridge alpha candidates must be finite and positive")
        intercept, coefficients = _fit(fit_features, fit_outputs, alpha, training_scaling)
        valid_standardized = (
            holdout_features - training_scaling.feature_mean
        ) / training_scaling.feature_scale
        prediction = intercept + valid_standardized @ coefficients
        truth = (holdout_outputs - training_scaling.output_mean) / training_scaling.output_scale
        scores[alpha] = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    best_score = min(scores.values())
    chosen = max(alpha for alpha, score in scores.items() if abs(score - best_score) <= 1e-12)
    intercept, coefficients = _fit(fit_features, fit_outputs, chosen, training_scaling)
    identity = _identity(layout, intercept, coefficients, training_scaling, chosen)
    model = FittedArxModel(layout, intercept, coefficients, training_scaling, chosen, identity)
    report = {
        "schema": "h3c_vector_arx_fit",
        "schema_version": 1,
        "selection_metric": "mean_standardized_output_rmse",
        "holdout_rows": len(holdout_features),
        "training_rows": len(fit_features),
        "total_rows": len(fit_features) + len(holdout_features),
        "final_fit_includes_holdout": False,
        "scores": {str(alpha): scores[alpha] for alpha in sorted(scores)},
        "selected_alpha": chosen,
        "selected_score": best_score,
        "tie_break": "larger_alpha",
        "model_identity": identity,
    }
    return model, report


def math_is_valid_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)
