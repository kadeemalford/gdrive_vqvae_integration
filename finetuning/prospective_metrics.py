"""Prospective physical metrics; historical selectors remain unchanged."""
from dataclasses import dataclass

import numpy as np

VERSION = "physical-pooled-train-scale-v1"


@dataclass
class PhysicalAccumulator:
    """Merge sufficient statistics before forming ratios, one physical channel."""

    count: int = 0
    target_sum: float = 0.
    target_square: float = 0.
    prediction_sum: float = 0.
    prediction_square: float = 0.
    cross: float = 0.
    error_sum: float = 0.
    sse: float = 0.
    max_error: float = 0.
    target_mean: float = 0.
    prediction_mean: float = 0.
    target_m2: float = 0.
    prediction_m2: float = 0.
    covariance_sum: float = 0.

    def update(self, target, prediction):
        target = np.asarray(target, dtype=np.float64).ravel()
        prediction = np.asarray(prediction, dtype=np.float64).ravel()
        if not target.size or target.shape != prediction.shape:
            raise ValueError("Expected matching nonempty arrays")
        if not np.isfinite(target).all() or not np.isfinite(prediction).all():
            raise ValueError("Nonfinite cells require a separate exact-bit audit")
        error = prediction - target
        batch_n = target.size
        target_mean, prediction_mean = float(target.mean()), float(prediction.mean())
        centered_target, centered_prediction = target - target_mean, prediction - prediction_mean
        delta_target = target_mean - self.target_mean
        delta_prediction = prediction_mean - self.prediction_mean
        weight = self.count * batch_n / (self.count + batch_n)
        self.target_m2 += float(centered_target @ centered_target) + delta_target ** 2 * weight
        self.prediction_m2 += float(centered_prediction @ centered_prediction) + delta_prediction ** 2 * weight
        self.covariance_sum += float(centered_target @ centered_prediction) + delta_target * delta_prediction * weight
        self.target_mean += delta_target * batch_n / (self.count + batch_n)
        self.prediction_mean += delta_prediction * batch_n / (self.count + batch_n)
        self.count += target.size
        self.target_sum += float(target.sum())
        self.target_square += float(target @ target)
        self.prediction_sum += float(prediction.sum())
        self.prediction_square += float(prediction @ prediction)
        self.cross += float(target @ prediction)
        self.error_sum += float(error.sum())
        self.sse += float(error @ error)
        self.max_error = max(self.max_error, float(np.abs(error).max()))

    def result(self, reference_scale):
        if not self.count or not np.isfinite(reference_scale) or reference_scale <= 0:
            raise ValueError("Positive frozen physical reference scale required")
        n = self.count
        variance = max(0., self.target_m2 / n)
        pred_variance = max(0., self.prediction_m2 / n)
        covariance = self.covariance_sum / n
        correlation = None
        if min(variance, pred_variance) > reference_scale ** 2:
            correlation = float(np.clip(covariance / np.sqrt(variance * pred_variance), -1, 1))
        rmse = float(np.sqrt(self.sse / n))
        return {"cells": n, "sse": self.sse, "target_energy": self.target_square,
                "target_std": float(np.sqrt(variance)), "rmse": rmse,
                "nrmse_star": rmse / max(np.sqrt(variance), reference_scale),
                "bias": self.error_sum / n, "max_error": self.max_error,
                "correlation": correlation, "correlation_defined": correlation is not None}
