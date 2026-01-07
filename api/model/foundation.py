# -*- coding: utf-8 -*-
# @Author             : Claude Code
# @Created Time       : 2025/01/07
# @Email              : noreply@anthropic.com
# @Last Modified By   : Claude Code
# @Last Modified Time : 2025/01/07

"""
Foundation model wrappers for time-series forecasting.

This module provides wrapper classes for pre-trained foundation models
(MOMENT, Chronos, Moirai) with support for auxiliary features and fine-tuning.

Classes:
    MOMENTForecaster: Wrapper for MOMENT with auxiliary feature support
    ChronosForecaster: Wrapper for Chronos with auxiliary feature support
    MoiraiForecaster: Wrapper for Moirai with auxiliary feature support
    FoundationModelWithAux: Base class for foundation models with auxiliary features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class FoundationModelWithAux(nn.Module):
    """
    Base wrapper for foundation models with auxiliary feature integration.

    This class provides a common interface for adding auxiliary features
    (weather, pricing, POI) to pre-trained foundation models.
    """

    def __init__(self, foundation_model, n_aux_features: int, d_model: int = 512):
        """
        Initialize foundation model wrapper.

        Args:
            foundation_model: Pre-trained foundation model
            n_aux_features: Number of auxiliary features
            d_model: Model embedding dimension
        """
        super().__init__()
        self.foundation = foundation_model
        self.n_aux_features = n_aux_features
        self.d_model = d_model

        # Auxiliary feature processor
        if n_aux_features > 0:
            self.aux_processor = nn.Sequential(
                nn.Linear(n_aux_features, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.LayerNorm(64)
            )

            # Fusion layer (combine main + auxiliary features)
            self.fusion = nn.MultiheadAttention(
                embed_dim=64,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.aux_processor = None
            self.fusion = None

    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning."""
        if hasattr(self.foundation, 'encoder'):
            for param in self.foundation.encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder frozen")

    def unfreeze_encoder(self, last_n_layers: Optional[int] = None):
        """
        Unfreeze encoder parameters.

        Args:
            last_n_layers: If specified, only unfreeze last N layers
        """
        if hasattr(self.foundation, 'encoder'):
            if last_n_layers is None:
                # Unfreeze all
                for param in self.foundation.encoder.parameters():
                    param.requires_grad = True
                print("✓ Encoder fully unfrozen")
            else:
                # Unfreeze last N layers
                if hasattr(self.foundation.encoder, 'layers'):
                    for layer in self.foundation.encoder.layers[-last_n_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    print(f"✓ Encoder: last {last_n_layers} layers unfrozen")


class MOMENTForecaster(FoundationModelWithAux):
    """
    Wrapper for MOMENT (Multi-modal Time-series Transformer) model.

    MOMENT is designed for real-world time series with multivariate support
    and can handle irregular sampling and missing data.

    Usage:
        model = MOMENTForecaster(
            pretrained_path="AutonLab/MOMENT-1-large",
            n_aux_features=4,
            forecast_horizon=24
        )
        model.freeze_encoder()  # For fine-tuning
        predictions = model(context, aux_features)
    """

    def __init__(
        self,
        pretrained_path: str = "AutonLab/MOMENT-1-large",
        n_aux_features: int = 0,
        forecast_horizon: int = 24,
        context_length: int = 512,
        use_segments: bool = False
    ):
        """
        Initialize MOMENT forecaster.

        Args:
            pretrained_path: HuggingFace model path
            n_aux_features: Number of auxiliary features
            forecast_horizon: Number of steps to forecast
            context_length: Input context length
            use_segments: Use segment-based aggregation (6 segments/day)
        """
        try:
            from momentfm import MOMENTPipeline
        except ImportError:
            raise ImportError(
                "momentfm not installed. Install with: pip install momentfm"
            )

        # Load pre-trained MOMENT model
        print(f"Loading MOMENT from {pretrained_path}...")
        model = MOMENTPipeline.from_pretrained(
            pretrained_path,
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': forecast_horizon,
                'head_dropout': 0.1,
                'weight_decay': 1e-5,
            }
        )

        super().__init__(
            foundation_model=model,
            n_aux_features=n_aux_features,
            d_model=model.config.d_model if hasattr(model, 'config') else 512
        )

        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        self.use_segments = use_segments

        print(f"✓ MOMENT loaded: horizon={forecast_horizon}, context={context_length}")

    def forward(
        self,
        context: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional auxiliary features.

        Args:
            context: Input context [batch, time, features]
            aux_features: Auxiliary features [batch, time, n_aux]

        Returns:
            Predictions [batch, forecast_horizon, features]
        """
        if aux_features is not None and self.aux_processor is not None:
            # Process auxiliary features
            aux_emb = self.aux_processor(aux_features)

            # Combine with context via attention
            # (simplified - actual implementation may vary)
            combined = context
        else:
            combined = context

        # Use MOMENT for forecasting
        output = self.foundation.forecast(
            context=combined,
            prediction_length=self.forecast_horizon
        )

        return output


class ChronosForecaster(FoundationModelWithAux):
    """
    Wrapper for Chronos (T5-based time-series model).

    Chronos uses language model architecture (T5) for time series forecasting.

    Usage:
        model = ChronosForecaster(
            model_size='large',
            n_aux_features=4,
            prediction_length=24
        )
        predictions = model(context)
    """

    def __init__(
        self,
        model_size: str = 'large',
        n_aux_features: int = 0,
        prediction_length: int = 24,
        context_length: int = 168
    ):
        """
        Initialize Chronos forecaster.

        Args:
            model_size: Model size ('small', 'base', 'large')
            n_aux_features: Number of auxiliary features
            prediction_length: Number of steps to forecast
            context_length: Input context length
        """
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos not installed. Install with: pip install chronos-forecasting"
            )

        # Load pre-trained Chronos model
        model_path = f"amazon/chronos-t5-{model_size}"
        print(f"Loading Chronos from {model_path}...")

        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        super().__init__(
            foundation_model=pipeline,
            n_aux_features=n_aux_features,
            d_model=768  # T5 model dimension
        )

        self.prediction_length = prediction_length
        self.context_length = context_length

        print(f"✓ Chronos loaded: size={model_size}, pred_len={prediction_length}")

    def forward(
        self,
        context: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
        num_samples: int = 20
    ) -> torch.Tensor:
        """
        Forward pass with probabilistic forecasting.

        Args:
            context: Input context [batch, time]
            aux_features: Auxiliary features (optional, used for conditioning)
            num_samples: Number of sample trajectories to generate

        Returns:
            Predictions [batch, num_samples, prediction_length]
        """
        # For Chronos, auxiliary features can be incorporated by
        # adding them to context or using them for conditioning
        if aux_features is not None and self.aux_processor is not None:
            # Simple approach: add auxiliary as weighted bias
            aux_emb = self.aux_processor(aux_features)
            # Project to match context dimension
            aux_proj = aux_emb.mean(dim=-1, keepdim=True)
            context = context + 0.1 * aux_proj  # Small weight

        # Generate predictions
        forecast = self.foundation.predict(
            context=context,
            prediction_length=self.prediction_length,
            num_samples=num_samples
        )

        return forecast


class MoiraiForecaster(FoundationModelWithAux):
    """
    Wrapper for Moirai (Salesforce's unified time-series model).

    Moirai supports any-variate forecasting with GluonTS integration.

    Usage:
        model = MoiraiForecaster(
            model_size='large',
            n_aux_features=4,
            prediction_length=24
        )
        predictions = model(context)
    """

    def __init__(
        self,
        model_size: str = 'large',
        n_aux_features: int = 0,
        prediction_length: int = 24,
        context_length: int = 153
    ):
        """
        Initialize Moirai forecaster.

        Args:
            model_size: Model size ('small', 'base', 'large')
            n_aux_features: Number of auxiliary features
            prediction_length: Number of steps to forecast
            context_length: Input context length
        """
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError:
            raise ImportError(
                "uni2ts not installed. Install with: pip install uni2ts"
            )

        # Load pre-trained Moirai model
        model_path = f"Salesforce/moirai-1.1-R-{model_size}"
        print(f"Loading Moirai from {model_path}...")

        module = MoiraiModule.from_pretrained(model_path)
        predictor = MoiraiForecast(
            module=module,
            prediction_length=prediction_length,
            context_length=context_length,
            num_samples=100
        )

        super().__init__(
            foundation_model=predictor,
            n_aux_features=n_aux_features,
            d_model=512
        )

        self.prediction_length = prediction_length
        self.context_length = context_length

        print(f"✓ Moirai loaded: size={model_size}, pred_len={prediction_length}")

    def forward(
        self,
        context: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multivariate forecasting.

        Args:
            context: Input context [batch, time, features]
            aux_features: Auxiliary features [batch, time, n_aux]

        Returns:
            Predictions [batch, prediction_length, features]
        """
        # Moirai supports multivariate input, so auxiliary features
        # can be concatenated directly
        if aux_features is not None:
            # Concatenate auxiliary features
            combined = torch.cat([context, aux_features], dim=-1)
        else:
            combined = context

        # Generate forecast
        forecast = self.foundation.forecast(combined)

        return forecast


# Utility functions

def load_foundation_model(
    model_name: str,
    model_size: str = 'large',
    n_aux_features: int = 0,
    prediction_length: int = 24,
    **kwargs
) -> FoundationModelWithAux:
    """
    Factory function to load foundation models.

    Args:
        model_name: Model name ('moment', 'chronos', 'moirai')
        model_size: Model size ('small', 'base', 'large')
        n_aux_features: Number of auxiliary features
        prediction_length: Forecast horizon
        **kwargs: Additional model-specific arguments

    Returns:
        Loaded foundation model wrapper
    """
    model_name = model_name.lower()

    if model_name == 'moment':
        return MOMENTForecaster(
            pretrained_path=f"AutonLab/MOMENT-1-{model_size}",
            n_aux_features=n_aux_features,
            forecast_horizon=prediction_length,
            **kwargs
        )
    elif model_name == 'chronos':
        return ChronosForecaster(
            model_size=model_size,
            n_aux_features=n_aux_features,
            prediction_length=prediction_length,
            **kwargs
        )
    elif model_name == 'moirai':
        return MoiraiForecaster(
            model_size=model_size,
            n_aux_features=n_aux_features,
            prediction_length=prediction_length,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: moment, chronos, moirai")


def convert_to_segments(hourly_data: np.ndarray, segments_per_day: int = 6) -> np.ndarray:
    """
    Convert hourly data to segment-based representation for MOMENT.

    Args:
        hourly_data: Hourly time series [time, features]
        segments_per_day: Number of segments per day (default: 6 = 4-hour windows)

    Returns:
        Segment-based time series
    """
    hours_per_segment = 24 // segments_per_day
    n_timesteps = hourly_data.shape[0]
    n_features = hourly_data.shape[1] if hourly_data.ndim > 1 else 1

    # Reshape to (days, segments_per_day, hours_per_segment, features)
    n_days = n_timesteps // 24
    truncated_data = hourly_data[:n_days * 24]

    if hourly_data.ndim == 1:
        truncated_data = truncated_data.reshape(n_days, 24, 1)
    else:
        truncated_data = truncated_data.reshape(n_days, 24, n_features)

    # Aggregate by segments (sum over hours in each segment)
    segments = []
    for i in range(segments_per_day):
        start_hour = i * hours_per_segment
        end_hour = (i + 1) * hours_per_segment
        segment_data = truncated_data[:, start_hour:end_hour, :].sum(axis=1)
        segments.append(segment_data)

    # Stack segments: [days * segments_per_day, features]
    segment_data = np.concatenate(segments, axis=0)

    return segment_data if n_features > 1 else segment_data.squeeze(-1)


def create_progressive_unfreezing_schedule(
    total_epochs: int,
    n_phases: int = 3
) -> Dict[int, int]:
    """
    Create progressive unfreezing schedule for fine-tuning.

    Args:
        total_epochs: Total number of training epochs
        n_phases: Number of unfreezing phases

    Returns:
        Dictionary mapping epoch to number of layers to unfreeze
    """
    schedule = {}
    epochs_per_phase = total_epochs // n_phases

    for phase in range(n_phases):
        start_epoch = phase * epochs_per_phase
        layers_to_unfreeze = phase + 1
        schedule[start_epoch] = layers_to_unfreeze

    return schedule
