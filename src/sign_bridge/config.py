"""Typed configuration for SignBridge.

Every tunable lives in ``config.yaml`` at the project root and is parsed
into the pydantic models below. Loading fails loudly on unknown keys or
out-of-range values, so a typo in YAML surfaces at startup, not mid-demo.

Resolution order for the config file:
    1. Explicit ``path`` argument.
    2. ``SIGN_BRIDGE_CONFIG`` environment variable.
    3. ``config.yaml`` found by walking up from the current directory.
    4. ``config.yaml`` in the project root of an editable install.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Language = Literal["asl", "psl"]
Architecture = Literal["lstm", "transformer", "conv_transformer"]


class _Strict(BaseModel):
    """Base model: reject unknown keys so YAML typos can't hide."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AppSection(_Strict):
    name: str = "SignBridge"
    version: str = "0.7.1"
    default_language: Language = "asl"


class PathsSection(_Strict):
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    video_dir: Path = Path("assets/sign_videos")
    recordings_dir: Path = Path("recordings")
    logs_dir: Path = Path("logs")


class CaptureSection(_Strict):
    camera_index: int = Field(0, ge=0)
    frame_width: int = Field(640, ge=160)
    frame_height: int = Field(480, ge=120)
    target_fps: int = Field(30, ge=5, le=120)
    mirror: bool = True


class AudioSection(_Strict):
    sample_rate: int = Field(16_000, ge=8_000)
    channels: int = Field(1, ge=1, le=2)
    max_record_seconds: float = Field(15.0, gt=0)


class LandmarksSection(_Strict):
    window_size: int = Field(30, ge=5)
    feature_dim: int = Field(162, ge=6)
    min_detection_confidence: float = Field(0.5, ge=0, le=1)
    min_tracking_confidence: float = Field(0.5, ge=0, le=1)
    model_complexity: int = Field(1, ge=0, le=2)


class TransformerSection(_Strict):
    d_model: int = Field(96, ge=8)
    num_heads: int = Field(4, ge=1)
    ff_dim: int = Field(192, ge=8)
    num_blocks: int = Field(2, ge=1)


class ConvTransformerSection(_Strict):
    dim: int = Field(192, ge=8)
    kernel_size: int = Field(17, ge=3)
    num_stacks: int = Field(2, ge=1)
    convs_per_stack: int = Field(3, ge=1)
    num_heads: int = Field(4, ge=1)
    expand: int = Field(2, ge=1)
    drop_path: float = Field(0.2, ge=0, lt=1)
    late_dropout: float = Field(0.5, ge=0, lt=1)
    weight_decay: float = Field(0.05, ge=0)
    label_smoothing: float = Field(0.1, ge=0, lt=1)


class ModelSection(_Strict):
    architecture: Architecture = "lstm"
    lstm_units: tuple[int, ...] = (128, 64)
    dense_units: int = Field(64, ge=4)
    dropout: float = Field(0.3, ge=0, lt=1)
    transformer: TransformerSection = TransformerSection()
    conv_transformer: ConvTransformerSection = ConvTransformerSection()


class TrainingSection(_Strict):
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(120, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    validation_split: float = Field(0.15, gt=0, lt=0.5)
    test_split: float = Field(0.15, gt=0, lt=0.5)
    cv_folds: int = Field(5, ge=2)
    early_stopping_patience: int = Field(12, ge=1)
    reduce_lr_patience: int = Field(5, ge=1)
    reduce_lr_factor: float = Field(0.5, gt=0, lt=1)
    use_class_weights: bool = True
    augment: bool = True
    seed: int = 42


class AugmentationSection(_Strict):
    rotation_deg: float = Field(15.0, ge=0)
    scale_pct: float = Field(10.0, ge=0)
    mirror: bool = True
    temporal_jitter_frames: int = Field(3, ge=0)
    copies_per_sample: int = Field(2, ge=0)


class InferenceSection(_Strict):
    predict_every_n_frames: int = Field(5, ge=1)
    confidence_threshold: float = Field(0.80, ge=0, le=1)
    lock_frames: int = Field(10, ge=1)
    cooldown_frames: int = Field(15, ge=0)
    window_seconds: float = Field(2.0, gt=0)
    hand_grace_seconds: float = Field(0.4, ge=0)
    sentence_pause_seconds: float = Field(1.5, gt=0)
    # How live frames become model inputs: "window" = sliding motion window
    # (models trained on fixed-length recordings); "segment" = one guess per
    # finished sign, found from hand motion (models trained on whole-sign
    # clips, e.g. Google ASL Signs); "auto" picks from the model's metadata.
    mode: Literal["auto", "window", "segment"] = "auto"
    segment_confidence: float = Field(0.35, ge=0, le=1)
    segment_margin: float = Field(0.05, ge=0, le=1)
    motion_start_speed: float = Field(0.6, gt=0)
    motion_stop_speed: float = Field(0.15, gt=0)
    hold_seconds: float = Field(0.4, gt=0)
    min_sign_seconds: float = Field(0.05, gt=0)
    max_sign_seconds: float = Field(6.0, gt=0)
    merge_gap_seconds: float = Field(0.3, ge=0)
    repeat_window_seconds: float = Field(1.0, ge=0)


class CollectorSection(_Strict):
    samples_per_label: int = Field(50, ge=1)
    clip_seconds: float = Field(2.0, gt=0)
    countdown_seconds: int = Field(3, ge=0)
    min_hands_visible: float = Field(0.6, ge=0, le=1)


class SttSection(_Strict):
    whisper_model: Literal["tiny", "base", "small", "medium"] = "base"
    language: str = "en"


class TtsSection(_Strict):
    engine: Literal["gtts", "pyttsx3"] = "gtts"
    gtts_lang: str = "en"
    pyttsx3_rate: int = Field(170, ge=80, le=400)


class UiSection(_Strict):
    theme: str = "field_guide"
    high_contrast_default: bool = False
    base_font_px: int = Field(18, ge=12, le=32)
    show_fps: bool = True
    captions_always_on: bool = True


class AiSection(_Strict):
    """AI sentence writer (signs → natural English); see language/sentence_ai.py."""

    enabled: bool = True
    provider: Literal["gemini", "anthropic", "openai", "none"] = "gemini"
    model: str = ""                      # empty = the provider's default model
    max_options: int = Field(3, ge=1, le=5)
    timeout_seconds: float = Field(10.0, gt=0)
    secrets_file: str = ".streamlit/secrets.toml"


class AppConfig(_Strict):
    """Root configuration object. Access via :func:`load_config`."""

    app: AppSection = AppSection()
    paths: PathsSection = PathsSection()
    capture: CaptureSection = CaptureSection()
    audio: AudioSection = AudioSection()
    landmarks: LandmarksSection = LandmarksSection()
    model: ModelSection = ModelSection()
    training: TrainingSection = TrainingSection()
    augmentation: AugmentationSection = AugmentationSection()
    inference: InferenceSection = InferenceSection()
    collector: CollectorSection = CollectorSection()
    stt: SttSection = SttSection()
    tts: TtsSection = TtsSection()
    ui: UiSection = UiSection()
    ai: AiSection = AiSection()

    def resolve(self, relative: Path) -> Path:
        """Resolve a configured path against the project root (config.yaml's dir)."""
        root = _find_config_path()
        base = root.parent if root is not None else Path.cwd()
        return (base / relative).resolve()


def _find_config_path() -> Path | None:
    """Walk up from CWD looking for config.yaml; honor env override."""
    env = os.environ.get("SIGN_BRIDGE_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        cfg = candidate / "config.yaml"
        if cfg.is_file():
            return cfg
    # Editable installs: fall back to the project root next to ``src/``, so
    # ``sign-bridge`` works no matter which folder the terminal is in.
    project_cfg = Path(__file__).resolve().parents[2] / "config.yaml"
    if project_cfg.is_file():
        return project_cfg
    return None


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> AppConfig:
    """Load and validate configuration.

    Args:
        path: Optional explicit path to a YAML file. When omitted, the
            file is discovered (env var, then upward search). When nothing
            is found, library defaults apply — useful in tests.

    Returns:
        A frozen, validated :class:`AppConfig`.
    """
    cfg_path = Path(path).expanduser().resolve() if path else _find_config_path()
    if cfg_path is None:
        return AppConfig()
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)
