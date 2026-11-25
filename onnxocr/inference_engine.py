import os
import platform
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import onnxruntime

from .logger import get_logger

log = get_logger("inference_engine")


Provider = Union[str, Tuple[str, Dict[str, Any]]]


InferenceSession = onnxruntime.InferenceSession
SessionOptions = onnxruntime.SessionOptions
GraphOptimizationLevel = onnxruntime.GraphOptimizationLevel


class EP(Enum):
    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    DIRECTML = "DmlExecutionProvider"
    CANN = "CANNExecutionProvider"


DEFAULT_CPU_EP_CFG: Dict[str, Any] = {}
DEFAULT_CUDA_EP_CFG: Dict[str, Any] = {
    "cudnn_conv_algo_search": "DEFAULT",
    "device_id": 0,
}
DEFAULT_DML_EP_CFG: Dict[str, Any] = {}
DEFAULT_CANN_EP_CFG: Dict[str, Any] = {}


def get_available_providers() -> List[str]:
    return onnxruntime.get_available_providers()


def get_device() -> str:
    return onnxruntime.get_device()


def is_session(value: Any) -> bool:
    return isinstance(value, InferenceSession)


def build_providers(
    use_gpu: bool = False,
    gpu_id: int = 0,
    providers: Optional[Sequence[Provider]] = None,
) -> List[Provider]:
    if providers is not None:
        return list(providers)

    if use_gpu:
        cuda_cfg = dict(DEFAULT_CUDA_EP_CFG)
        cuda_cfg["device_id"] = gpu_id
        return [(EP.CUDA.value, cuda_cfg), EP.CPU.value]
    return [EP.CPU.value]


def build_providers_from_engine_cfg(engine_cfg: Any) -> List[Provider]:
    """Build ONNXRuntime execution providers from the common engine_cfg shape.

    RapidTable, RapidLayout and RapidDoc use slightly different vendored copies of
    an ``engine_cfg`` object. This function is the single compatibility point so
    downstream vendors only need to customize provider behavior here.
    """

    available = get_available_providers()
    providers: List[Provider] = [(EP.CPU.value, _cfg_dict(engine_cfg, "cpu_ep_cfg", DEFAULT_CPU_EP_CFG))]

    if _cfg_bool(engine_cfg, "use_cuda") and EP.CUDA.value in available:
        providers.insert(0, (EP.CUDA.value, _cfg_dict(engine_cfg, "cuda_ep_cfg", DEFAULT_CUDA_EP_CFG)))

    if _cfg_bool(engine_cfg, "use_dml") and _is_windows() and EP.DIRECTML.value in available:
        providers.insert(0, (EP.DIRECTML.value, _cfg_dict(engine_cfg, "dm_ep_cfg", DEFAULT_DML_EP_CFG)))

    if _cfg_bool(engine_cfg, "use_cann") and EP.CANN.value in available:
        providers.insert(0, (EP.CANN.value, _cfg_dict(engine_cfg, "cann_ep_cfg", DEFAULT_CANN_EP_CFG)))

    return providers


class ProviderConfig:
    """Compatibility wrapper used by vendored RapidAI modules."""

    def __init__(self, engine_cfg: Any):
        self.engine_cfg = engine_cfg

    def get_ep_list(self) -> List[Provider]:
        return build_providers_from_engine_cfg(self.engine_cfg)

    def verify_providers(self, session_providers: Sequence[str]) -> None:
        if not session_providers:
            raise ValueError("Session providers is empty.")


def _cfg_bool(engine_cfg: Any, key: str, default: bool = False) -> bool:
    value = _cfg_get(engine_cfg, key, default)
    return bool(value)


def _cfg_dict(engine_cfg: Any, key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(default)
    value = _cfg_get(engine_cfg, key, default)
    if value is not None:
        result.update(dict(value))

    prefix = f"{key}."
    if hasattr(engine_cfg, "items"):
        for cfg_key, cfg_value in engine_cfg.items():
            if isinstance(cfg_key, str) and cfg_key.startswith(prefix):
                result[cfg_key[len(prefix) :]] = cfg_value
    return result


def _cfg_get(engine_cfg: Any, key: str, default: Any = None) -> Any:
    if engine_cfg is None:
        return default
    if hasattr(engine_cfg, "get"):
        return engine_cfg.get(key, default)
    return getattr(engine_cfg, key, default)


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    parsed = int(value.strip())
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0, got {parsed}")
    return parsed


def _env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _default_session_options() -> SessionOptions:
    """Build a SessionOptions with graph optimization and memory pattern enabled.

    These settings provide 10-30% CPU inference speedup with no accuracy impact.
    """
    log_severity = int(os.getenv("ONNXOCR_ORT_LOG_SEVERITY", "3"))
    onnxruntime.set_default_logger_severity(log_severity)
    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.log_severity_level = log_severity

    intra_op = _env_int("ONNXOCR_ORT_INTRA_OP_THREADS")
    if intra_op is not None and intra_op > 0:
        opts.intra_op_num_threads = intra_op

    inter_op = _env_int("ONNXOCR_ORT_INTER_OP_THREADS")
    if inter_op is not None and inter_op > 0:
        opts.inter_op_num_threads = inter_op

    exec_mode = os.getenv("ONNXOCR_ORT_EXECUTION_MODE", "").strip().lower()
    if exec_mode == "sequential" or (not exec_mode and inter_op == 1):
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    elif exec_mode == "parallel":
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL

    allow_spinning = _env_bool("ONNXOCR_ORT_ALLOW_SPINNING")
    if allow_spinning is not None:
        flag = "1" if allow_spinning else "0"
        opts.add_session_config_entry("session.intra_op.allow_spinning", flag)
        opts.add_session_config_entry("session.inter_op.allow_spinning", flag)

    return opts


class _CpuArenaShrinkSession(InferenceSession):
    """Return unused CPU arena regions after inference while keeping session reuse."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cpu_arena_run_options = None
        if self.get_session_options().enable_cpu_mem_arena and EP.CPU.value in self.get_providers():
            self._cpu_arena_run_options = onnxruntime.RunOptions()
            self._cpu_arena_run_options.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", "cpu:0"
            )
            log.info("CPU arena shrinkage enabled")

    def run(self, output_names, input_feed, run_options=None):
        if run_options is None:
            run_options = self._cpu_arena_run_options
        return super().run(output_names, input_feed, run_options)


def create_session(
    model_path: str,
    providers: Optional[Sequence[Provider]] = None,
    use_gpu: bool = False,
    gpu_id: int = 0,
    sess_options: Optional[SessionOptions] = None,
) -> InferenceSession:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Please download models first: python scripts/download_models.py"
        )
    if sess_options is None:
        sess_options = _default_session_options()
    session_providers = build_providers(
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        providers=providers,
    )
    log.info("Creating ONNX session: {}, providers={}", model_path, session_providers)
    session_cls = _CpuArenaShrinkSession if _env_bool("ONNXOCR_CPU_ARENA_SHRINK") else InferenceSession
    try:
        return session_cls(
            model_path,
            sess_options=sess_options,
            providers=session_providers,
        )
    except Exception as e:
        log.error("Failed to load model: {}, error: {}", model_path, e)
        raise
