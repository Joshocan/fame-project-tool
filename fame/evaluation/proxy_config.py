from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ProxySelectorConfig:
    require_sat: bool = False
    top_k: int = 5
    evidence_weight: float = 0.7
    consensus_weight: float = 0.3
    random_seed: int = 42


@dataclass(frozen=True)
class EvidenceConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    top_k_chunks: int = 5
    label_weight: float = 0.4
    trace_weight: float = 0.4
    trace_presence_weight: float = 0.2


@dataclass(frozen=True)
class ConsensusConfig:
    feature_weight: float = 0.6
    edge_weight: float = 0.4
    constraint_weight: float = 0.0


@dataclass(frozen=True)
class AblationConfig:
    use_evidence: bool = True
    use_consensus: bool = True
    use_admissibility: bool = True
    use_sat_gate: bool = False


def _read_mapping(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if cfg_path.suffix.lower() == '.json':
        import json
        data = json.loads(cfg_path.read_text(encoding='utf-8'))
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError('PyYAML is required to load YAML config files.') from exc
        data = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {cfg_path}")
    return data


def _coerce(cls: type, raw: Dict[str, Any] | None) -> Any:
    raw = raw or {}
    allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in allowed})


def load_proxy_config(path: str | Path) -> ProxySelectorConfig:
    data = _read_mapping(path)
    return _coerce(ProxySelectorConfig, data.get('proxy_selector') or data)


def load_evidence_config(path: str | Path) -> EvidenceConfig:
    data = _read_mapping(path)
    return _coerce(EvidenceConfig, data.get('evidence') or data)


def load_consensus_config(path: str | Path) -> ConsensusConfig:
    data = _read_mapping(path)
    return _coerce(ConsensusConfig, data.get('consensus') or data)


def load_ablation_config(path: str | Path) -> AblationConfig:
    data = _read_mapping(path)
    return _coerce(AblationConfig, data.get('ablation') or data)
