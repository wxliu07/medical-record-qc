import importlib
from typing import Any, Dict, List


def _build_text(report: Dict[str, Any]) -> str:
    c = report.get("content", {})
    return "\n".join(
        [
            f"描述: {c.get('描述', '')}",
            f"检查所见: {c.get('检查所见', '')}",
            f"检查提示: {c.get('检查提示', '')}",
        ]
    )


def _fallback_imaging_parse(report: Dict[str, Any], qc_rules: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    findings = report.get("content", {}).get("检查所见", "")
    impression = report.get("content", {}).get("检查提示", "")
    text = f"{findings} {impression}"

    anatomy_vocab = ["肺", "右肺", "左肺", "脑", "额叶", "顶叶", "肝", "肝右叶", "肝左叶"]
    lesion_vocab = ["结节", "病变", "梗死", "出血", "占位", "渗出"]

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for term in anatomy_vocab:
        if term in text:
            nodes.append({"id": f"anat_{term}", "text": term, "type": "anatomy"})

    for term in lesion_vocab:
        if term in text:
            nodes.append({"id": f"les_{term}", "text": term, "type": "lesion"})

    anatomy_nodes = [n for n in nodes if n["type"] == "anatomy"]
    lesion_nodes = [n for n in nodes if n["type"] == "lesion"]
    for lesion in lesion_nodes:
        for anatomy in anatomy_nodes:
            edges.append(
                {
                    "source": lesion["id"],
                    "target": anatomy["id"],
                    "relation": "located_at",
                }
            )

    entities = {
        "患者信息": report.get("content", {}).get("描述", ""),
        "检查所见": findings,
        "检查提示": impression,
    }

    return {
        "entities": entities,
        "nodes": nodes,
        "edges": edges,
        "relations": edges,
        "source": "fallback_imaging_parser",
        "degraded": True,
        "raw_output": _build_text(report),
        "error": reason,
    }


def _extract_with_radgraph(report: Dict[str, Any]) -> Dict[str, Any]:
    # Dynamic import keeps radgraph as an optional dependency and enables fallback when unavailable.
    module = importlib.import_module("radgraph")

    text = _build_text(report)
    model = None
    if hasattr(module, "RadGraph"):
        model = module.RadGraph()
    elif hasattr(module, "RadGraphParser"):
        model = module.RadGraphParser()
    if model is None:
        raise RuntimeError("radgraph module loaded but no known parser class found")

    # The official RadGraph class is callable and returns a dict keyed by string indexes.
    if hasattr(model, "__call__"):
        parsed = model([text])
    elif hasattr(model, "predict"):
        parsed = model.predict([text])
    else:
        raise RuntimeError("no callable inference method found on radgraph model")

    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError("radgraph parser returned empty output")

    first_key = next(iter(parsed.keys()))
    item = parsed.get(first_key, {})
    entity_map = item.get("entities", {}) if isinstance(item, dict) else {}

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for entity_id, entity_data in entity_map.items():
        label = entity_data.get("label", "")
        node_type = "lesion" if str(label).startswith("Observation") else "anatomy"
        nodes.append(
            {
                "id": str(entity_id),
                "text": entity_data.get("tokens", ""),
                "type": node_type,
                "label": label,
            }
        )
        for rel in entity_data.get("relations", []):
            if isinstance(rel, list) and len(rel) == 2:
                edges.append(
                    {
                        "source": str(entity_id),
                        "target": str(rel[1]),
                        "relation": str(rel[0]),
                    }
                )

    return {
        "entities": {
            "患者信息": report.get("content", {}).get("描述", ""),
            "检查所见": report.get("content", {}).get("检查所见", ""),
            "检查提示": report.get("content", {}).get("检查提示", ""),
        },
        "nodes": nodes,
        "edges": edges,
        "relations": edges,
        "source": "radgraph",
        "degraded": False,
        "raw_output": item,
    }


def extract_imaging_ner_re(
    report: Dict[str, Any],
    qc_rules: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    rad_cfg = model_config.get("radgraph", {})
    enabled = bool(rad_cfg.get("enabled", True))
    allow_fallback = bool(rad_cfg.get("fallback_enabled", True))

    if not enabled:
        return _fallback_imaging_parse(report, qc_rules, reason="radgraph disabled by config")

    try:
        return _extract_with_radgraph(report)
    except Exception as exc:
        if allow_fallback:
            return _fallback_imaging_parse(report, qc_rules, reason=str(exc))
        raise
