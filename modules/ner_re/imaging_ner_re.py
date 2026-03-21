import importlib
import inspect
from typing import Any, Dict, List


def _build_text(report: Dict[str, Any]) -> str:
    """
    构建用于实体识别的完整文本
    
    功能：从报告字典中提取关键字段，组合成适合NLP模型处理的连续文本
    字段包括：描述（患者基本信息）、检查所见、检查提示
    
    参数:
        report: 包含报告内容的字典，格式为{"content": {"描述": "", "检查所见": "", "检查提示": ""}}
    
    返回:
        组合后的文本字符串，各字段以换行符分隔
    """
    c = report.get("content", {})
    return "\n".join(
        [
            f"描述: {c.get('描述', '')}",
            f"检查所见: {c.get('检查所见', '')}",
            f"检查提示: {c.get('检查提示', '')}",
        ]
    )


def _fallback_imaging_parse(report: Dict[str, Any], qc_rules: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    """
    备用影像学解析函数（降级处理模式）
    
    功能：当RadGraph模型不可用或解析失败时，使用基于规则的关键词匹配方法
          提取解剖部位和病变实体，并建立它们之间的关联关系
    
    参数:
        report: 原始报告数据
        qc_rules: 质量控制规则配置（当前未使用，保留接口一致性）
        reason: 降级原因描述，用于错误追踪
    
    返回:
        结构化的解析结果，包含实体、节点、边和降级标记
    """
    # 提取报告中的关键文本内容
    findings = report.get("content", {}).get("检查所见", "")
    impression = report.get("content", {}).get("检查提示", "")
    text = f"{findings} {impression}"

    # 预定义的解剖部位词汇表（可扩展）
    anatomy_vocab = ["肺", "右肺", "左肺", "脑", "额叶", "顶叶", "肝", "肝右叶", "肝左叶"]
    # 预定义的病变/异常词汇表
    lesion_vocab = ["结节", "病变", "梗死", "出血", "占位", "渗出"]

    # 初始化节点和边列表
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # 基于词汇匹配创建解剖部位节点
    for term in anatomy_vocab:
        if term in text:
            nodes.append({"id": f"anat_{term}", "text": term, "type": "anatomy"})

    # 基于词汇匹配创建病变节点
    for term in lesion_vocab:
        if term in text:
            nodes.append({"id": f"les_{term}", "text": term, "type": "lesion"})

    # 建立病变与解剖部位的关系（假设所有病变都位于所有出现的解剖部位）
    anatomy_nodes = [n for n in nodes if n["type"] == "anatomy"]
    lesion_nodes = [n for n in nodes if n["type"] == "lesion"]
    for lesion in lesion_nodes:
        for anatomy in anatomy_nodes:
            edges.append(
                {
                    "source": lesion["id"],
                    "target": anatomy["id"],
                    "relation": "located_at",  # 关系类型：位于
                }
            )

    # 提取原始文本字段作为基本实体
    entities = {
        "患者信息": report.get("content", {}).get("描述", ""),
        "检查所见": findings,
        "检查提示": impression,
    }

    # 返回标准格式的解析结果，包含降级标记
    return {
        "entities": entities,          # 基础实体字段
        "nodes": nodes,                 # 图谱节点
        "edges": edges,                  # 图谱边
        "relations": edges,              # 关系列表（兼容性字段）
        "source": "fallback_imaging_parser",  # 解析器来源标识
        "degraded": True,                 # 降级处理标记
        "raw_output": _build_text(report),  # 原始输入文本
        "error": reason,                    # 降级原因
    }


def _extract_with_radgraph(report: Dict[str, Any], model_type: str = "radgraph-xl") -> Dict[str, Any]:
    """
    使用RadGraph模型进行影像学报告解析
    
    功能：调用RadGraph深度学习模型，从影像学报告中提取实体和关系，
          构建结构化的医学知识图谱
    
    参数:
        report: 原始报告数据
    
    返回:
        结构化的解析结果，包含实体、节点、边等
    
    异常:
        RuntimeError: 当RadGraph模块加载失败或解析结果为空时抛出
    """
    # 动态导入RadGraph包，使其成为可选依赖
    module = importlib.import_module("radgraph")

    # 构建模型输入文本
    text = _build_text(report)
    
    # 初始化RadGraph模型（兼容不同版本的API）
    model = None
    parser_class = None
    if hasattr(module, "RadGraph"):
        parser_class = module.RadGraph
    elif hasattr(module, "RadGraphParser"):
        parser_class = module.RadGraphParser
    if parser_class is not None:
        try:
            sig = inspect.signature(parser_class)
            accepts_model_type = any(
                param.kind == inspect.Parameter.VAR_KEYWORD or param.name == "model_type"
                for param in sig.parameters.values()
            )
        except (TypeError, ValueError):
            # 某些可调用对象无法安全获取签名，保守降级到无参构造
            accepts_model_type = False
        if accepts_model_type:
            model = parser_class(model_type=model_type)
        else:
            model = parser_class()
    if model is None:
        raise RuntimeError("radgraph module loaded but no known parser class found")

    # 调用模型进行预测（兼容不同版本的调用方式）
    if hasattr(model, "__call__"):
        parsed = model([text])  # 新版RadGraph直接调用
    elif hasattr(model, "predict"):
        parsed = model.predict([text])  # 旧版使用predict方法
    else:
        raise RuntimeError("no callable inference method found on radgraph model")

    # 验证解析结果
    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError("radgraph parser returned empty output")

    # 处理模型输出结果
    first_key = next(iter(parsed.keys()))
    item = parsed.get(first_key, {})
    entity_map = item.get("entities", {}) if isinstance(item, dict) else {}

    # 构建图谱节点和边
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for entity_id, entity_data in entity_map.items():
        # 确定节点类型（病变或解剖部位）
        label = entity_data.get("label", "")
        node_type = "lesion" if str(label).startswith("Observation") else "anatomy"
        
        # 创建节点
        nodes.append(
            {
                "id": str(entity_id),
                "text": entity_data.get("tokens", ""),
                "type": node_type,
                "label": label,
            }
        )
        
        # 创建关系边
        for rel in entity_data.get("relations", []):
            if isinstance(rel, list) and len(rel) == 2:
                edges.append(
                    {
                        "source": str(entity_id),
                        "target": str(rel[1]),
                        "relation": str(rel[0]),
                    }
                )

    # 返回标准格式的解析结果
    return {
        "entities": {
            "患者信息": report.get("content", {}).get("描述", ""),
            "检查所见": report.get("content", {}).get("检查所见", ""),
            "检查提示": report.get("content", {}).get("检查提示", ""),
        },
        "nodes": nodes,
        "edges": edges,
        "relations": edges,
        "source": "radgraph",  # 解析器来源标识
        "degraded": False,      # 正常处理标记
        "raw_output": item,     # 模型原始输出
    }


def extract_imaging_ner_re(
    report: Dict[str, Any],
    qc_rules: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    影像学报告实体关系提取主函数
    
    功能：根据配置选择使用RadGraph模型或备用规则方法进行实体关系提取，
          实现降级处理和异常捕获的完整流程
    
    参数:
        report: 原始报告数据
        qc_rules: 质量控制规则配置
        model_config: 模型配置，包含radgraph子配置项
    
    返回:
        结构化的解析结果
    
    异常:
        当RadGraph启用且禁用fallback时，向上传递原始异常
    """
    # 获取RadGraph配置
    rad_cfg = model_config.get("radgraph", {})
    enabled = bool(rad_cfg.get("enabled", True))  # 是否启用RadGraph
    allow_fallback = bool(rad_cfg.get("fallback_enabled", True))  # 是否允许降级

    # 场景1：RadGraph被配置禁用，直接使用降级解析
    if not enabled:
        return _fallback_imaging_parse(report, qc_rules, reason="radgraph disabled by config")

    # 场景2：尝试使用RadGraph解析
    try:
        model_type = str(rad_cfg.get("model_type", "radgraph-xl")).strip() or "radgraph-xl"
        return _extract_with_radgraph(report, model_type=model_type)
    except Exception as exc:
        # 如果允许降级，则使用备用解析器并记录异常原因
        if allow_fallback:
            return _fallback_imaging_parse(report, qc_rules, reason=str(exc))
        # 如果不允许降级，向上抛出异常
        raise