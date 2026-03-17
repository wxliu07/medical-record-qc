import json
import os
from datetime import datetime
from typing import Dict, List

from llm_ie.engines import LiteLLMInferenceEngine
from llm_ie.extractors import DirectFrameExtractor
from llm_ie.chunkers import SentenceUnitChunker


# ==============================
# 1 DeepSeek API配置
# ==============================

API_KEY = "sk-72b2cd26fb33451992ebc1e762fab86c"
MODEL = "deepseek/deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"

engine = LiteLLMInferenceEngine(
    model=MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ==============================
# 2 质控规则
# ==============================

QC_RULES = {
    "指标类": {
        "白细胞计数": {
            "range": (3.5, 9.5),
            "unit": "10^9/L"
        },
        "红细胞计数": {
            "range": (4.3, 5.8),
            "unit": "10^12/L"
        }
    },
    "影像类": {
        "missing": "检查所见不能为空",
        "logic_conflict": "检查所见正常但提示异常"
    }
}


# ==============================
# 3 LLM Prompt
# ==============================

QC_PROMPT = """
你是医疗报告质控AI。

任务：

1 抽取医疗实体

实体类型：

疾病
指标
检查所见
检查提示
患者信息

2 抽取关系

例如：

疾病-指标
疾病-症状
指标-异常

3 检测问题

缺失项
逻辑矛盾
异常指标

输出JSON：

{
"entities":{},
"relations":[],
"issues":[]
}

报告：

{{text}}
"""


# ==============================
# 4 规则质控
# ==============================

def apply_rules(entities: Dict, relations: List, report_type: str):
    errors = []

    # 指标检查
    if report_type == "指标类":
        indicators = entities.get("指标", {})

        # 处理两种可能的格式
        if isinstance(indicators, list):
            # 如果是列表格式，尝试从relations中获取具体值
            print("指标是列表格式，尝试从relations补充信息")

            # 从relations中提取指标-值对
            indicator_values = {}
            for rel in relations:
                if isinstance(rel, dict) and rel.get('relation') == '指标-异常':
                    # 假设relation格式如：{"head": "白细胞计数", "tail": "8.0", "relation": "指标-异常"}
                    indicator_values[rel.get('head', '')] = rel.get('tail', '')

            # 使用从relations提取的值进行检查
            for name in indicators:
                value = indicator_values.get(name, '')
                if name not in QC_RULES["指标类"]:
                    continue

                try:
                    # 尝试提取数值
                    number_str = str(value).split()[0] if value else ''
                    number = float(number_str)
                except (ValueError, IndexError):
                    print(f"无法解析指标值: {name}={value}")
                    continue

                low, high = QC_RULES["指标类"][name]["range"]
                if number < low or number > high:
                    errors.append(
                        f"{name}异常 {number} (正常范围 {low}-{high})"
                    )

        elif isinstance(indicators, dict):
            # 原有的字典格式处理
            for name, value in indicators.items():
                if name not in QC_RULES["指标类"]:
                    continue
                try:
                    number = float(str(value).split()[0])
                except:
                    continue
                low, high = QC_RULES["指标类"][name]["range"]
                if number < low or number > high:
                    errors.append(
                        f"{name}异常 {number} (正常范围 {low}-{high})"
                    )

    # 影像报告检查
    if report_type == "影像类":
        # 检查所见可能是列表，需要处理
        findings = entities.get("检查所见", [])
        if isinstance(findings, list):
            if not findings:
                errors.append("缺失检查所见")
        elif not findings:
            errors.append("缺失检查所见")

        # 检查逻辑矛盾
        findings_normal = any('正常' in str(f) for f in (findings if isinstance(findings, list) else [findings]))
        impressions = entities.get("检查提示", [])
        impressions_abnormal = any(
            '异常' in str(i) for i in (impressions if isinstance(impressions, list) else [impressions]))

        if findings_normal and impressions_abnormal:
            errors.append("检查所见与检查提示矛盾")

    # 关系矛盾检测
    for r in relations:
        if isinstance(r, dict) and r.get("relation") == "矛盾":
            errors.append("实体关系矛盾")
        elif isinstance(r, str) and r == "矛盾":
            errors.append("实体关系矛盾")

    return errors


# ==============================
# 5 LLM信息抽取
# ==============================

def extract_medical_info(text):
    """从医疗文本中提取结构化信息"""
    extractor = DirectFrameExtractor(
        inference_engine=engine,
        unit_chunker=SentenceUnitChunker(),
        prompt_template=QC_PROMPT
    )

    try:
        frames = extractor.extract(text)
    except Exception as e:
        print(f"提取过程异常: {e}")
        return None

    print(f"提取到的frames数量: {len(frames) if frames else 0}")

    if not frames:
        print("警告: 没有提取到任何frames")
        return None

    frame = frames[0]

    # 获取LLM生成的内容
    content = None

    # 方法1: 直接获取gen_text
    if hasattr(frame, 'gen_text') and frame.gen_text:
        content = frame.gen_text
        print("从gen_text获取内容")

    # 方法2: 通过方法获取
    elif hasattr(frame, 'get_generated_text'):
        content = frame.get_generated_text()
        print("从get_generated_text()获取内容")

    # 方法3: 检查其他可能属性
    elif hasattr(frame, 'generated_text'):
        content = frame.generated_text
        print("从generated_text获取内容")

    elif hasattr(frame, 'output'):
        content = frame.output
        print("从output获取内容")

    else:
        # 调试：打印所有属性
        print("可用的frame属性:")
        for attr in dir(frame):
            if not attr.startswith('_'):
                try:
                    val = getattr(frame, attr)
                    if val and not callable(val):
                        print(f"  - {attr}: {type(val)}")
                except:
                    pass

        # 最后尝试str转换
        content = str(frame)
        print("使用str(frame)获取内容")

    # 处理内容
    if content is None:
        print("错误: 无法获取生成内容")
        return None

    if isinstance(content, str):
        content = content.strip()
        print(f"内容预览 (前200字符): {content[:200]}...")

        # 清理可能的markdown代码块
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]

        if content.endswith('```'):
            content = content[:-3]

        content = content.strip()

        # 尝试解析JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"位置: 行 {e.lineno}, 列 {e.colno}")
            print(f"问题字符附近: {e.doc[max(0, e.pos - 20):e.pos + 20]}")
            return None
    else:
        # 如果content不是字符串，直接返回
        return content


# ==============================
# 6 LLM逻辑质控
# ==============================

def llm_reasoning_qc(entities, relations, issues):

    prompt = f"""
以下是医疗报告信息

entities:
{entities}

relations:
{relations}

issues:
{issues}

判断报告是否正确

输出JSON：

{{
"result":"correct 或 error",
"reason":"原因"
}}
"""

    extractor = DirectFrameExtractor(
        inference_engine=engine,
        unit_chunker=SentenceUnitChunker(),
        prompt_template=prompt
    )
    frames = extractor.extract("")
    if not frames:
        return {"result": "unknown"}
    frame = frames[0]
    # 检查frame的属性结构
    if hasattr(frame, 'frame') and frame.frame:
        result = frame.frame
    elif hasattr(frame, 'data'):
        result = frame.data
    else:
        # 尝试获取content
        if hasattr(frame, 'content'):
            content = frame.content
        else:
            # 作为字典处理
            content = str(frame)

        try:
            result = json.loads(content)
        except:
            return {"result": "unknown"}
    return result


# ==============================
# 7 病历摘要生成
# ==============================

def generate_case_summary(entities):
    disease = entities.get("疾病", "未知疾病")
    indicators = entities.get("指标", {})
    summary = f"患者诊断为{disease}。"
    if indicators:
        summary += "主要指标："
        for k, v in indicators.items():
            summary += f"{k}{v} "
    return summary


# ==============================
# 8 主流程
# ==============================

def quality_control_report(text, report_type="指标类"):
    extraction = extract_medical_info(text)
    if extraction is None:
        return {"error": "信息抽取失败"}
    entities = extraction.get("entities", {})
    relations = extraction.get("relations", [])
    issues = extraction.get("issues", [])
    # 规则质控
    rule_errors = apply_rules(entities, relations, report_type)
    issues.extend(rule_errors)

    # LLM推理质控
    reasoning = llm_reasoning_qc(entities, relations, issues)
    correct = len(issues) == 0

    if correct:
        summary = generate_case_summary(entities)
    else:
        summary = ""

    return {
        "entities": entities,
        "relations": relations,
        "issues": issues,
        "reasoning": reasoning,
        "summary": summary
    }


def save_results_multiple_formats(result: dict, base_filename: str = None, output_dir: str = "qc_results"):
    """
    将结果保存为多种格式（JSON, TXT, MD）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if base_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qc_report_{timestamp}"

    saved_files = []

    # 1. 保存为JSON（完整数据）
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    saved_files.append(json_path)
    print(f"✅ JSON格式保存: {json_path}")

    # 2. 保存为TXT（可读格式）
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("医疗报告质控结果\n")
        f.write("=" * 50 + "\n\n")

        # 实体信息
        f.write("【抽取的实体】\n")
        for entity_type, entities in result.get("entities", {}).items():
            f.write(f"  {entity_type}: {entities}\n")

        # 关系信息
        f.write("\n【抽取的关系】\n")
        for rel in result.get("relations", []):
            f.write(f"  {rel}\n")

        # 问题列表
        f.write("\n【发现的问题】\n")
        for issue in result.get("issues", []):
            if isinstance(issue, dict):
                f.write(f"  • [{issue.get('type', '未知')}] {issue.get('description', '')}\n")
            else:
                f.write(f"  • {issue}\n")

        # 推理结果
        f.write("\n【推理结果】\n")
        reasoning = result.get("reasoning", {})
        if isinstance(reasoning, dict):
            f.write(f"  结果: {reasoning.get('result', 'unknown')}\n")
            f.write(f"  原因: {reasoning.get('reason', '无')}\n")

        # 摘要
        f.write("\n【病历摘要】\n")
        f.write(f"  {result.get('summary', '无摘要')}\n")

    saved_files.append(txt_path)
    print(f"✅ TXT格式保存: {txt_path}")

    # 3. 保存为Markdown（更美观）
    md_path = os.path.join(output_dir, f"{base_filename}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 医疗报告质控结果\n\n")

        f.write("## 📋 抽取的实体\n\n")
        for entity_type, entities in result.get("entities", {}).items():
            f.write(f"### {entity_type}\n")
            f.write(f"```\n{entities}\n```\n\n")

        f.write("## 🔗 抽取的关系\n\n")
        for rel in result.get("relations", []):
            f.write(f"- `{rel}`\n")
        f.write("\n")

        f.write("## ⚠️ 发现的问题\n\n")
        for issue in result.get("issues", []):
            if isinstance(issue, dict):
                f.write(f"- **[{issue.get('type', '未知')}]** {issue.get('description', '')}\n")
            else:
                f.write(f"- {issue}\n")
        f.write("\n")

        f.write("## 🤔 推理结果\n\n")
        reasoning = result.get("reasoning", {})
        if isinstance(reasoning, dict):
            f.write(f"- **结果**: {reasoning.get('result', 'unknown')}\n")
            f.write(f"- **原因**: {reasoning.get('reason', '无')}\n")

        f.write("\n## 📝 病历摘要\n\n")
        f.write(f"{result.get('summary', '无摘要')}\n")

    saved_files.append(md_path)
    print(f"✅ Markdown格式保存: {md_path}")

    return saved_files


# ==============================
# 9 示例运行
# ==============================

if __name__ == "__main__":

    report = """

检查报告描述：
患者男，50岁，高血压。

检查所见：
血压160/100 mmHg，
白细胞计数8.0 10^9/L。

检查提示：
建议控制血压。

"""

    # result = quality_control_report(report)


    # 执行质控
    print("开始医疗报告质控...")
    result = quality_control_report(report)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 打印结果
    print("\n" + "=" * 60)
    print("质控结果:")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 保存结果到本地
    print("\n" + "=" * 60)
    print("保存结果到本地:")
    print("=" * 60)


    # 方式2：保存多种格式
    saved_files = save_results_multiple_formats(result, base_filename="医疗质控报告")

    print("\n✅ 所有文件保存完成！")