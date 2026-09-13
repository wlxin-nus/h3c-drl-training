"""Canonical paired source for the three production system prompts."""

from __future__ import annotations

import copy
import re
from typing import Any, Literal

from h3c.agents.contracts import (
    allocation_constraint_text,
    allocation_contract,
    compact_patch_contract,
)

Role = Literal["orchestrator", "executor", "reflector"]
Language = Literal["en", "zh"]


def _unit(
    section_id: str,
    title_en: str,
    body_en: str,
    title_zh: str,
    body_zh: str,
    kind: str = "constraint",
) -> dict[str, str]:
    return {
        "id": section_id,
        "title_en": title_en,
        "body_en": body_en,
        "title_zh": title_zh,
        "body_zh": body_zh,
        "kind": kind,
    }


def _causal_unit(
    section_id: str,
    title_en: str,
    body_en: str,
    title_zh: str,
    body_zh: str,
) -> dict[str, str]:
    unit = _unit(
        section_id,
        title_en,
        body_en,
        title_zh,
        body_zh,
        kind="information",
    )
    unit["conditional"] = "causal"
    return unit


def _objective_unit() -> dict[str, str]:
    return _unit(
        "objective_reference",
        "OBJECTIVE",
        "Maintain comfort while reducing energy cost as much as possible. The reward from a "
        "completed interval is a quantitative reference for the same frozen trade-off among "
        "energy cost, comfort, and action smoothness; a higher cumulative reward indicates a "
        "better overall result.",
        "目标",
        "在维持舒适的同时尽可能降低能耗花费。已完成时段的 reward 是冻结评价函数对能耗花费、"
        "舒适度和动作平滑性的量化参考；累计 reward 越高，表示上述同一目标的综合结果越好。",
        kind="information",
    )


PAIRED_PROMPT_UNITS: dict[Role, list[dict[str, str]]] = {
    "orchestrator": [
        _unit(
            "role",
            "ROLE",
            "Allocate cross-zone allowance over action_times in control_interval.",
            "角色",
            "在 control_interval 的 action_times 上分配跨区域共享额度。",
        ),
        _objective_unit(),
        _unit(
            "decision",
            "DECISION",
            "Balance comfort and energy cost from the supplied state, forecast, working memory and previous Budget use. Output allocation, priority and per-zone rationales.",
            "决策",
            "使用当前状态、预测、工作记忆与上一时段 Budget，在舒适和能源成本间权衡。输出额度、优先级及每区一条额度或优先级理由。",
        ),
        _unit(
            "hard_boundaries",
            "HARD BOUNDARIES",
            "For accepted program changes, charge_c = max over proof states of max(0, setpoint_before_c - setpoint_after_c); not cumulative action, power or pre-cooling offset.",
            "硬边界",
            "Budget 只覆盖 control_interval 内获准的程序修改。charge_c = 证明状态空间中 max(0, 修改前设定点 - 修改后设定点) 的最大值；不是累计动作、功率或预冷偏移。",
        ),
        _unit(
            "output",
            "OUTPUT",
            "Bare JSON only.",
            "输出",
            "仅返回裸 JSON。",
        ),
    ],
    "executor": [
        _unit(
            "role",
            "ROLE",
            "Maintain one zone's control specification.",
            "角色",
            "维护一个区域的可执行控制规格。",
        ),
        _objective_unit(),
        _unit(
            "decision",
            "DECISION",
            "From current state, comfort headroom, causal evidence, allowance and WORKING MEMORY, choose one atomic specification operation balancing energy and |PMV| ≤ 0.5. Before an edit, take the target rule's matching-state base from its effect projection and apply both the current and proposed action formulas. If a formula uses the last setpoint, use the explicitly shown current last physical setpoint.",
            "决策",
            "根据当前状态、舒适余量、因果证据、额度与 WORKING MEMORY，选择一个原子规格操作，权衡能耗与 |PMV| ≤ 0.5。修改前，从目标规则的命中状态作用投影取得状态基准，并分别套用当前与拟议动作公式；公式使用上一设定点时，采用明确显示的当前上一物理设定点。",
        ),
        _unit(
            "hard_boundaries",
            "HARD BOUNDARIES",
            "charge_c=max over proof states of max(0,setpoint_before_c-setpoint_after_c), excluding cumulative actions, power and pre-cooling offset. Reserved allowance is zone-only; the site-wide shared pool settles after all proposals by ascending shared_settlement_priority_rank. Edit only the supplied specification; no direct setpoint.",
            "硬边界",
            "charge_c=证明状态中 max(0,修改前设定点-修改后设定点) 的最大值，不含累计动作、功率或预冷偏移。保留额度仅属本区；站点共享池在全部提案后按 shared_settlement_priority_rank 升序结算。只修改给定规格，不输出直接设定点。",
        ),
        _unit(
            "output",
            "OUTPUT",
            "Bare JSON only; no prose, fence or extra fields.\n",
            "输出",
            '只返回一个精确的 root JSON 对象：{"patch":[{...}]}。不要返回解释文字、Markdown 围栏或未知字段。操作契约：\n',
        ),
    ],
    "reflector": [
        _unit(
            "role",
            "ROLE",
            "Derive zone Lessons from the completed control interval.",
            "角色",
            "从已完成控制时段提炼每区 Lesson。",
        ),
        _objective_unit(),
        _unit(
            "evidence",
            "EVIDENCE",
            "For each zone, derive one useful observed relationship or trade-off from completed Context, Action and Outcome. Ground it in the observed reward components: site energy is shared, while comfort and smoothness contributions are zone-specific.",
            "证据",
            "从每区已完成的 Context、Action 与 Outcome 中提炼一条有用的已观察关系或权衡，并以已观测的 reward 分量为依据：站点能耗是共享量，舒适与平滑贡献属于各区域。",
            "information",
        ),
        _unit(
            "hard_boundaries",
            "HARD BOUNDARIES",
            "The completed interval is the sole Lesson evidence. Write observations or trade-offs, not commands, targets or unsupported claims.",
            "硬边界",
            "Lesson 事实只来自已完成时段；表达观察或权衡，不写命令、目标或无证据结论。",
        ),
        _unit(
            "output",
            "OUTPUT",
            'Bare JSON only: {"hourly_lessons":[{"zone":...,"lesson":...}]}. Each zone once; no extra fields.',
            "输出",
            '只返回裸 JSON：{"hourly_lessons":[{"zone":...,"lesson":...}]}。每区一次，无额外字段。',
        ),
    ],
}


def _machine_contract(role: Role, causal_enabled: bool, language: Language) -> str:
    if role == "orchestrator":
        names = allocation_contract(causal_enabled=causal_enabled)
        prefix = " Keys exactly: " if language == "en" else " 字段仅限："
        return (
            prefix
            + ", ".join(names)
            + ". "
            + allocation_constraint_text(causal_enabled=causal_enabled, language=language)
        )
    if role == "executor":
        return compact_patch_contract(causal_enabled=causal_enabled, language=language)
    return ""


_CAUSAL_TOKENS = (
    "causal",
    "graph",
    "causal_edge_ids",
    "expected_effects",
    "derived_from_edge",
)


def assert_causal_disabled_text_clean(text: str) -> None:
    lowered = text.lower()
    leaked = [token for token in _CAUSAL_TOKENS if token in lowered]
    if re.search(r"\bce_[0-9a-f]{8}\b", text, re.IGNORECASE):
        leaked.append("edge_identifier")
    if leaked:
        raise ValueError(f"causal module text leaked in disabled prompt: {leaked}")


def system_prompt(
    role: Role,
    *,
    causal_enabled: bool = True,
    language: Language = "en",
    coordination_enabled: bool = True,
    long_term_memory: bool = False,
) -> str:
    rows: list[str] = []
    for unit in PAIRED_PROMPT_UNITS[role]:
        if unit.get("conditional") == "causal" and not causal_enabled:
            continue
        title = unit["title_en"] if language == "en" else unit["title_zh"]
        body = unit["body_en"] if language == "en" else unit["body_zh"]
        if role == "executor" and unit["id"] == "decision" and not causal_enabled:
            body = body.replace("causal evidence, ", "")
            body = body.replace("因果证据、", "")
        if role == "reflector" and unit["id"] == "hard_boundaries" and not causal_enabled:
            body = body.replace(" or unsupported causal claim", " or unsupported claim")
            body = body.replace("或无证据因果结论", "或无证据结论")
        if role == "executor" and not coordination_enabled:
            if unit["id"] == "decision":
                body = body.replace("allowance and ", "").replace("额度与 ", "")
            elif unit["id"] == "hard_boundaries":
                if language == "en":
                    body = (
                        "Operate only through the supplied control specification and operation "
                        "contract; do not write Python, a complete controller or a direct setpoint."
                    )
                else:
                    body = "只能通过给定控制规格与操作契约工作；不得编写 Python、完整控制器或直接设定点。"
        if unit["id"] == "output" and role in ("executor", "orchestrator"):
            body += _machine_contract(role, causal_enabled, language)
        if role == "executor" and unit["id"] == "output" and long_term_memory:
            body = (
                'JSON only: {"patch":[{...}],"memory_refs":'
                '[{"regime":...,"revision":...}]}. memory_refs lists each shown active '
                "experience used at most once; [] allowed. No extra fields.\n"
                if language == "en"
                else '只返回 JSON：{"patch":[{...}],"memory_refs":'
                '[{"regime":...,"revision":...}]}。memory_refs 只列出实际使用的已展示有效经验，'
                "每条最多一次；允许 []，无额外字段。\n"
            )
            body += _machine_contract(role, causal_enabled, language)
        if role == "reflector" and unit["id"] == "evidence" and long_term_memory:
            body = (
                "For each zone, derive one observed relationship or trade-off from completed "
                "Context, Action and Outcome; compare shown eligible slots and select at most "
                "one operation."
                if language == "en"
                else "每区从已完成的 Context、Action 与 Outcome 提炼一条已观察关系"
                "或权衡；比较展示的合格状态槽，最多选一个操作。"
            )
        if role == "reflector" and unit["id"] == "hard_boundaries" and long_term_memory:
            body = (
                "The completed interval is the sole Lesson evidence; slots are comparison-only. "
                "Experiences summarize run-wide thermal response, control preference or trade-off "
                "by regime. No commands, targets or unsupported claims. Modify a shown regime "
                "only; replace/delete uses its revision."
                if language == "en"
                else "已完成时段是 Lesson 唯一事实来源，经验槽只用于比较。经验按状态"
                "概括全程热响应、控制偏好或权衡，不写命令、目标或无证据结论。只修改"
                "展示状态；replace/delete 使用其 revision。"
            )
        if role == "reflector" and unit["id"] == "output" and long_term_memory:
            body = (
                'Bare JSON: {"hourly_lessons":[{"zone":...,"lesson":...}],'
                '"memory_operations":[...]}. Each zone once per list. Operations require zone,op; '
                "extra fields:\n"
                "| op | required |\n| --- | --- |\n| no_change | — |\n"
                "| add | regime, experience |\n| replace | regime, expected_revision, experience |\n"
                "| delete | regime, expected_revision |\nNo prose, fence or extra fields."
                if language == "en"
                else '返回裸 JSON：{"hourly_lessons":[{"zone":...,"lesson":...}],'
                '"memory_operations":[...]}；每个给定区域在两个列表中各出现一次。每个记忆操作'
                "都需要 zone 与 op，额外必填字段：\n| op | 必填 |\n| --- | --- |\n"
                "| no_change | — |\n| add | regime, experience |\n"
                "| replace | regime, expected_revision, experience |\n"
                "| delete | regime, expected_revision |\n不返回解释文字、Markdown 围栏或额外字段。"
            )
        rows.append(title + "\n" + body)
    rendered = "\n\n".join(rows) + "\n"
    if not causal_enabled:
        assert_causal_disabled_text_clean(rendered)
    return rendered


def prompt_units(role: Role | None = None) -> Any:
    return copy.deepcopy(PAIRED_PROMPT_UNITS if role is None else PAIRED_PROMPT_UNITS[role])


ORCHESTRATOR_SYSTEM_PROMPT = system_prompt("orchestrator")
EXECUTOR_SYSTEM_PROMPT = system_prompt("executor")
REFLECTOR_SYSTEM_PROMPT = system_prompt("reflector")
