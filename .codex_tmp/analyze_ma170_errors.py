from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\Code\Python\GeneralParameter\J35_001_none\PR\XGD")
OUTPUT_DIR = Path(r"D:\Code\Python\surrogate_model\analysis")
TARGET_MACH = 1.7

STAGES = [
    ("S0", "基础模型", "XGD_user_for_win7_fixed_exc_channel_fdr_overfit"),
    ("S1", "+H0300_M098", "XGD_user_for_win7_fixed_exc_channel_fdr_overfit_plus_H0300_M098"),
    ("S2", "+H0550_M105", "XGD_user_for_win7_fixed_exc_channel_fdr_overfit_plus_H0550_M105"),
    ("S3", "+H0550_M117", "XGD_user_for_win7_fixed_exc_channel_fdr_overfit_plus_H0550_M117"),
    ("S4", "+H0823_M170", "XGD_user_for_win7_fixed_exc_channel_fdr_overfit_plus_H0823_M170"),
]
KINDS = {"VA": "A", "VF": "F", "VG": "G"}
EXCITATIONS = ["FXD_DC", "FXD_FDC"]
KEYS = ["metric", "excitation", "channel", "slot"]


def fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def pct_change(final: float, initial: float) -> float:
    if pd.isna(final) or pd.isna(initial) or initial == 0:
        return np.nan
    return (final - initial) / initial * 100.0


def read_all() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for stage_index, (stage, stage_label, folder) in enumerate(STAGES):
        error_dir = ROOT / folder / "ERROR_PR_XGD"
        for file_prefix, metric in KINDS.items():
            for excitation in EXCITATIONS:
                path = error_dir / f"ERROR_{file_prefix}_H0823_{excitation}.dat"
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                frame["M"] = pd.to_numeric(frame["M"], errors="coerce")
                selected = frame[np.isclose(frame["M"], TARGET_MACH, atol=1e-12, rtol=0.0)].copy()
                if selected.empty:
                    raise ValueError(f"{path} 中没有 M={TARGET_MACH}")
                if selected["ch"].duplicated().any():
                    raise ValueError(f"{path} 的 M={TARGET_MACH} 存在重复通道")
                for _, row in selected.iterrows():
                    for slot in (1, 2, 3):
                        value = pd.to_numeric(row[f"err{slot}"], errors="coerce")
                        records.append(
                            {
                                "stage_index": stage_index,
                                "stage": stage,
                                "stage_label": stage_label,
                                "model_folder": folder,
                                "metric": metric,
                                "excitation": excitation,
                                "channel": str(row["ch"]).strip(),
                                "slot": slot,
                                "error_pct": float(value) if pd.notna(value) else np.nan,
                                "source_file": str(path),
                            }
                        )
    return pd.DataFrame(records)


def aggregate(frame: pd.DataFrame, groups: list[str], scope: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group in frame.groupby(groups, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        values = group["error_pct"].dropna()
        row = {key: value for key, value in zip(groups, group_values)}
        row.update(
            {
                "scope": scope,
                "valid_n": int(values.size),
                "missing_n": int(group["error_pct"].isna().sum()),
                "mean": values.mean() if not values.empty else np.nan,
                "median": values.median() if not values.empty else np.nan,
                "p90": values.quantile(0.9) if not values.empty else np.nan,
                "max": values.max() if not values.empty else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_complete(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = long.pivot(index=KEYS, columns="stage", values="error_pct").reset_index()
    stage_cols = [stage for stage, _, _ in STAGES]
    complete = wide.dropna(subset=stage_cols).copy()
    for previous, current in zip(stage_cols, stage_cols[1:]):
        complete[f"delta_{previous}_{current}"] = complete[current] - complete[previous]
    complete["delta_S0_S4"] = complete["S4"] - complete["S0"]
    complete["relative_change_S0_S4_pct"] = np.where(
        complete["S0"] != 0,
        complete["delta_S0_S4"] / complete["S0"] * 100.0,
        np.nan,
    )
    return wide, complete


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 2) -> str:
    headers = [title for _, title in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, (float, np.floating)):
                values.append(fmt(float(value), digits))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(long: pd.DataFrame, wide: pd.DataFrame, complete: pd.DataFrame) -> str:
    stage_order = [stage for stage, _, _ in STAGES]
    label_map = {stage: label for stage, label, _ in STAGES}

    overall = aggregate(long, ["stage_index", "stage", "stage_label"], "overall").sort_values("stage_index")
    by_metric = aggregate(long, ["stage_index", "stage", "stage_label", "metric"], "metric").sort_values(["stage_index", "metric"])
    by_exc = aggregate(long, ["stage_index", "stage", "stage_label", "excitation"], "excitation").sort_values(["stage_index", "excitation"])
    by_slot = aggregate(long, ["stage_index", "stage", "stage_label", "slot"], "slot").sort_values(["stage_index", "slot"])

    pair_rows = []
    for previous, current in zip(stage_order, stage_order[1:]):
        delta = complete[current] - complete[previous]
        pair_rows.append(
            {
                "transition": f"{previous}→{current}",
                "mean_delta": delta.mean(),
                "median_delta": delta.median(),
                "improved": int((delta < 0).sum()),
                "worsened": int((delta > 0).sum()),
                "unchanged": int(np.isclose(delta, 0).sum()),
                "improved_pct": (delta < 0).mean() * 100.0,
            }
        )
    transitions = pd.DataFrame(pair_rows)

    final_delta = complete.copy()
    channel_change = (
        final_delta.groupby("channel", sort=False)
        .agg(S0=("S0", "mean"), S4=("S4", "mean"), delta=("delta_S0_S4", "mean"), comparable_n=("S0", "size"))
        .reset_index()
        .sort_values("delta")
    )
    metric_change = (
        final_delta.groupby("metric", sort=False)
        .agg(S0=("S0", "mean"), S4=("S4", "mean"), delta=("delta_S0_S4", "mean"), comparable_n=("S0", "size"))
        .reset_index()
    )
    metric_change["relative_change_pct"] = [pct_change(row.S4, row.S0) for row in metric_change.itertuples()]
    exc_change = (
        final_delta.groupby("excitation", sort=False)
        .agg(S0=("S0", "mean"), S4=("S4", "mean"), delta=("delta_S0_S4", "mean"), comparable_n=("S0", "size"))
        .reset_index()
    )
    exc_change["relative_change_pct"] = [pct_change(row.S4, row.S0) for row in exc_change.itertuples()]
    slot_change = (
        final_delta.groupby("slot", sort=False)
        .agg(S0=("S0", "mean"), S4=("S4", "mean"), delta=("delta_S0_S4", "mean"), comparable_n=("S0", "size"))
        .reset_index()
    )
    slot_change["relative_change_pct"] = [pct_change(row.S4, row.S0) for row in slot_change.itertuples()]

    trajectory_columns = KEYS + stage_order + ["delta_S0_S4"]
    best = final_delta.nsmallest(10, "delta_S0_S4")[trajectory_columns]
    worst = final_delta.nlargest(10, "delta_S0_S4")[trajectory_columns]

    s0 = overall.iloc[0]
    s4 = overall.iloc[-1]
    overall_relative = pct_change(float(s4["mean"]), float(s0["mean"]))
    final_delta_values = complete["delta_S0_S4"]
    final_improved_share = (final_delta_values < 0).mean() * 100.0
    best_stage = overall.loc[overall["mean"].idxmin()]

    conclusions = []
    if s4["mean"] < s0["mean"]:
        conclusions.append(f"最终阶段的总体平均误差比基础模型下降 {abs(overall_relative):.2f}%")
    else:
        conclusions.append(f"最终阶段的总体平均误差比基础模型上升 {overall_relative:.2f}%")
    conclusions.append(f"五阶段中总体平均误差最低的是 {best_stage['stage']}（{best_stage['stage_label']}）")
    conclusions.append(f"完整可比项中有 {final_improved_share:.1f}% 在最终阶段优于基础模型")

    lines = [
        "# 1.7 Ma 下 A/F/G 误差演化报告",
        "",
        "## 1. 分析范围与口径",
        "",
        "- 阶段顺序：S0 基础模型 → S1 +H0300_M098 → S2 +H0550_M105 → S3 +H0550_M117 → S4 +H0823_M170。",
        "- 工况：仅使用 M=1.7 的误差行。",
        "- 指标：A=峰值误差（VA），F=频率误差（VF），G=阻尼误差（VG）。",
        "- 范围：FXD_DC 与 FXD_FDC、全部通道、槽位 1/2/3。误差单位均为百分比。",
        "- 空槽位按缺失处理，不以 0 参与统计。阶段间变化仅使用五阶段均非空的完整可比项。",
        "",
        "## 2. 核心结论",
        "",
        *[f"- {item}。" for item in conclusions],
        f"- 演化并非单调：总体均值依次为 " + " → ".join(fmt(value) for value in overall["mean"]) + "。S1略恶化、S2改善、S3再次恶化，S4改善最明显。",
        f"- S4 的均值、中位数和 P90 均优于 S0，但最大误差由 {s0['max']:.2f}% 上升到 {s4['max']:.2f}%，说明仍存在严重局部退化。",
        "- 均值会受到少数大相对误差影响，因此报告同时给出中位数和 P90；判断模型稳定性时不应只看均值。",
        "",
        "## 3. 总体误差演化",
        "",
        markdown_table(overall, [("stage", "阶段"), ("stage_label", "模型增量"), ("valid_n", "有效项"), ("missing_n", "缺失项"), ("mean", "均值"), ("median", "中位数"), ("p90", "P90"), ("max", "最大值")]),
        "",
        "### 逐阶段变化（完整可比项）",
        "",
        markdown_table(transitions, [("transition", "阶段变化"), ("mean_delta", "平均增减/百分点"), ("median_delta", "中位增减/百分点"), ("improved", "改善项数"), ("worsened", "恶化项数"), ("unchanged", "不变项数"), ("improved_pct", "改善占比/%")]),
        "",
        "## 4. A/F/G 分项变化",
        "",
        markdown_table(by_metric, [("stage", "阶段"), ("metric", "指标"), ("valid_n", "有效项"), ("mean", "均值"), ("median", "中位数"), ("p90", "P90"), ("max", "最大值")]),
        "",
        "### 最终阶段相对基础模型（完整可比项）",
        "",
        markdown_table(metric_change, [("metric", "指标"), ("comparable_n", "可比项"), ("S0", "S0均值"), ("S4", "S4均值"), ("delta", "增减/百分点"), ("relative_change_pct", "相对变化/%")]),
        "",
        "## 5. 分激励方式变化",
        "",
        markdown_table(by_exc, [("stage", "阶段"), ("excitation", "激励"), ("valid_n", "有效项"), ("mean", "均值"), ("median", "中位数"), ("p90", "P90"), ("max", "最大值")]),
        "",
        "### 最终阶段相对基础模型",
        "",
        markdown_table(exc_change, [("excitation", "激励"), ("comparable_n", "可比项"), ("S0", "S0均值"), ("S4", "S4均值"), ("delta", "增减/百分点"), ("relative_change_pct", "相对变化/%")]),
        "",
        "## 6. 分槽位变化",
        "",
        markdown_table(by_slot, [("stage", "阶段"), ("slot", "槽位"), ("valid_n", "有效项"), ("mean", "均值"), ("median", "中位数"), ("p90", "P90"), ("max", "最大值")]),
        "",
        "### 最终阶段相对基础模型",
        "",
        markdown_table(slot_change, [("slot", "槽位"), ("comparable_n", "可比项"), ("S0", "S0均值"), ("S4", "S4均值"), ("delta", "增减/百分点"), ("relative_change_pct", "相对变化/%")]),
        "",
        "## 7. 分通道最终变化",
        "",
        markdown_table(channel_change, [("channel", "通道"), ("comparable_n", "可比项"), ("S0", "S0均值"), ("S4", "S4均值"), ("delta", "增减/百分点")]),
        "",
        "负的增减表示误差下降，正值表示误差上升。通道均值综合了 A/F/G、两种激励方式及该通道存在的槽位。",
        "",
        "## 8. 改善最大与恶化最大项目",
        "",
        "### 改善最大的 10 项",
        "",
        markdown_table(best, [("metric", "指标"), ("excitation", "激励"), ("channel", "通道"), ("slot", "槽位"), ("S0", "S0"), ("S1", "S1"), ("S2", "S2"), ("S3", "S3"), ("S4", "S4"), ("delta_S0_S4", "首尾增减/百分点")]),
        "",
        "### 恶化最大的 10 项",
        "",
        markdown_table(worst, [("metric", "指标"), ("excitation", "激励"), ("channel", "通道"), ("slot", "槽位"), ("S0", "S0"), ("S1", "S1"), ("S2", "S2"), ("S3", "S3"), ("S4", "S4"), ("delta_S0_S4", "首尾增减/百分点")]),
        "",
        "## 9. 解释与建议",
        "",
        "- 本报告分析的是相对误差。参考值很小时，即使绝对偏差不大，也可能产生很大的百分比误差；应重点结合中位数、P90 和最大值定位此类项目。",
        "- 新增样本并不保证所有通道和槽位单调改善。逐阶段表中的改善占比反映新增单点对现有 1.7 Ma 工况的迁移影响。",
        "- 建议优先复核“恶化最大的 10 项”，再结合明细 CSV 查看其五阶段轨迹，判断是局部拟合迁移还是参考值接近零导致的比例放大。",
        "- 若用于选最终模型，除总体均值外，应至少同时约束中位数、P90，以及 A/F/G 三类指标不能出现不可接受的单项退化。",
        "",
        "## 10. 数据完整性",
        "",
        f"- 原始长表记录数（含缺失槽位）：{len(long)}。",
        f"- 五阶段完整可比项目数：{len(complete)}。",
        f"- 潜在项目总数（A/F/G × 两种激励 × 12 通道 × 3 槽位）：{3*2*12*3}。",
        "- 详细五阶段数值、逐阶段差值和来源维度见 `ma170_error_detail.csv`；聚合统计见 `ma170_error_summary.csv`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long = read_all()
    wide, complete = pivot_complete(long)

    summary_frames = [
        aggregate(long, ["stage_index", "stage", "stage_label"], "overall"),
        aggregate(long, ["stage_index", "stage", "stage_label", "metric"], "metric"),
        aggregate(long, ["stage_index", "stage", "stage_label", "excitation"], "excitation"),
        aggregate(long, ["stage_index", "stage", "stage_label", "slot"], "slot"),
        aggregate(long, ["stage_index", "stage", "stage_label", "channel"], "channel"),
    ]
    summary = pd.concat(summary_frames, ignore_index=True, sort=False)

    detail_path = OUTPUT_DIR / "ma170_error_detail.csv"
    summary_path = OUTPUT_DIR / "ma170_error_summary.csv"
    report_path = OUTPUT_DIR / "ma170_error_evolution_report.md"
    wide.sort_values(KEYS).to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(long, wide, complete), encoding="utf-8")

    print(f"LONG_ROWS={len(long)} VALID={long['error_pct'].notna().sum()} MISSING={long['error_pct'].isna().sum()}")
    print(f"WIDE_ROWS={len(wide)} COMPLETE_ROWS={len(complete)}")
    print(f"DETAIL={detail_path}")
    print(f"SUMMARY={summary_path}")
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
