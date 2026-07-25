from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from build_quick_model_guide import (
    BLUE,
    DARK_BLUE,
    FONT,
    GRAY,
    INK,
    LIGHT_GRAY,
    NAVY,
    add_p,
    bullet,
    callout,
    code,
    set_font,
    setup_page,
    setup_styles,
    table,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "surrogate_model_模型建立修正与预测使用说明.docx"
GOLD = "7A5A00"
RED = "9B1C1C"


def page_break(doc):
    doc.add_page_break()


def keep_with_next(paragraph):
    paragraph.paragraph_format.keep_with_next = True
    return paragraph


def warning(doc, label, text):
    t = doc.add_table(rows=1, cols=1)
    from build_quick_model_guide import geometry, shade

    geometry(t, [9360])
    cell = t.cell(0, 0)
    shade(cell, "FFF8E8")
    para = cell.paragraphs[0]
    para.paragraph_format.space_after = Pt(0)
    set_font(para.add_run(f"{label}  "), bold=True, color=GOLD)
    set_font(para.add_run(text))
    add_p(doc, "")


def build():
    doc = Document()
    setup_styles(doc)
    setup_page(doc)
    header = doc.sections[0].header.paragraphs[0]
    header.clear()
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(header.add_run("SURROGATE MODEL · CUSTOMER USAGE GUIDE"), size=8.5, bold=True, color=GRAY)

    # Make the document read as a customer-facing operational brief.
    title_style = doc.styles["Title"]
    title_style.font.size = Pt(24)
    title_style.font.color.rgb = RGBColor.from_string(NAVY)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    h1 = doc.styles["Heading 1"]
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(8)
    h2 = doc.styles["Heading 2"]
    h2.paragraph_format.space_before = Pt(11)
    h2.paragraph_format.space_after = Pt(6)

    # Customer-pack-like opening block, without decorative layout tables.
    kicker = add_p(doc, "代理模型系统 · 客户使用说明")
    kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
    kicker.paragraph_format.space_after = Pt(6)
    set_font(kicker.runs[0], size=10, bold=True, color=GOLD)
    title = add_p(doc, "模型建立、修正与预测使用说明", "Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = add_p(
        doc,
        f"适用版本：当前 user_for_win7_fixed_exc_channel_fdr 分支  |  {date.today().isoformat()}",
        "Subtitle",
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    callout(
        doc,
        "说明目的",
        "本文件面向模型使用方，说明从基础模型建立、修正数据集生成、修正模型重训到预测输出的完整过程，"
        "并列出会直接影响模型结果的关键参数和使用边界。",
    )

    add_p(doc, "1. 完整使用流程", "Heading 1")
    table(
        doc,
        ["阶段", "输入", "系统处理", "主要输出"],
        [
            ("基础建模", "仿真数据 + 初始实测数据", "建立各激励/通道的四分量多保真模型", "基础模型版本"),
            ("模型修正", "父模型 + 新增/替换实测工作点", "构建修正数据集、合并历史数据、重训受影响网络", "新的修正模型版本"),
            ("模型预测", "模型版本 + H/M + 激励/通道", "四分量预测、重建极点与残差、计算响应", "PR、HF/HG/HA、VF/VG/VA 等结果"),
        ],
        [1700, 2500, 3400, 1760],
    )
    warning(
        doc,
        "必须理解",
        "基础建模与修正都会生成独立模型版本。修正不是直接改写父模型，也不是简单追加训练；"
        "系统会保留父版本，并基于合并后的完整数据对受影响网络重新训练。",
    )

    add_p(doc, "2. 模型建立", "Heading 1")
    add_p(doc, "2.1 输入数据", "Heading 2")
    bullet(doc, "仿真数据：用于学习整体规律和扩展工作点覆盖范围。")
    bullet(doc, "实测数据：用于学习真实系统与仿真模型之间的偏差。")
    bullet(
        doc,
        "每条样本包含高度 H、马赫数 M、激励 ID、通道 ID，以及模态极点和残差信息；"
        "最终模型最多使用前 3 个有效模态槽。",
    )
    bullet(
        doc,
        "系统按“激励 × 响应通道”建立模型，只训练数据中实际存在的组合；"
        "缺少完整数据的组合不会生成可部署模型。",
    )
    add_p(doc, "2.2 四个独立分量网络", "Heading 2")
    table(
        doc,
        ["分量网络", "预测内容", "对最终结果的影响"],
        [
            ("frequency", "模态频率", "决定响应峰值位置"),
            ("damping", "模态阻尼", "决定衰减和共振强弱"),
            ("residue_magnitude", "残差幅值", "决定响应幅值"),
            ("residue_phase", "残差相位（sinθ、cosθ）", "决定复残差相位关系"),
        ],
        [2600, 2800, 3960],
    )
    add_p(doc, "2.3 两阶段多保真训练", "Heading 2")
    table(
        doc,
        ["阶段", "数据", "训练内容", "默认最大轮数"],
        [
            ("Stage 1", "仿真数据", "训练 low 低保真网络，学习仿真规律", "频率/残差 500；阻尼 200"),
            ("Stage 2", "实测 + 仿真", "学习实测修正，同时保留仿真约束", "频率/残差 500；阻尼 200"),
        ],
        [1700, 1900, 3600, 2160],
    )
    bullet(doc, "连续 50 轮没有有效改进时提前停止，因此实际训练轮数通常小于最大轮数。")
    bullet(doc, "默认按训练损失选择最佳检查点；如明确指定 validation，则按验证损失选择。")
    bullet(doc, "训练完成后，每个激励/通道必须具备四个完整分量，才能进入预测清单。")

    page_break(doc)
    add_p(doc, "3. 修正数据集的构建", "Heading 1")
    add_p(
        doc,
        "修正过程首先要把新增试飞/实测数据转换为与基础模型一致的结构化数据。"
        "系统支持两种输入方式，最终都会得到按工作点拆分的修正数据文件。",
    )
    add_p(doc, "3.1 方式一：直接提供修正数据文件", "Heading 2")
    bullet(doc, "在 user_input.dat 第 4 行起列出一个或多个已整理好的修正数据文件。")
    bullet(doc, "文件必须满足基础数据格式，并包含完整的工作点通道集合。")
    bullet(doc, "适用于外部已完成模态识别、已生成 SF_XGD_*.txt 文件的情况。")

    add_p(doc, "3.2 方式二：从原始 CG/试验 .dat 自动生成", "Heading 2")
    add_p(
        doc,
        "当 user_input.dat 第 4 行以后为空时，系统自动进入 CG/OE 数据构建模式。"
        "该模式使用 selected_CG_path.txt 和极点频段配置.xlsx，从原始时域数据中识别模态。",
    )
    # This is the only numbered list, so numbering remains intuitive.
    for text in (
        "读取 paths.ini 中配置的 CG 数据根目录，并按 selected_CG_path.txt 选择需要修正的 .dat 文件。",
        "从文件名解析高度 H、马赫数 M 和激励类型；文件名必须包含可识别的 H、M、FXD/PW/JFY 及 FDC/DC 标识。",
        "读取输入信号 PM_O1，并按激励位置选择对应的响应通道；缺失通道不会生成该通道样本。",
        "按照极点频段配置.xlsx 中每个通道的 3 个频段、允许误差和可选 OE 阶数识别候选模态。",
        "对输入/输出信号执行 OE 模态识别，将频谱峰值与稳定极点匹配，得到连续极点和复残差。",
        "按 H、M、激励 ID、通道 ID、极点实部/虚部、残差幅值/相位写成 SF_XGD_*.txt 修正文件。",
        "临时修正文件完成合并后自动删除；唯一持久副本写入新模型版本的 dataset_snapshot。",
    ):
        add_p(doc, text, "List Number")

    add_p(doc, "3.3 自动识别中会影响数据的参数", "Heading 2")
    table(
        doc,
        ["参数", "当前默认值", "影响"],
        [
            ("采样频率 FS", "512 Hz", "用于频谱和离散/连续极点换算"),
            ("输入信号列", "PM_O1", "原始文件必须包含该输入列"),
            ("频谱峰值最小突出度", "0 dB", "决定候选频率峰是否被保留"),
            ("默认 OE 阶数", "na=30，nb=30，nk=0", "影响系统辨识复杂度；配置表可按通道覆盖阶数"),
            ("OE 最大迭代次数", "50", "影响拟合收敛程度和耗时"),
            ("频段/容差", "来自极点频段配置.xlsx", "决定每个槽位允许匹配的频率范围"),
        ],
        [2800, 2500, 4060],
        keep_together=True,
    )
    warning(
        doc,
        "数据责任",
        "selected_CG_path.txt、原始文件命名、PM_O1/响应通道列以及极点频段配置表共同决定修正数据。"
        "其中任一项错误，都可能造成工作点无法生成、模态槽位缺失或修正对象错误。",
    )

    add_p(doc, "4. 修正数据的合并与模型重训", "Heading 1")
    add_p(doc, "4.1 工作点如何合并", "Heading 2")
    add_p(
        doc,
        "系统以“高度 H × 100、马赫数 M × 100、激励 ID”作为工作点标识，"
        "一个工作点包含该激励下的一组响应通道。",
    )
    bullet(doc, "修正工作点已存在：用新数据整组替换父版本中的同一工作点。")
    bullet(doc, "修正工作点不存在：作为新工作点加入数据集。")
    bullet(doc, "通道集合必须完整；替换时必须与父工作点通道一致，新增时应与该激励的常规通道集合一致。")
    bullet(doc, "同一工作点不能在多个修正文件中重复出现，否则系统拒绝继续。")
    bullet(doc, "修正工作点强制进入训练集；若原来属于验证集，则从验证集中移除。")
    callout(
        doc,
        "可追溯性",
        "新版本保存合并后的 dataset_snapshot、数据指纹、替换/新增工作点记录和父版本信息，"
        "可回溯本次修正改变了哪些数据。",
    )

    add_p(doc, "4.2 哪些网络会重训", "Heading 2")
    bullet(doc, "默认重训修正数据中涉及的全部激励/通道组合及四个分量网络。")
    bullet(doc, "可使用 --excitation、--channel、--component 进一步限定修正范围。")
    bullet(doc, "修正采用合并后的历史数据 + 本次修正数据完整重训，不从父模型权重继续微调。")
    bullet(doc, "未选中的组合和分量直接继承父版本，不发生变化。")

    add_p(doc, "4.3 提交与失败保护", "Heading 2")
    bullet(doc, "系统先在临时目录训练，一个激励/通道的所有选中分量全部成功后才替换正式目录。")
    bullet(doc, "任一选中分量失败时，临时结果被丢弃，父模型继续保留，并记录到 training_failures.json。")
    bullet(doc, "新模型版本名必须与父版本不同，且不能覆盖已有版本目录。")
    add_p(doc, "4.4 修正时没有验证点", "Heading 2")
    add_p(
        doc,
        "若修正组合仍有验证工作点，则沿用父版本验证划分。若没有验证点，系统改用实测训练损失选模："
        "已有多保真父模型时，达到父模型对应训练损失后默认再训练 50 轮；"
        "首次从仿真模型升级到多保真模型时，则按 Stage 2 最大轮数和实测训练损失选择。",
    )

    add_p(doc, "5. 预测流程", "Heading 1")
    add_p(
        doc,
        "预测时，系统从 models_manifest.json 查找目标激励/通道，加载四个分量检查点及各自归一化器，"
        "预测频率、阻尼、残差幅值和相位，反归一化后重建极点与复残差，再进入响应计算。",
    )
    bullet(doc, "有实测数据的模型使用 high（实测修正后）输出；只有仿真数据的模型使用 low 输出。")
    bullet(doc, "清单中不存在的激励/通道组合不会进行有效预测，可能表现为空或零结果。")
    bullet(doc, "默认 extrapolation_policy=allow，即直接使用网络输出进行外推。")
    bullet(
        doc,
        "如显式选择 smooth_linear，则只对阻尼在实测训练马赫数范围外做平滑线性外推；"
        "训练高度范围之外仍使用网络输出。",
    )
    bullet(
        doc,
        "预测阻尼默认下限为 2、默认无上限。扫描结果中越界只清空对应阻尼格；"
        "PR/RESULT_PR 响应计算会拒绝整个越界模态槽。",
    )

    page_break(doc)
    add_p(doc, "6. 关键参数与使用影响", "Heading 1")
    table(
        doc,
        ["类别", "参数/默认值", "使用方需要知道的影响"],
        [
            ("模型结构", "每组合 4 个分量网络；K_max=3", "缺少任一分量时该组合不能完整部署；只保留前 3 个模态槽"),
            ("训练轮数", "频率/残差每阶段 500；阻尼每阶段 200", "为最大轮数，可能因早停提前结束"),
            ("早停", "patience=50，min_delta=1e-6", "决定何时停止以及最终检查点"),
            ("批量大小", "sim=16，real=8", "影响显存、训练速度和优化波动"),
            ("学习率", "Stage 1/2 均为 1e-3", "影响收敛速度和稳定性"),
            ("Stage 2 损失", "实测权重 1.0；仿真权重 0.2", "决定修正对实测拟合与仿真规律保持的平衡"),
            ("检查点选择", "默认 training", "改为 validation 会改变最终选中的模型轮次"),
            ("并行训练", "parallel_workers 默认 1", "提高并行数会增加显存/内存占用；不会改变任务含义"),
            ("修正追加轮数", "达到父损失后再训练 50 轮", "仅用于无验证点且有可比父损失的修正场景"),
            ("外推策略", "allow；可选 smooth_linear", "直接决定阻尼超出训练马赫数范围时的行为"),
            ("阻尼有效范围", "预测下限 2；上限默认关闭", "越界槽位会被清空或拒绝参与响应计算"),
        ],
        [1900, 2800, 4660],
    )
    warning(
        doc,
        "参数变更原则",
        "以上参数均会影响训练耗时、模型选择或预测结果。交付后如需修改，"
        "应生成新的模型版本并保留原版本，不能仅修改配置后继续使用旧检查点。",
    )

    add_p(doc, "7. 使用入口与必要配置", "Heading 1")
    table(
        doc,
        ["任务", "user_input.dat 内容", "运行入口"],
        [
            ("基础建模", "第 1 行 paths.ini；第 2 行模型版本名", "cli.train"),
            (
                "模型修正",
                "第 1 行 paths.ini；第 2 行父版本；第 3 行新版本；第 4 行起为可选修正文件",
                "cli.train_incremental",
            ),
            ("模型预测", "按既有预测任务文件提供模型版本和工作点", "test.Predict_user"),
        ],
        [1700, 5000, 2660],
        keep_together=True,
    )
    add_p(doc, "常用命令示例", "Heading 2")
    code(
        doc,
        "python -m surrogate_model.cli.train --parallel_workers 4 --num_workers 0\n"
        "python -m surrogate_model.cli.train --resume --parallel_workers 4\n"
        "python -m surrogate_model.cli.train_incremental --component damping\n"
        "python -m surrogate_model.cli.train_incremental "
        "--excitation FXD_DC --parallel_workers 4 --num_workers 0",
    )

    add_p(doc, "8. 使用前检查清单", "Heading 1")
    for text in (
        "确认模型版本名正确，修正时父版本、新版本不相同，且新版本目录尚不存在。",
        "确认基础训练目录同时包含 sim_dataset 和 real_dataset。",
        "自动构建修正数据时，确认 selected_CG_path.txt、原始 .dat、PM_O1、响应通道和极点频段配置表一致。",
        "确认修正工作点的通道集合完整，没有同一工作点在多个文件中重复。",
        "训练结束后检查 models_manifest.json 的模型数量，以及 training_failures.json 是否存在失败记录。",
        "预测前确认目标激励/通道存在于 models_manifest.json，并确认外推策略与阻尼阈值满足使用要求。",
        "保存本次使用的模型版本、数据快照和训练配置；不要直接覆盖已交付模型版本。",
    ):
        bullet(doc, text)

    doc.core_properties.title = "代理模型建立、修正与预测使用说明"
    doc.core_properties.subject = "客户侧模型生命周期与关键参数说明"
    doc.core_properties.author = "surrogate_model 项目组"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
