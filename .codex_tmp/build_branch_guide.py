from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "surrogate_model_当前分支技术说明.docx"

FONT = "Microsoft YaHei"
MONO = "Consolas"
NAVY = "17365D"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
LIGHT_BLUE = "E8EEF5"
PALE_BLUE = "F4F7FB"
LIGHT_GRAY = "F2F4F7"
MID_GRAY = "6B7280"
INK = "202124"
RED = "9B1C1C"
GOLD = "7A5A00"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int], indent_dxa: int = 120) -> None:
    total = sum(widths_dxa)
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths_dxa[idx]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)


def set_run_font(run, size=None, bold=None, color=None, name=FONT, italic=None) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for name, size, color, before, after in (
        ("Title", 28, NAVY, 0, 8),
        ("Subtitle", 13, MID_GRAY, 0, 14),
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 14, 7),
        ("Heading 3", 11.5, DARK_BLUE, 10, 5),
    ):
        style = styles[name]
        style.font.name = FONT
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = name != "Subtitle"
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Number"):
        style = styles[name]
        style.font.name = FONT
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.font.size = Pt(10.5)
        style.paragraph_format.left_indent = Inches(0.375)
        style.paragraph_format.first_line_indent = Inches(-0.188)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.25


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("第 ")
    set_run_font(run, size=9, color=MID_GRAY)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)
    run = paragraph.add_run(" 页")
    set_run_font(run, size=9, color=MID_GRAY)


def configure_page(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.82)
    sec.bottom_margin = Inches(0.78)
    sec.left_margin = Inches(0.88)
    sec.right_margin = Inches(0.88)
    sec.header_distance = Inches(0.35)
    sec.footer_distance = Inches(0.35)
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("SURROGATE MODEL  ·  CURRENT BRANCH GUIDE")
    set_run_font(run, size=8.5, bold=True, color=MID_GRAY)
    add_page_number(sec.footer.paragraphs[0])


def p(doc, text="", *, style=None, bold_prefix=None, color=None, keep=False):
    para = doc.add_paragraph(style=style)
    para.paragraph_format.keep_together = keep
    if bold_prefix and text.startswith(bold_prefix):
        first = para.add_run(bold_prefix)
        set_run_font(first, bold=True, color=color)
        rest = para.add_run(text[len(bold_prefix):])
        set_run_font(rest, color=color)
    else:
        run = para.add_run(text)
        set_run_font(run, color=color)
    return para


def bullet(doc, text):
    return p(doc, text, style="List Bullet")


def numbered(doc, text):
    return p(doc, text, style="List Number")


def code_block(doc, lines: str):
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Inches(0.18)
    para.paragraph_format.right_indent = Inches(0.18)
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after = Pt(8)
    para.paragraph_format.line_spacing = 1.05
    p_pr = para._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F6F8FA")
    p_pr.append(shd)
    run = para.add_run(lines)
    set_run_font(run, size=8.4, color="24292F", name=MONO)
    return para


def note(doc, label: str, text: str, kind="info"):
    fill = {"info": PALE_BLUE, "warn": "FFF8E8", "risk": "FDECEC"}[kind]
    accent = {"info": BLUE, "warn": GOLD, "risk": RED}[kind]
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    set_table_geometry(table, [9360], indent_dxa=120)
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    para = cell.paragraphs[0]
    para.paragraph_format.space_after = Pt(0)
    r = para.add_run(f"{label}  ")
    set_run_font(r, bold=True, color=accent)
    r = para.add_run(text)
    set_run_font(r)
    p(doc, "")


def add_table(
    doc,
    headers,
    rows,
    widths,
    *,
    header_fill=LIGHT_BLUE,
    font_size=9.2,
    keep_together=False,
):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    set_table_geometry(table, widths, indent_dxa=120)
    hdr = table.rows[0]
    hdr._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for i, value in enumerate(headers):
        cell = hdr.cells[i]
        set_cell_shading(cell, header_fill)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(0)
        run = para.add_run(str(value))
        set_run_font(run, size=font_size, bold=True, color=NAVY)
    for row_values in rows:
        row = table.add_row()
        for i, value in enumerate(row_values):
            cell = row.cells[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            if i == 0 and len(headers) <= 3:
                run = para.add_run(str(value))
                set_run_font(run, size=font_size, bold=True, color=DARK_BLUE)
            else:
                run = para.add_run(str(value))
                set_run_font(run, size=font_size)
    if keep_together:
        for row in table.rows[:-1]:
            for cell in row.cells:
                p_pr = cell.paragraphs[0]._p.get_or_add_pPr()
                if p_pr.find(qn("w:keepNext")) is None:
                    p_pr.append(OxmlElement("w:keepNext"))
    p(doc, "")
    return table


def page_break(doc):
    doc.add_page_break()


def build() -> None:
    doc = Document()
    configure_styles(doc)
    configure_page(doc)
    props = doc.core_properties
    props.title = "surrogate_model 当前分支技术说明"
    props.subject = "构建、训练、修正、预测与维护说明"
    props.author = "surrogate_model 项目组"
    props.keywords = "surrogate model, FDR, multifidelity, training, correction"

    # Cover
    p(doc, "技术说明 / 操作手册", color=GOLD).alignment = WD_ALIGN_PARAGRAPH.CENTER
    title = p(doc, "surrogate_model 当前分支技术说明", style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = p(doc, "构建、基础训练、数据修正、预测与模型产物", style="Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p(doc, "")
    meta = add_table(
        doc,
        ["项目项", "当前值"],
        [
            ("分支", "user_for_win7_fixed_exc_channel_fdr"),
            ("提交", "ec42657（文档生成时的 HEAD）"),
            ("软件包", "surrogate_model 0.1.0"),
            ("模型布局", "每个激励/响应通道组合 4 个独立 FDR 分量网络"),
            ("文档日期", date.today().isoformat()),
        ],
        [2200, 7160],
        header_fill=LIGHT_GRAY,
        font_size=10,
    )
    p(doc, "")
    lead = p(
        doc,
        "本文档以当前分支代码为准，面向需要安装、训练、执行修正、恢复任务和检查模型产物的开发与使用人员。"
        "命令示例默认在仓库上一级目录执行，使 Python 能按 surrogate_model 包名导入本项目。",
        keep=True,
    )
    lead.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    note(
        doc,
        "一句话概览",
        "系统以高度 H、马赫数 M、激励 ID 和通道 ID 为输入，分别预测频率、阻尼、残差幅值与残差相位；"
        "基础训练先学习仿真低保真头，再用实测数据学习线性映射与非线性修正头。",
    )
    page_break(doc)

    p(doc, "1. 项目定位与当前分支能力", style="Heading 1")
    p(
        doc,
        "本项目是面向 PZK/XGD 场景的代理模型 Python 包。数据中的每个工作点由 H、M、激励、通道和最多 3 个模态槽组成。"
        "当前分支固定使用 freq_damping 极点表示和 polar 残差表示，将一个激励/通道组合拆成四类独立网络。",
    )
    add_table(
        doc,
        ["分量目录", "学习目标", "输出含义"],
        [
            ("frequency", "freq", "每个槽位的频率"),
            ("damping", "damping", "每个槽位的阻尼；显示/阈值检查时按代码路径换算"),
            ("residue_magnitude", "magnitude", "物理残差幅值"),
            ("residue_phase", "sin_theta, cos_theta", "相位的正余弦，网络输出后单位化"),
        ],
        [2200, 2600, 4560],
    )
    p(doc, "主要能力", style="Heading 2")
    for item in (
        "按数据中实际存在的激励/通道组合建模；无可用训练/验证划分的组合会记录失败，但不会阻断其他组合。",
        "支持按激励、通道、分量和高度筛选训练范围；筛选器不区分大小写，可重复传入，也可用逗号或分号分隔。",
        "支持多个独立分量网络并行训练；不支持 torchrun/DDP，因为任务粒度本身就是独立网络。",
        "支持断点式恢复：完整分量直接保留，缺失或不完整分量重新从第 1 轮开始。",
        "支持基于父版本的修正版本：继承未选模型，合并历史与修正数据，对受影响分量完整重训。",
        "保存数据快照、清单、训练配置、日志、归一化器和失败记录，便于追踪与部署。",
    ):
        bullet(doc, item)

    p(doc, "代码结构", style="Heading 2")
    add_table(
        doc,
        ["路径", "职责"],
        [
            ("config.py", "集中默认超参数、验证划分、阻尼过滤和训练配置"),
            ("cli/train.py", "基础版本训练入口、选择器、并行队列、恢复检查"),
            ("cli/train_incremental.py", "修正版本入口、历史合并、父模型继承、原子替换"),
            ("train/dataloaders.py", "划分、过滤、归一化与 DataLoader 构建"),
            ("train/loops.py", "两阶段训练、早停、调度器、日志与检查点"),
            ("train/full_training_runner.py", "按分量应用配置、组织产物和元数据"),
            ("models/blocks.py", "低保真头、线性高保真映射、非线性适配头"),
            ("test/Predict_user.py", "模型加载、外推策略、阻尼范围检查和响应输出"),
        ],
        [2600, 6760],
    )

    p(doc, "2. 环境与构建", style="Heading 1")
    p(doc, "2.1 建议环境", style="Heading 2")
    p(
        doc,
        "仓库的 setup.py/pyproject.toml 只声明了构建后端，没有声明运行依赖版本。"
        "因此部署环境需要显式准备依赖。代码直接使用 PyTorch、NumPy、SciPy、pandas、scikit-learn；"
        "测试工具使用 pytest（测试文件中也包含 unittest 风格用例）。GPU 训练还要求与 CUDA 匹配的 PyTorch。",
    )
    code_block(
        doc,
        "python -m venv .venv\n"
        ".\\.venv\\Scripts\\Activate.ps1\n"
        "python -m pip install --upgrade pip setuptools wheel\n"
        "python -m pip install torch numpy scipy pandas scikit-learn pytest\n"
        "python -m pip install -e .",
    )
    note(
        doc,
        "依赖提示",
        "应根据目标机器的 CUDA/CPU 条件安装对应的 PyTorch 构建；项目本身不会替你锁定 torch 版本。",
        "warn",
    )
    p(doc, "2.2 包路径与运行目录", style="Heading 2")
    p(
        doc,
        "安装后包名为 surrogate_model。推荐在仓库的父目录运行 python -m surrogate_model...。"
        "训练入口会以 cli 目录为 base_dir，并要求 user_input.dat、sim_dataset、real_dataset 位于该入口旁边。",
    )
    code_block(
        doc,
        "# 仓库位于 D:\\Code\\Python\\surrogate_model 时\n"
        "Set-Location D:\\Code\\Python\n"
        "python -m surrogate_model.cli.train --help\n"
        "python -m surrogate_model.cli.train_incremental --help",
    )
    p(doc, "2.3 必需目录与配置", style="Heading 2")
    add_table(
        doc,
        ["项目", "基础训练", "修正训练"],
        [
            ("cli/user_input.dat", "至少 2 行：paths.ini、模型版本名", "至少 3 行：paths.ini、父版本、新版本；第 4 行起为修正文件"),
            ("cli/sim_dataset", "必需；允许同工作点重复键", "必需；用于多保真修正"),
            ("cli/real_dataset", "必需；基础实测数据", "不直接作为修正输入"),
            ("paths.ini", "从 [MX_XGD]/path_mx_xgd 解析模型根目录", "还从 [CG] 解析 CG 数据目录"),
            ("父版本产物", "不需要", "必须有 models_manifest.json；优先使用 dataset_snapshot"),
        ],
        [2300, 3500, 3560],
    )
    p(doc, "user_input.dat 示例", style="Heading 3")
    code_block(
        doc,
        "D:\\...\\J35_001_none_Config.ini\n"
        "base_model_v1\n"
        "# 修正模式再增加：\n"
        "corrected_model_v2\n"
        "D:\\...\\real_dataset\\SF_XGD_FXD_DC_H0823_M170.txt",
    )
    p(doc, "3. 数据契约与预处理", style="Heading 1")
    p(doc, "3.1 输入格式", style="Heading 2")
    p(
        doc,
        "数据行基础格式为 H, M, exc_id, ch_id，后续每个模态槽用 4 个值表示：Re、Im、rho/magnitude、theta。"
        "解析器也兼容在四个基础字段后额外出现两个字段的旧格式。文件名必须遵循 "
        "SF_XGD_* 或 FZ_XGD_* 命名，并与行内 H、M、激励 ID 一致。",
    )
    add_table(
        doc,
        ["约束", "当前实现"],
        [
            ("最大槽位数", "K_max = 3；多余槽位忽略，缺失槽位补零并 mask=0"),
            ("无效极点", "Re/Im 非有限、Re/Im 同为 0 或 Im≈0 的槽位屏蔽，保留原槽位顺序"),
            ("激励", "0..5：FXD_FDC、FXD_DC、PW_FDC、PW_DC、JFY_FDC、JFY_DC"),
            ("通道", "固定 34 个名称，由 LW1 开始，到 YWL1Z 结束"),
            ("目标表示", "pole_target=freq_damping；residue_target=polar"),
        ],
        [2450, 6910],
    )
    p(doc, "3.2 划分与归一化", style="Heading 2")
    p(
        doc,
        "默认验证策略为 height_mach_points，显式验证点为 (3.0,0.98)、(5.5,1.05)、"
        "(5.5,1.17)、(5.5,1.3)、(8.23,1.7)。验证比例字段为 0.2、随机种子为 42，"
        "但具体划分以所选策略实现为准。使用 --height 限制高度后，如果该高度没有配置验证点，"
        "程序自动退回该高度最大马赫数作为验证点。",
    )
    bullet(doc, "输入归一化只使用 H、M；仿真和实测共享仿真坐标系的输入变换。")
    bullet(doc, "仿真与实测输出归一化器分别在各自训练拆分上拟合。")
    bullet(doc, "归一化器按保真度命名：Normalizer_Input/Output_Sim.pth 与 _Real.pth。")
    bullet(doc, "每个分量目录保存独立输出变换；residue_magnitude 直接学习物理幅值。")

    p(doc, "3.3 槽位过滤", style="Heading 2")
    p(
        doc,
        "仿真数据启用鲁棒槽位异常筛查：最少样本 8、MAD 阈值 12、尺度下限比例 0.02。"
        "实测阻尼上限筛查也默认启用，超限时对四个 FDR 分量共同屏蔽整个模态槽，"
        "训练和验证拆分均应用，并写入 damping_slot_filter.log。",
    )
    add_table(
        doc,
        ["激励 ID", "阻尼阈值", "应用通道 ID"],
        [
            ("0 FXD_FDC", "0.15", "8–19"),
            ("1 FXD_DC", "0.15", "8–19"),
            ("2 PW_FDC", "0.30", "20–27"),
            ("3 PW_DC", "0.30", "20–27"),
            ("4 JFY_FDC", "0.30", "0–7"),
            ("5 JFY_DC", "0.30", "0–7"),
        ],
        [2600, 2200, 4560],
    )

    p(doc, "4. 模型结构", style="Heading 1")
    p(
        doc,
        "每个激励/通道/分量都有独立的 TargetMultifidelityMFDNN。"
        "输入由标准化 H、M 加激励嵌入和通道嵌入组成；默认嵌入维度分别为 4 和 12。"
        "低保真 MLP 的隐藏层为 [64,128,64]，激活函数 GELU，dropout=0。",
    )
    add_table(
        doc,
        ["模块", "结构与作用"],
        [
            ("low", "低保真 TargetHead；从仿真数据学习目标分量"),
            ("lin", "逐槽位仿射映射，初始权重为单位阵、偏置为 0"),
            ("adaptation", "输入 H/M、嵌入与 low 均值；默认隐藏层 [32,32]，学习非线性实测修正"),
            ("high", "lin(low) + adaptation(...)；有实测数据时作为部署输出"),
        ],
        [2100, 7260],
        keep_together=True,
    )
    code_block(
        doc,
        "输入: [Hn, Mn, excitation_embedding, channel_embedding]\n"
        "Stage 1: low = MLP(input)\n"
        "Stage 2: high = Linear(low) + Adaptation(Hn, Mn, embeddings, mean(low))\n"
        "相位分量: (sin_theta, cos_theta) / sqrt(sin² + cos² + 1e-8)",
    )
    note(
        doc,
        "注意",
        "虽然模型保留激励/通道嵌入接口，但当前目录粒度已经是单个激励/通道组合；"
        "这样可以保持统一前向接口和检查点元数据。",
    )
    p(doc, "5. 训练细节", style="Heading 1")
    p(doc, "5.1 默认超参数", style="Heading 2")
    add_table(
        doc,
        ["参数", "默认值", "说明"],
        [
            ("batch_sim / batch_real", "16 / 8", "仿真与实测批量大小"),
            ("num_workers / prefetch", "8 / 4", "并行分量训练时未指定 num_workers 会自动改为 0"),
            ("Stage 1 / Stage 2 学习率", "1e-3 / 1e-3", "均使用 AdamW"),
            ("weight_decay", "5e-5", "全分量默认相同"),
            ("学习率调度", "ReduceLROnPlateau", "监控训练损失；factor=0.5，patience=20，min_lr=1e-6"),
            ("梯度裁剪", "max_norm=5.0", "两阶段均使用"),
            ("早停 min_delta", "1e-6", "改进小于该值不计为新最佳"),
            ("检查点选择", "training", "默认按训练损失；可改为 validation"),
            ("Stage 2 损失权重", "real=1.0, sim=0.2", "loss_total = real_weight*loss_real + sim_weight*loss_sim"),
        ],
        [2700, 2200, 4460],
    )
    p(doc, "5.2 训练轮数与早停", style="Heading 2")
    add_table(
        doc,
        ["分量", "Stage 1 最大轮数", "Stage 2 最大轮数", "两阶段早停耐心"],
        [
            ("frequency", "500", "500", "50"),
            ("damping", "200", "200", "50"),
            ("residue_magnitude", "500", "500", "50"),
            ("residue_phase", "500", "500", "50"),
        ],
        [2700, 2200, 2200, 2260],
    )
    p(
        doc,
        "最大轮数不是必然执行轮数。若连续 50 轮没有超过 min_delta 的改进，会提前停止。"
        "Stage 1 只解冻 low 头，激励/通道嵌入保持冻结；Stage 2 解冻 lin 和 adaptation，"
        "默认还解冻 low，除非传入 --freeze_low_stage2。",
    )
    p(doc, "5.3 两阶段流程", style="Heading 2")
    numbered(doc, "为当前激励/通道/分量构建仿真与实测训练/验证 DataLoader，并拟合训练拆分归一化器。")
    numbered(doc, "Stage 1 仅用仿真数据训练 low；每轮计算仿真训练损失和可用的验证损失。")
    numbered(doc, "恢复 Stage 1 最佳 low 权重；若没有实测数据，保存 sim_only 模型并结束。")
    numbered(doc, "Stage 2 用实测批次训练 high，同时循环抽取仿真批次作为软约束。")
    numbered(doc, "按 training 或 validation 指标保存最佳模型；调度器始终以训练损失降学习率。")
    numbered(doc, "保存 model.pth、归一化器、train_log.csv、training_config.json 和槽位支持信息。")
    p(doc, "5.4 检查点门槛", style="Heading 2")
    p(
        doc,
        "通用训练损失上限为 0.05；分量默认上限为 frequency=0.05、damping=None、"
        "residue_magnitude=0.02、residue_phase=0.02。使用验证指标时，只有实测训练损失满足上限的轮次"
        "才有资格成为正式最佳检查点；如果没有合格轮次，代码会退回原始指标最好的有限检查点。",
    )

    p(doc, "6. 基础模型构建流程", style="Heading 1")
    p(doc, "6.1 最小运行", style="Heading 2")
    code_block(doc, "python -m surrogate_model.cli.train")
    p(
        doc,
        "入口读取 cli/user_input.dat 前两行，加载 cli/sim_dataset 与 cli/real_dataset，"
        "按实测数据中出现的组合排队训练。每个组合最多产生 4 个独立网络。",
    )
    p(doc, "6.2 推荐并行与恢复命令", style="Heading 2")
    code_block(
        doc,
        "python -m surrogate_model.cli.train --parallel_workers 4 --num_workers 0\n"
        "python -m surrogate_model.cli.train --resume --parallel_workers 4 --num_workers 0",
    )
    note(
        doc,
        "并行原则",
        "parallel_workers 是独立网络进程数。单 GPU 建议从 2、3、4 逐个试测吞吐和显存；"
        "不要对同一输出目录同时启动多条相同训练命令。",
        "warn",
    )
    p(doc, "6.3 精确选择训练范围", style="Heading 2")
    code_block(
        doc,
        "python -m surrogate_model.cli.train --excitation FXD_DC --channel LVT1,LVT2\n"
        "python -m surrogate_model.cli.train --component damping --height 8.23\n"
        "python -m surrogate_model.cli.train --components frequency,damping "
        "--excitation 0,1 --channels 8,9 --parallel_workers 3",
    )
    p(
        doc,
        "显式选择的 component 即使与 --resume 同时使用也会原地重训；未选择的分量不进入队列。"
        "若不指定 component，则 --resume 仅跳过完整产物。",
    )
    p(doc, "6.4 常用超参数覆盖", style="Heading 2")
    code_block(
        doc,
        "python -m surrogate_model.cli.train `\n"
        "  --epochs_s1 300 --epochs_s2 300 `\n"
        "  --batch_sim 32 --batch_real 16 `\n"
        "  --lr 0.001 --lr_high 0.0005 `\n"
        "  --checkpoint_selection validation",
    )
    p(
        doc,
        "--epochs 会同时覆盖两阶段；--epochs_s1/--epochs_s2 分别覆盖。"
        "分量级检查点策略与 Stage 2 权重可使用 component=value 形式配置。",
    )
    code_block(
        doc,
        "python -m surrogate_model.cli.train `\n"
        "  --component_checkpoint_selection damping=validation,frequency=training `\n"
        "  --component_real_loss_weight damping=1.0 `\n"
        "  --component_sim_loss_weight damping=0.2",
    )
    p(doc, "7. 修正流程（增量版本）", style="Heading 1")
    note(
        doc,
        "核心语义",
        "修正流程不是加载父检查点后继续微调。它先继承父版本完整目录，再将历史与修正数据按工作点合并，"
        "只对受影响且被选中的分量从头完整重训。",
        "info",
    )
    p(doc, "7.1 输入与候选组合", style="Heading 2")
    p(
        doc,
        "user_input.dat 第 2 行是父模型版本，第 3 行是新模型版本，第 4 行起是新增/替换数据文件。"
        "这些修正文件中的激励/通道组合构成候选集合；--excitation、--channel 和 --component 继续收窄集合。",
    )
    p(
        doc,
        "如果第 4 行以后为空，程序进入 CG/OE 模式：读取 selected_CG_path.txt 和极点频段配置 Excel，"
        "执行批量 OE 提取，在临时目录生成拆分文件；临时文件合并后删除，只在新版本 dataset_snapshot 中保留持久副本。",
    )
    p(doc, "7.2 合并与划分规则", style="Heading 2")
    numbered(doc, "验证父版本存在且含 models_manifest.json，拒绝覆盖不安全的新版本目录。")
    numbered(doc, "复制所有父模型目录到新版本，并将状态标记为 inherited。")
    numbered(doc, "读取父数据快照；旧版本无快照时可用 --history_dataset 或配置中的迁移路径。")
    numbered(doc, "按 (H,M,excitation) 工作点合并历史与修正数据；重复修正组会报错。")
    numbered(doc, "修正工作点加入训练集合，并从父验证集合移除；其他父划分尽量保持不变。")
    numbered(doc, "为受影响组合建立临时训练目录，所有选中分量成功后再替换正式目录。")
    numbered(doc, "任一选中分量失败时删除临时结果，保留父模型，并写入 training_failures.json。")
    p(doc, "7.3 无验证点时的停止规则", style="Heading 2")
    p(
        doc,
        "如果修正后该组合仍有验证点，沿用父版本划分。若没有验证点，检查点策略强制为 training。"
        "对已有多保真父模型，程序读取父模型每个分量的训练损失作为目标；新训练达到父损失后，"
        "默认再训练 50 轮（--epochs_after_parent_loss）即停止。"
        "如果父模型是 sim_only，因不存在可比的实测损失，则按 Stage 2 最大轮数和实测训练损失选择。",
    )
    p(doc, "7.4 示例", style="Heading 2")
    code_block(
        doc,
        "python -m surrogate_model.cli.train_incremental --component damping\n"
        "python -m surrogate_model.cli.train_incremental --component damping "
        "--excitation PW_FDC --channel LHT1\n"
        "python -m surrogate_model.cli.train_incremental --excitation FXD_DC,FXD_FDC "
        "--parallel_workers 8 --num_workers 0",
    )

    p(doc, "8. 模型产物与恢复判定", style="Heading 1")
    code_block(
        doc,
        "<model_version>/\n"
        "  dataset_snapshot/                 # 实测数据快照\n"
        "  sim_dataset_snapshot/             # 仿真数据快照\n"
        "  dataset_manifest.json\n"
        "  sim_dataset_manifest.json\n"
        "  models_manifest.json              # 可部署组合清单（schema v5）\n"
        "  training_failures.json\n"
        "  FXD_FDC/LVT1/frequency/\n"
        "    model.pth\n"
        "    Normalizer_Input_Sim.pth\n"
        "    Normalizer_Output_Sim.pth\n"
        "    Normalizer_Input_Real.pth\n"
        "    Normalizer_Output_Real.pth\n"
        "    training_config.json\n"
        "    train_log.csv\n"
        "    slot_support.json",
    )
    p(
        doc,
        "resume 会检查检查点、对应保真度归一化器、配置、日志、slot_support 等是否完整，"
        "还会核对训练 profile 版本、保真度模式和高度选择。优化器和调度器状态不保存，"
        "因此不完整分量不是续接某一 epoch，而是重新训练。",
    )
    add_table(
        doc,
        ["文件", "用途"],
        [
            ("models_manifest.json", "预测端可加载组合清单；完整四分量组合才可部署"),
            ("training_config.json", "实际生效超参数、数据指纹、归一化器哈希、选择结果"),
            ("train_log.csv", "逐轮 Stage、损失、验证损失、学习率"),
            ("slot_support.json", "每个槽位的训练支持范围与相关元数据"),
            ("training_failures.json", "跳过或失败组合、异常类型、父模型是否保留"),
        ],
        [2750, 6610],
    )
    p(doc, "9. 预测、外推与阻尼检查", style="Heading 1")
    p(
        doc,
        "预测入口按 models_manifest.json 找到匹配的激励/通道组合，加载四个分量检查点和各自归一化器，"
        "反归一化后重建极点实部/虚部和复残差，再进入既有响应计算流程。清单中不存在的组合按设计输出空/零结果。",
    )
    p(doc, "9.1 外推策略", style="Heading 2")
    bullet(doc, "默认 --extrapolation_policy allow：所有分量直接使用网络输出。")
    bullet(
        doc,
        "显式 smooth_linear：仅对阻尼在实测训练马赫数支持范围外做平滑线性尾部；边界通过三次 Hermite 过渡，"
        "保证值与一阶导数连续。",
    )
    bullet(doc, "过渡宽度和线性拟合宽度默认都是 0.05 Mach；高度超出训练高度范围时仍用网络预测。")
    p(doc, "9.2 预测阻尼范围", style="Heading 2")
    p(
        doc,
        "代码按显示侧阻尼值执行范围检查。默认下限为 2，上限不限制。"
        "HF/HG/HA 与 VF/VG/VA 扫描文件中，越界只清空对应 HG/VG 阻尼单元格，频率和响应幅值保留；"
        "PR 与 RESULT_PR 路径会拒绝整个模态槽。边界值包含在合法范围内。",
    )
    code_block(
        doc,
        "python -m surrogate_model.test.Predict_user "
        "--prediction_damping_min 2 --prediction_damping_max 60\n"
        "python -m surrogate_model.test.Predict_user --prediction_damping_min none\n"
        "python -m surrogate_model.test.Predict_user --extrapolation_policy smooth_linear "
        "--damping_extrapolation_transition_mach 0.05",
    )

    p(doc, "10. 当前分支的主要修正点", style="Heading 1")
    p(
        doc,
        "根据当前分支最近提交和代码状态，本轮演进主要围绕训练选择、阻尼数据质量、外推和产物一致性。",
    )
    add_table(
        doc,
        ["修正主题", "当前行为"],
        [
            ("按高度训练", "基础训练可用 --height；验证点随选择高度收敛，必要时退回最大 Mach"),
            ("阻尼异常处理", "仿真槽位做鲁棒筛查，实测槽位按激励/通道阈值整体屏蔽"),
            ("阻尼外推", "默认允许网络直接外推；可显式启用 C1 连续的 smooth_linear"),
            ("训练策略", "四分量可独立设置轮数、检查点指标和 Stage 2 实测/仿真权重"),
            ("归一化器命名", "统一为 Sim/Real 明确命名，并在 manifest/config 中记录哈希"),
            ("并行与恢复", "按分量多进程队列；完整分量跳过，不完整分量重启"),
            ("修正安全性", "组合级临时目录与原子替换；失败保留父模型"),
            ("预测阻尼检查", "扫描输出按单元格清空，PR 响应链路按整槽拒绝"),
        ],
        [2750, 6610],
    )

    p(doc, "11. 验证、排错与交付检查", style="Heading 1")
    p(doc, "11.1 测试", style="Heading 2")
    code_block(doc, "python -m pytest -q")
    p(
        doc,
        "测试覆盖训练选择、修正选择、FDR 拆分、并行任务、阻尼多保真烟雾测试和 OE 批处理配置。"
        "如果提示 No module named pytest，需要先安装 pytest；这不代表代码测试失败，而是测试工具未安装。",
    )
    p(doc, "11.2 常见问题", style="Heading 2")
    add_table(
        doc,
        ["现象", "检查方向"],
        [
            ("找不到数据目录", "确认以包方式运行，且 sim_dataset/real_dataset 位于 cli 入口旁"),
            ("找不到模型根目录", "检查 paths.ini 的 [MX_XGD] / path_mx_xgd"),
            ("修正找不到父版本", "确认父目录和 models_manifest.json 存在，版本名无空格"),
            ("某组合被跳过", "查看 training_failures.json 和该组合训练/验证拆分是否可用"),
            ("显存不足", "降低 parallel_workers、batch_sim、batch_real；多进程时 num_workers=0"),
            ("resume 仍重训", "检查产物完整性、profile 版本、保真度模式、高度选择是否一致"),
            ("阻尼槽位消失", "检查 damping_slot_filter.log 与预测 damping min/max"),
            ("预测组合为空", "确认 models_manifest.json 中存在该 excitation/channel 完整四分量组合"),
        ],
        [2650, 6710],
    )
    p(doc, "11.3 训练交付清单", style="Heading 2")
    for item in (
        "记录 user_input.dat、命令行参数、当前 Git 提交和模型版本名。",
        "确认 dataset_manifest.json 与 sim_dataset_manifest.json 已生成，数据指纹可追踪。",
        "确认 models_manifest.json 的 model_count/network_count 与预期一致。",
        "逐项检查 training_failures.json；失败组合是否允许保留父模型或跳过。",
        "检查各分量 training_config.json 的实际轮数、指标、损失权重和 normalizer 哈希。",
        "检查 train_log.csv 的最佳轮次、早停行为、学习率变化和有限损失。",
        "用代表性 H/M/激励/通道执行 Predict_user，检查阻尼边界与输出目录。",
        "不要并发写同一个模型版本目录；确认新版本名不会覆盖既有版本。",
    ):
        bullet(doc, item)

    p(doc, "附录 A：参数速查", style="Heading 1")
    add_table(
        doc,
        ["参数", "作用"],
        [
            ("--epochs / --epochs_s1 / --epochs_s2", "覆盖两阶段或单阶段最大轮数"),
            ("--lr / --lr_high", "Stage 1 / Stage 2 学习率"),
            ("--batch_sim / --batch_real", "仿真/实测批量大小"),
            ("--parallel_workers", "独立分量训练进程数"),
            ("--resume", "跳过完整分量；显式选择的分量仍重训"),
            ("--excitation / --channel / --component", "限定训练或修正范围"),
            ("--height", "基础训练限定高度"),
            ("--checkpoint_selection", "training 或 validation"),
            ("--component_*", "按分量覆盖检查点策略或损失权重"),
            ("--freeze_low_stage2", "Stage 2 冻结低保真头"),
            ("--epochs_after_parent_loss", "无验证修正达到父损失后继续轮数，默认 50"),
            ("--device", "auto / cuda / cpu"),
        ],
        [3550, 5810],
    )
    p(doc, "附录 B：本文档依据", style="Heading 1")
    p(
        doc,
        "本说明依据当前工作树中的 README.md、config.py、cli/train.py、cli/train_incremental.py、"
        "train/dataloaders.py、train/loops.py、train/full_training_runner.py、models/blocks.py、"
        "utils/excitation_channel_models.py、data/io.py 与 test/Predict_user.py 编制。"
        "如代码后续变更，应以新版本 training_config.json 和实际入口参数为最终依据。",
    )

    # Prevent orphaned headings and set metadata.
    for paragraph in doc.paragraphs:
        if paragraph.style and paragraph.style.name.startswith("Heading"):
            paragraph.paragraph_format.keep_with_next = True
        for run in paragraph.runs:
            if run.font.name is None:
                set_run_font(run)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
