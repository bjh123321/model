from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "surrogate_model_模型运行逻辑简明说明.docx"

FONT = "Microsoft YaHei"
MONO = "Consolas"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
NAVY = "17365D"
GRAY = "6B7280"
INK = "202124"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
PALE_BLUE = "F4F7FB"


def set_font(run, *, size=None, bold=None, color=None, name=FONT):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    for attr in ("ascii", "hAnsi", "eastAsia"):
        rpr.rFonts.set(qn(f"w:{attr}"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def setup_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    for name, size, color, before, after in (
        ("Title", 22, NAVY, 0, 6),
        ("Subtitle", 12, GRAY, 0, 14),
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 14, 7),
        ("Heading 3", 12, DARK_BLUE, 10, 5),
    ):
        style = doc.styles[name]
        style.font.name = FONT
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = name != "Subtitle"
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    for name in ("List Bullet", "List Number"):
        style = doc.styles[name]
        style.font.name = FONT
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.font.size = Pt(10.5)
        style.paragraph_format.left_indent = Inches(0.375)
        style.paragraph_format.first_line_indent = Inches(-0.188)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.25


def setup_page(doc):
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.left_margin = Inches(1)
    sec.right_margin = Inches(1)
    sec.header_distance = Inches(0.492)
    sec.footer_distance = Inches(0.492)
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(header.add_run("SURROGATE MODEL · QUICK GUIDE"), size=8.5, bold=True, color=GRAY)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(footer.add_run("第 "), size=9, color=GRAY)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    set_font(footer.add_run(" 页"), size=9, color=GRAY)


def add_p(doc, text="", style=None):
    para = doc.add_paragraph(style=style)
    set_font(para.add_run(text))
    return para


def bullet(doc, text):
    return add_p(doc, text, "List Bullet")


def numbered(doc, text):
    return add_p(doc, text, "List Number")


def set_cell_margins(cell, top=90, start=120, bottom=90, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
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


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def geometry(table, widths):
    table.autofit = False
    total = sum(widths)
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = OxmlElement("w:tblInd")
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    tbl_pr.append(tbl_ind)
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
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
            tc_w.set(qn("w:w"), str(widths[idx]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)


def table(doc, headers, rows, widths, *, keep_together=False):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    geometry(t, widths)
    header_row = t.rows[0]
    tr_pr = header_row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "1")
    tr_pr.append(tbl_header)
    for i, text in enumerate(headers):
        cell = header_row.cells[i]
        shade(cell, LIGHT_BLUE)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(0)
        set_font(para.add_run(text), size=9.5, bold=True, color=NAVY)
    for values in rows:
        row = t.add_row()
        for i, text in enumerate(values):
            cell = row.cells[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            set_font(
                para.add_run(str(text)),
                size=9.4,
                bold=(i == 0),
                color=(DARK_BLUE if i == 0 else INK),
            )
    if keep_together:
        for row in t.rows[:-1]:
            for cell in row.cells:
                p_pr = cell.paragraphs[0]._p.get_or_add_pPr()
                if p_pr.find(qn("w:keepNext")) is None:
                    p_pr.append(OxmlElement("w:keepNext"))
    add_p(doc, "")
    return t


def code(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Inches(0.16)
    para.paragraph_format.right_indent = Inches(0.16)
    para.paragraph_format.space_before = Pt(3)
    para.paragraph_format.space_after = Pt(7)
    para.paragraph_format.line_spacing = 1.05
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F6F8FA")
    para._p.get_or_add_pPr().append(shd)
    set_font(para.add_run(text), size=8.5, color="24292F", name=MONO)


def callout(doc, label, text):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    geometry(t, [9360])
    cell = t.cell(0, 0)
    shade(cell, PALE_BLUE)
    para = cell.paragraphs[0]
    para.paragraph_format.space_after = Pt(0)
    set_font(para.add_run(f"{label}  "), bold=True, color=BLUE)
    set_font(para.add_run(text))
    add_p(doc, "")


def build():
    doc = Document()
    setup_styles(doc)
    setup_page(doc)
    doc.core_properties.title = "surrogate_model 模型运行逻辑简明说明"
    doc.core_properties.author = "surrogate_model 项目组"

    title = add_p(doc, "surrogate_model 模型运行逻辑简明说明", "Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = add_p(
        doc,
        f"当前分支：user_for_win7_fixed_exc_channel_fdr  |  更新：{date.today().isoformat()}",
        "Subtitle",
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    callout(
        doc,
        "核心逻辑",
        "输入高度 H、马赫数 M、激励 ID 和通道 ID；四个独立网络分别预测频率、阻尼、"
        "残差幅值和残差相位，再组合成极点/残差并计算最终响应。",
    )

    add_p(doc, "1. 模型做什么", "Heading 1")
    add_p(
        doc,
        "每个“激励 × 响应通道”组合单独建模，每个工作点最多保留 3 个模态槽。"
        "一个组合下包含 4 个 FDR 分量网络：",
    )
    table(
        doc,
        ["网络", "预测量", "用途"],
        [
            ("frequency", "频率 freq", "确定模态频率"),
            ("damping", "阻尼 damping", "确定模态衰减"),
            ("residue_magnitude", "残差幅值 magnitude", "确定响应强度"),
            ("residue_phase", "sinθ、cosθ", "重建复残差相位"),
        ],
        [2700, 2700, 3960],
    )
    add_p(doc, "模型运行链路", "Heading 2")
    numbered(doc, "读取 H、M、激励 ID、通道 ID 和模态数据。")
    numbered(doc, "对 H/M 与各分量目标做训练集归一化。")
    numbered(doc, "四个网络独立预测频率、阻尼、幅值和相位。")
    numbered(doc, "反归一化并重建极点实部/虚部及复残差。")
    numbered(doc, "将极点和残差送入既有响应计算流程，生成 PR、HF/HG/HA、VF/VG/VA 等输出。")

    add_p(doc, "2. 模型结构与训练", "Heading 1")
    add_p(
        doc,
        "每个分量网络采用两阶段多保真结构：low 学习仿真规律，lin 和 adaptation 学习实测修正。",
    )
    code(
        doc,
        "Stage 1: low = MLP(H, M, excitation_embedding, channel_embedding)\n"
        "Stage 2: high = Linear(low) + Adaptation(H, M, embeddings, low)",
    )
    table(
        doc,
        ["阶段", "使用数据", "训练内容", "默认轮数"],
        [
            ("Stage 1", "仿真数据", "只训练 low 低保真头", "频率/残差 500；阻尼 200"),
            ("Stage 2", "实测 + 仿真", "训练 lin、adaptation，默认同时更新 low", "频率/残差 500；阻尼 200"),
        ],
        [1700, 2000, 3300, 2360],
    )
    bullet(doc, "优化器：AdamW；两阶段默认学习率均为 1e-3，weight_decay=5e-5。")
    bullet(doc, "Stage 2 损失：1.0 × 实测损失 + 0.2 × 仿真损失。")
    bullet(doc, "默认按训练损失选择检查点；连续 50 轮无有效改进则早停。")
    bullet(doc, "默认 batch_sim=16、batch_real=8；梯度裁剪 max_norm=5。")

    add_p(doc, "3. 基础训练如何运行", "Heading 1")
    bullet(doc, "从 cli/user_input.dat 读取 paths.ini 和模型版本名。")
    bullet(doc, "加载 cli/sim_dataset 与 cli/real_dataset。")
    bullet(doc, "扫描数据中实际存在的激励/通道组合，并为每个组合建立 4 个分量任务。")
    bullet(doc, "按 --parallel_workers 并行训练独立任务；某个组合失败不影响其他组合。")
    bullet(doc, "完整组合写入 models_manifest.json，失败信息写入 training_failures.json。")
    code(
        doc,
        "python -m surrogate_model.cli.train\n"
        "python -m surrogate_model.cli.train --parallel_workers 4 --num_workers 0\n"
        "python -m surrogate_model.cli.train --resume --parallel_workers 4",
    )
    add_p(doc, "按需训练", "Heading 2")
    code(
        doc,
        "python -m surrogate_model.cli.train --component damping\n"
        "python -m surrogate_model.cli.train --excitation FXD_DC --channel LVT1\n"
        "python -m surrogate_model.cli.train --height 8.23",
    )
    callout(
        doc,
        "恢复规则",
        "--resume 会保留产物完整的分量；不完整分量从第 1 轮重新训练，因为优化器和调度器状态没有保存。",
    )

    add_p(doc, "4. 修正流程如何运行", "Heading 1")
    add_p(
        doc,
        "修正流程不是在父模型上继续微调，而是生成一个新版本，并对受影响的网络完整重训。",
    )
    bullet(doc, "user_input.dat 指定父版本、新版本和修正数据文件。")
    bullet(doc, "先复制父版本全部模型，未受影响的组合直接继承。")
    bullet(doc, "将父版本历史数据与修正数据按工作点合并；修正点进入训练集。")
    bullet(doc, "只对修正数据涉及且选择器命中的激励/通道/分量重新训练。")
    bullet(doc, "一个组合的选中分量全部成功后才替换正式目录；失败时保留父模型。")
    code(
        doc,
        "python -m surrogate_model.cli.train_incremental --component damping\n"
        "python -m surrogate_model.cli.train_incremental "
        "--excitation FXD_DC --parallel_workers 4 --num_workers 0",
    )
    callout(
        doc,
        "无验证点时",
        "修正训练按实测训练损失选模型；已有多保真父模型时，达到父模型损失后默认再训练 50 轮。",
    )

    add_p(doc, "5. 预测与模型产物", "Heading 1")
    prediction_intro = add_p(
        doc,
        "预测端从 models_manifest.json 找到对应激励/通道，加载四个检查点和各自归一化器。"
        "有实测数据的模型使用 high 输出；只有仿真数据的模型使用 low 输出。",
    )
    bullet(doc, "每个分量目录保存 model.pth、training_config.json、train_log.csv 和 Sim/Real 归一化器。")
    bullet(doc, "模型版本根目录保存 models_manifest.json、training_failures.json 和数据快照。")
    bullet(doc, "默认 extrapolation_policy=allow，直接使用网络外推。")
    bullet(doc, "可显式使用 smooth_linear，对训练马赫数范围外的阻尼做平滑线性外推。")
    bullet(doc, "预测阻尼默认下限为 2；扫描输出只清空对应阻尼格，PR 响应链路会拒绝整个越界模态槽。")

    bullet(doc, "两个入口：首次建模用 cli.train；新增或替换少量实测工作点时用 cli.train_incremental。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
