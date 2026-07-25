from pathlib import Path

from PIL import Image, ImageDraw


root = Path(__file__).resolve().parent / "docx_render"
pages = sorted(root.glob("final-page-*.png"))
for group_index in range(0, len(pages), 7):
    group = pages[group_index : group_index + 7]
    thumbs = []
    for path in group:
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 470))
        thumbs.append((path.name, image.copy()))
    canvas = Image.new("RGB", (3 * 390, 3 * 510), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (name, image) in enumerate(thumbs):
        x = (idx % 3) * 390 + 15
        y = (idx // 3) * 510 + 25
        draw.text((x, 5 + (idx // 3) * 510), name, fill="black")
        canvas.paste(image, (x, y))
    canvas.save(root / f"final-contact-{group_index // 7 + 1}.png")
