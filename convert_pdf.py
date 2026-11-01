import re
import os
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image, PageBreak
)
from reportlab.lib.units import inch

def convert_markdown_to_pdf(md_path, pdf_path, diagram_path='workflow_diagram.png'):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch
    )

    styles = getSampleStyleSheet()

    primary_color = colors.HexColor('#1E3A8A')
    text_dark = colors.HexColor('#1F2937')
    border_gray = colors.HexColor('#D1D5DB')

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=5
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.8,
        leading=12.2,
        textColor=text_dark,
        spaceAfter=3.5
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.4,
        leading=11.5,
        textColor=text_dark,
        leftIndent=10,
        spaceAfter=2
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=text_dark
    )

    flowables = []
    in_code_block = False
    in_table = False
    table_rows = []

    def format_inline(text):
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" color="#2563EB"><u>\1</u></a>', text)
        return text

    for line in lines:
        stripped = line.strip()

        # Insert page break before Section 3 so Page 1 contains Problem + Diagram, Page 2 contains Datasets + Milestones
        if stripped.startswith('## 3. Dataset'):
            flowables.append(PageBreak())

        if stripped.startswith('`'):
            if in_code_block:
                if os.path.exists(diagram_path):
                    with PILImage.open(diagram_path) as im:
                        orig_w, orig_h = im.size
                    aspect = orig_h / orig_w
                    target_w = 5.6 * inch
                    target_h = target_w * aspect  # ~5.03 inch
                    flowables.append(Image(diagram_path, width=target_w, height=target_h))
                    flowables.append(Spacer(1, 4))
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            continue

        if '|' in stripped and stripped.startswith('|') and stripped.endswith('|'):
            if re.match(r'^\|[\s\-:|]+\|$', stripped):
                continue
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            table_rows.append(cells)
            in_table = True
            continue
        else:
            if in_table and len(table_rows) > 0:
                data = []
                header = [Paragraph(format_inline(c), table_header_style) for c in table_rows[0]]
                data.append(header)
                for row in table_rows[1:]:
                    data.append([Paragraph(format_inline(c), table_cell_style) for c in row])

                col_widths = None
                if len(data[0]) == 3 and 'Week' in table_rows[0][0]:
                    col_widths = [1.1 * inch, 1.8 * inch, 4.5 * inch]
                elif len(data[0]) == 3:
                    col_widths = [1.5 * inch, 1.1 * inch, 4.8 * inch]

                t = Table(data, colWidths=col_widths, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('INNERGRID', (0, 0), (-1, -1), 0.4, border_gray),
                    ('BOX', (0, 0), (-1, -1), 0.6, primary_color),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')])
                ]))
                flowables.append(t)
                flowables.append(Spacer(1, 6))
                table_rows = []
                in_table = False

        if not stripped:
            flowables.append(Spacer(1, 2.5))
            continue

        if stripped == '---':
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=border_gray, spaceBefore=3, spaceAfter=3))
            continue

        if stripped.startswith('# '):
            flowables.append(Paragraph(format_inline(stripped[2:]), title_style))
        elif stripped.startswith('## '):
            flowables.append(Paragraph(format_inline(stripped[3:]), h2_style))
        elif stripped.startswith('- ') or stripped.startswith('* '):
            flowables.append(Paragraph('&bull; ' + format_inline(stripped[2:]), bullet_style))
        else:
            flowables.append(Paragraph(format_inline(stripped), body_style))

    if in_table and len(table_rows) > 0:
        data = [[Paragraph(format_inline(c), table_cell_style) for c in row] for row in table_rows]
        t = Table(data)
        flowables.append(t)

    doc.build(flowables)
    print(f'Successfully built clean 2-page PDF at {pdf_path}')

if __name__ == '__main__':
    convert_markdown_to_pdf('proposal.md', 'proposal.pdf')
