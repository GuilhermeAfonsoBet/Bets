#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera PDF do relatório de análise H3B.

Uso:
    pip install markdown2 weasyprint
    python generate_pdf_report.py
"""

import sys
from pathlib import Path


def generate_with_markdown2():
    """Gera PDF usando markdown2 + weasyprint."""
    try:
        import markdown2
        from weasyprint import HTML
    except ImportError:
        return False
    
    md_path = Path("docs/analise_h3b_estrategia.md")
    pdf_path = Path("docs/analise_h3b_estrategia.pdf")
    
    md_content = md_path.read_text(encoding="utf-8")
    
    html_content = markdown2.markdown(
        md_content, 
        extras=["tables", "fenced-code-blocks", "header-ids"]
    )
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{
                color: #1a1a2e;
                border-bottom: 3px solid #16213e;
                padding-bottom: 10px;
                font-size: 22pt;
            }}
            h2 {{
                color: #16213e;
                border-bottom: 1px solid #ccc;
                padding-bottom: 5px;
                margin-top: 30px;
                font-size: 16pt;
            }}
            h3 {{
                color: #0f3460;
                margin-top: 20px;
                font-size: 13pt;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                font-size: 10pt;
            }}
            th {{
                background-color: #16213e;
                color: white;
                padding: 8px 12px;
                text-align: left;
                font-weight: 600;
            }}
            td {{
                padding: 6px 12px;
                border: 1px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10pt;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #16213e;
                font-size: 9pt;
                overflow-x: auto;
            }}
            strong {{
                color: #1a1a2e;
            }}
            blockquote {{
                border-left: 4px solid #e94560;
                margin: 15px 0;
                padding: 10px 20px;
                background-color: #fff5f5;
            }}
            hr {{
                border: none;
                border-top: 2px solid #eee;
                margin: 30px 0;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    HTML(string=full_html).write_pdf(str(pdf_path))
    return str(pdf_path)


def generate_with_fpdf():
    """Gera PDF usando fpdf2 (fallback mais leve)."""
    try:
        from fpdf import FPDF
    except ImportError:
        return False
    
    md_path = Path("docs/analise_h3b_estrategia.md")
    pdf_path = Path("docs/analise_h3b_estrategia.pdf")
    
    content = md_path.read_text(encoding="utf-8")
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Tenta adicionar fonte com suporte a unicode
    try:
        pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        pdf.add_font("DejaVu", "B", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        font_name = "DejaVu"
    except:
        font_name = "Helvetica"
    
    in_code_block = False
    
    for line in content.split('\n'):
        stripped = line.strip()
        
        # Toggle code blocks
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        
        if in_code_block:
            pdf.set_font(font_name, '', 8)
            pdf.cell(0, 5, line.rstrip(), new_x="LMARGIN", new_y="NEXT")
            continue
        
        if stripped.startswith('# ') and not stripped.startswith('## '):
            pdf.set_font(font_name, 'B', 18)
            pdf.cell(0, 12, stripped[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
        elif stripped.startswith('## '):
            pdf.ln(5)
            pdf.set_font(font_name, 'B', 14)
            pdf.cell(0, 10, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
        elif stripped.startswith('### '):
            pdf.ln(3)
            pdf.set_font(font_name, 'B', 12)
            pdf.cell(0, 8, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif stripped.startswith('| ') and '|' in stripped[1:]:
            # Table row
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            if all(c.replace('-', '').replace(':', '') == '' for c in cells):
                continue  # Skip separator row
            num_cols = max(len(cells), 1)
            col_width = min(190 / num_cols, 95)
            for cell in cells:
                is_header = cell.startswith('**')
                if is_header:
                    pdf.set_font(font_name, 'B', 7)
                    cell = cell.strip('*')
                else:
                    pdf.set_font(font_name, '', 7)
                pdf.cell(col_width, 5, cell[:40], border=1)
            pdf.ln()
        elif stripped.startswith('---'):
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
        elif stripped == '':
            pdf.ln(3)
        else:
            pdf.set_font(font_name, '', 10)
            clean = stripped.replace('**', '').replace('*', '')
            if len(clean) > 0:
                pdf.set_x(10)  # Reset x position
                try:
                    pdf.multi_cell(0, 6, clean)
                except Exception:
                    pdf.cell(0, 6, clean[:80], new_x="LMARGIN", new_y="NEXT")
    
    pdf.output(str(pdf_path))
    return str(pdf_path)


if __name__ == "__main__":
    print("Gerando PDF do relatório H3B...")
    
    # Tenta weasyprint primeiro (melhor qualidade)
    result = generate_with_markdown2()
    if result:
        print(f"✅ PDF gerado com weasyprint: {result}")
        sys.exit(0)
    
    print("weasyprint não disponível, tentando fpdf2...")
    
    # Fallback: fpdf2
    result = generate_with_fpdf()
    if result:
        print(f"✅ PDF gerado com fpdf2: {result}")
        sys.exit(0)
    
    print("❌ Nenhuma lib de PDF disponível.")
    print("   Instale uma delas:")
    print("   pip install fpdf2")
    print("   pip install markdown2 weasyprint")
    print("")
    print("   Ou use o arquivo Markdown diretamente:")
    print("   docs/analise_h3b_estrategia.md")
