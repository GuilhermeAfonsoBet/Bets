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

import markdown2
from weasyprint import HTML


def main():
    md_path = Path("docs/analise_h3b_estrategia.md")
    pdf_path = Path("docs/analise_h3b_estrategia.pdf")
    
    md_content = md_path.read_text(encoding="utf-8")
    
    # Substitui emojis por texto (fontes padrão não têm emojis)
    emoji_map = {
        "✅": "[OK]",
        "❌": "[X]",
        "⚪": "[~]",
        "📊": "[#]",
        "💰": "[$]",
        "⏱️": "[T]",
        "⚠️": "[!]",
        "💾": "[S]",
        "🔍": "[?]",
        "→": "->",
        "↑": "UP",
        "↓": "DOWN",
    }
    for emoji, replacement in emoji_map.items():
        md_content = md_content.replace(emoji, replacement)
    
    html_content = markdown2.markdown(
        md_content, 
        extras=["tables", "fenced-code-blocks", "header-ids", "break-on-newline"]
    )
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm 1.8cm;
                @bottom-center {{
                    content: "Pagina " counter(page) " de " counter(pages);
                    font-size: 8pt;
                    color: #999;
                }}
            }}
            body {{
                font-family: 'DejaVu Sans', 'Liberation Sans', Arial, Helvetica, sans-serif;
                font-size: 10pt;
                line-height: 1.5;
                color: #333;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }}
            h1 {{
                color: #1a1a2e;
                border-bottom: 3px solid #16213e;
                padding-bottom: 8px;
                font-size: 20pt;
                margin-top: 0;
            }}
            h2 {{
                color: #16213e;
                border-bottom: 1.5px solid #ccc;
                padding-bottom: 4px;
                margin-top: 25px;
                font-size: 14pt;
                page-break-after: avoid;
            }}
            h3 {{
                color: #0f3460;
                margin-top: 18px;
                font-size: 11pt;
                page-break-after: avoid;
            }}
            p {{
                margin: 6px 0;
                text-align: justify;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
                font-size: 8.5pt;
                table-layout: auto;
                word-wrap: break-word;
                page-break-inside: avoid;
            }}
            th {{
                background-color: #16213e;
                color: white;
                padding: 5px 8px;
                text-align: left;
                font-weight: 600;
                font-size: 8pt;
            }}
            td {{
                padding: 4px 8px;
                border: 1px solid #ddd;
                vertical-align: top;
                word-wrap: break-word;
                max-width: 200px;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            code {{
                background-color: #f0f0f0;
                padding: 1px 4px;
                border-radius: 2px;
                font-size: 9pt;
                font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px 12px;
                border-radius: 4px;
                border-left: 3px solid #16213e;
                font-size: 8pt;
                font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
                overflow: hidden;
                page-break-inside: avoid;
            }}
            strong {{
                color: #1a1a2e;
            }}
            hr {{
                border: none;
                border-top: 1.5px solid #ddd;
                margin: 20px 0;
            }}
            ul, ol {{
                margin: 6px 0;
                padding-left: 20px;
            }}
            li {{
                margin: 3px 0;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    HTML(string=full_html).write_pdf(str(pdf_path))
    print(f"PDF gerado: {pdf_path}")


if __name__ == "__main__":
    main()
