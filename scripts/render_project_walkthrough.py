"""Render the meeting's editable Markdown walkthrough as a readable PDF."""
import argparse
from pathlib import Path
import re
from urllib.parse import quote
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image


def render(source,destination,repo):
    for name,file in [('Body','arial.ttf'),('BodyBold','arialbd.ttf'),('BodyItalic','ariali.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    ink=colors.HexColor('#21303a');blue=colors.HexColor('#216781');muted=colors.HexColor('#536570')
    styles={
        'p':ParagraphStyle('Body',fontName='Body',fontSize=10.4,leading=14.3,textColor=ink,spaceAfter=6),
        'h1':ParagraphStyle('Title',fontName='BodyBold',fontSize=24,leading=29,textColor=ink,spaceAfter=16),
        'h2':ParagraphStyle('Section',fontName='BodyBold',fontSize=18,leading=23,textColor=blue,spaceAfter=12,keepWithNext=True),
        'h3':ParagraphStyle('Subsection',fontName='BodyBold',fontSize=12,leading=16,textColor=blue,spaceBefore=8,spaceAfter=6,keepWithNext=True),
        'list':ParagraphStyle('List',fontName='Body',fontSize=10.4,leading=14.3,textColor=ink,leftIndent=13,firstLineIndent=-10,spaceAfter=5),
        'cell':ParagraphStyle('Cell',fontName='Body',fontSize=9,leading=12,textColor=ink),
        'caption':ParagraphStyle('Caption',fontName='BodyItalic',fontSize=8.4,leading=11,textColor=muted,spaceAfter=7),
    }
    base='https://github.com/jrubow/cs557-project/blob/codex/windows-validation/'
    def inline(text):
        text=escape(text)
        def link(m):
            label,target=m.group(1),m.group(2)
            if not target.startswith(('https://','http://')):
                resolved=(source.parent/target).resolve()
                rel=resolved.relative_to(repo.resolve()).as_posix()
                target=(base.replace('/blob/','/tree/') if resolved.is_dir() else base)+quote(rel,safe='/')
            return '<link href="'+target+'" color="#216781">'+label+'</link>'
        text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
        text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
        text=re.sub(r'`([^`]+)`',r'<font name="BodyBold">\1</font>',text)
        return text
    blocks=[];lines=source.read_text(encoding='utf-8').splitlines();i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line=='<!-- pagebreak -->':blocks.append(PageBreak());continue
        if line.startswith('#'):
            count=len(line)-len(line.lstrip('#'));style=styles[{1:'h1',2:'h2'}.get(count,'h3')]
            blocks.append(Paragraph(inline(line[count:].strip()),style));continue
        if line.startswith('!['):
            m=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
            path=(source.parent/m.group(2)).resolve();w,h=ImageReader(str(path)).getSize()
            width=500; height=width*h/w
            limit=150 if path.name=='reserved-test.png' else (235 if path.name=='behavior-interpretation.png' else 250)
            if height>limit:width*=limit/height;height=limit
            blocks.append(Image(str(path),width=width,height=height,hAlign='CENTER'))
            blocks.append(Spacer(1,6));continue
        if line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip());i+=1
            data=[]
            for row in rows:
                cells=[c.strip() for c in row.strip('|').split('|')]
                if all(re.fullmatch(r'[-: ]+',c) for c in cells):continue
                data.append([Paragraph(inline(c),styles['cell']) for c in cells])
            n=len(data[0]);widths={2:[125,375],3:[155,165,180],4:[197,101,101,101]}[n]
            if 'What earns' in rows[0]:widths=[55,295,150]
            if 'Five-wave wins' in rows[0] and n==3:widths=[250,130,120]
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f3')),
                ('LINEBELOW',(0,0),(-1,0),.6,blue),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d9e0e4')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            blocks.extend([t,Spacer(1,10)]);continue
        if line.startswith('- ') or re.match(r'\d+\. ',line):
            if line.startswith('- '):line='- '+line[2:]
            blocks.append(Paragraph(inline(line),styles['list']));continue
        paragraph=[line]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','<!--','- ')):
            paragraph.append(lines[i].strip());i+=1
        blocks.append(Paragraph(inline(' '.join(paragraph)),styles['p']))
    def decorate(canvas,doc):
        canvas.saveState();canvas.setFillColor(muted);canvas.setFont('Body',8)
        canvas.drawString(48,764,'CS557  /  Minecraft combat learning')
        canvas.drawRightString(564,764,'Collaborator meeting  /  November 25, 2026')
        canvas.setStrokeColor(colors.HexColor('#d9e0e4'));canvas.line(48,754,564,754)
        canvas.drawString(48,28,'Plain-English project walkthrough  |  Results frozen November 22, 2026')
        canvas.drawRightString(564,28,str(doc.page));canvas.restoreState()
    destination.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(destination),pagesize=(612,792),leftMargin=56,rightMargin=56,
        topMargin=52,bottomMargin=48,title='Teaching an agent to fight zombies in Minecraft',author='CS557 project team')
    doc.build(blocks,onFirstPage=decorate,onLaterPages=decorate)
    print(str(destination))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.source.resolve(),a.output.resolve(),Path(__file__).resolve().parents[1])
