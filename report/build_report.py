"""Builds report/ELCVIA_report.pdf following the ELCVIA camera-ready template.

Template rules (ELCVIA_Template.pdf, Tables 1 and 2):
  A4, top/left margin 2 cm, text area 17 cm x 25 cm,
  Times: title 16 bold, section 14 bold, subsection 12 bold,
  abstract 10, body 11, captions 11.

Run:  python report/build_report.py
"""
import os  # import the os module for file paths
import sys  # import sys (not used much here)
from reportlab.lib.pagesizes import A4  # import the A4 page size
from reportlab.lib.units import cm  # import cm, a unit for centimetres
from reportlab.lib.styles import ParagraphStyle  # import ParagraphStyle to define text styles
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT  # import text alignment constants
from reportlab.lib import colors  # import named colours
from reportlab.pdfbase import pdfmetrics  # import pdfmetrics to register fonts
from reportlab.pdfbase.ttfonts import TTFont  # import TTFont to load TrueType fonts
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,  # import the building blocks of the PDF layout
                                Spacer, Table, TableStyle, Image, KeepTogether,  # (more layout classes: spacing, tables, images)
                                Flowable, ListFlowable, ListItem)  # (more layout classes: custom flowables and lists)

HERE = os.path.dirname(os.path.abspath(__file__))  # folder that contains this script
ROOT = os.path.dirname(HERE)  # parent folder = project root
OUT = os.path.join(HERE, "ELCVIA_report.pdf")  # path of the PDF we will create

# ---- fonts (Times New Roman, the "Times" of the template) -----------------
FD = r"C:\Windows\Fonts"  # folder with Windows fonts
pdfmetrics.registerFont(TTFont("T", os.path.join(FD, "times.ttf")))  # register normal Times font as 'T'
pdfmetrics.registerFont(TTFont("T-B", os.path.join(FD, "timesbd.ttf")))  # register bold Times as 'T-B'
pdfmetrics.registerFont(TTFont("T-I", os.path.join(FD, "timesi.ttf")))  # register italic Times as 'T-I'
pdfmetrics.registerFont(TTFont("T-BI", os.path.join(FD, "timesbi.ttf")))  # register bold-italic Times as 'T-BI'
pdfmetrics.registerFontFamily("T", normal="T", bold="T-B", italic="T-I", boldItalic="T-BI")  # tell reportlab which fonts belong to one family

TITLE = "Cross-Manipulation Generalisation in Deepfake Face Detection"  # report title
SHORT = "Tripathi et al."  # short author name for page headers
AUTHORS = "Arpita Tripathi, Arun Kumar, Amit Kumar and Alok Raj"  # list of authors
AFFIL = "Deep Learning Course Project, Sitare University, India"  # affiliation line
RECEIVED = "Received 1 October 2026; revised 20 November 2026; submitted 22 November 2026"  # received/submitted dates line

# ---- styles ----------------------------------------------------------------
body = ParagraphStyle("body", fontName="T", fontSize=11, leading=13.6,  # style for normal body text
                      alignment=TA_JUSTIFY, spaceAfter=6)  # (font alignment and spacing for body text)
body_ni = body  # another name for the same body style
st_title = ParagraphStyle("title", fontName="T-B", fontSize=16, leading=20,  # style for the title
                          alignment=TA_CENTER, spaceAfter=14)  # (centered, space after)
st_auth = ParagraphStyle("auth", fontName="T", fontSize=12, leading=15,  # style for the authors line
                         alignment=TA_CENTER, spaceAfter=10)  # (centered, space after)
st_aff = ParagraphStyle("aff", fontName="T-I", fontSize=10, leading=13,  # style for the affiliation line
                        alignment=TA_CENTER, spaceAfter=10)  # (italic centered)
st_recv = ParagraphStyle("recv", fontName="T", fontSize=10, leading=13,  # style for the received line
                         alignment=TA_CENTER, spaceAfter=8)  # (centered, space after)
st_abs_h = ParagraphStyle("absh", fontName="T-B", fontSize=10, leading=13,  # style for the 'Abstract' heading
                          alignment=TA_CENTER, spaceAfter=4)  # (centered, space after)
st_abs = ParagraphStyle("abs", fontName="T", fontSize=10, leading=12,  # style for abstract paragraphs
                        alignment=TA_JUSTIFY, firstLineIndent=14, leftIndent=14,  # (justified with indents)
                        rightIndent=14, spaceAfter=0)  # (right indent, no space after)
st_kw = ParagraphStyle("kw", fontName="T", fontSize=10, leading=12,  # style for the key words line
                       alignment=TA_JUSTIFY, leftIndent=14, rightIndent=14)  # (justified with indents)
h1 = ParagraphStyle("h1", fontName="T-B", fontSize=14, leading=17,  # style for section headings (level 1)
                    spaceBefore=12, spaceAfter=7, keepWithNext=1)  # (spacing, stay with next paragraph)
h2 = ParagraphStyle("h2", fontName="T-B", fontSize=12, leading=15,  # style for subsection headings (level 2)
                    spaceBefore=8, spaceAfter=5, keepWithNext=1)  # (spacing, stay with next paragraph)
cap = ParagraphStyle("cap", fontName="T", fontSize=11, leading=13,  # style for figure and table captions
                     alignment=TA_CENTER, spaceBefore=4, spaceAfter=10)  # (centered with spacing)
cell = ParagraphStyle("cell", fontName="T", fontSize=10, leading=12, alignment=TA_CENTER)  # style for table cells
cellb = ParagraphStyle("cellb", parent=cell, fontName="T-B")  # bold version of the cell style
cell_l = ParagraphStyle("cell_l", parent=cell, alignment=TA_LEFT)  # left-aligned version of the cell style
bullet = ParagraphStyle("bullet", parent=body, spaceAfter=3)  # style for bullet items
ref = ParagraphStyle("ref", parent=body, leftIndent=22, firstLineIndent=-22,  # style for reference entries
                     alignment=TA_LEFT, spaceAfter=6)  # (hanging indent, left aligned)


class Rule(Flowable):  # a small flowable that draws a horizontal line
    def __init__(self, width=17 * cm, thick=0.6):  # set up the line with width and thickness
        super().__init__()  # call the parent class setup
        self.w, self.t = width, thick  # save width and thickness

    def wrap(self, aw, ah):  # tell reportlab how much space the line needs
        return self.w, 4  # width w and height 4 points

    def draw(self):  # draw the line on the page
        self.canv.setLineWidth(self.t)  # set the line thickness
        self.canv.line(0, 2, self.w, 2)  # draw a line across the width


def P(t, s=body):  # helper to make a paragraph quickly
    return Paragraph(t, s)  # return a Paragraph with the text and style


def bullets(items):  # helper that builds a bullet list
    return ListFlowable([ListItem(P(i, bullet), leftIndent=18) for i in items],  # make one list item per text (each as a paragraph)
                        bulletType="bullet", start="•", leftIndent=18,  # use a bullet symbol and set the indent
                        bulletFontName="T", bulletFontSize=11)  # set the bullet font and size


def numbered(items):  # helper that builds a numbered list
    return ListFlowable([ListItem(P(i, bullet), leftIndent=22) for i in items],  # make one list item per text
                        bulletType="1", leftIndent=22, bulletFontName="T",  # use numbers and set the indent
                        bulletFontSize=11, bulletFormat="%s.")  # set bullet font, size and number format


def grid_table(rows, col_w, bold_diag=False, header=True):  # helper that builds a table from rows of text
    data = []  # list that will hold all table rows
    for r, row in enumerate(rows):  # loop over rows with row number r
        out = []  # list for the cells of this row
        for c, v in enumerate(row):  # loop over cells with column number c
            if r == 0 and header:  # header row: bold cell
                out.append(P(v, cellb))  # add the bold cell
            elif bold_diag and c == r:  # diagonal cell (same method train/test): bold
                out.append(P(v, cellb))  # add the bold cell
            elif c == 0:  # first column: left-aligned cell
                out.append(P(v, cell_l))  # add the left-aligned cell
            else:  # all other cells
                out.append(P(v, cell))  # add a normal centered cell
        data.append(out)  # add the finished row to the data
    t = Table(data, colWidths=col_w)  # create the table with given column widths
    t.setStyle(TableStyle([  # apply the table style
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),  # draw a thin grid on all cells
        ("LINEAFTER", (0, 0), (0, -1), 1.2, colors.black),  # thicker line after the first column
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.black),  # thicker line under the header row
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),  # center cell text vertically
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),  # set top and bottom cell padding
    ]))  # close the style list
    return t  # give back the table


def fig(name, caption, w=11.5 * cm):  # helper that makes a figure with a caption
    _n["fig"] += 1  # count one more figure
    img = Image(os.path.join(ROOT, "figures", name), width=w, height=w * 750 / 1200)  # load the image with the right aspect ratio
    return KeepTogether([img, P(f"Figure {_n['fig']}: {caption}", cap)])  # keep image and caption together on one page


_n = {"fig": 0, "tab": 3}  # counters for figures and tables


def add_fig(s, name, caption, w=12 * cm, intro=None):  # function that adds a figure only if its file exists
    """Append an optional figure (skipped silently if the PNG does not exist yet)."""
    path = os.path.join(ROOT, "figures", name)  # full path of the image file
    if not os.path.exists(path):  # check that the file exists
        return False  # skip silently if missing
    from PIL import Image as PI  # import PIL Image under another name
    iw, ih = PI.open(path).size  # read the image size
    _n["fig"] += 1  # count one more figure
    if intro:  # if an intro sentence is given
        s.append(P(intro.replace("{n}", str(_n["fig"]))))  # add the intro with {n} replaced by the figure number
    s.append(KeepTogether([Image(path, width=w, height=w * ih / iw),  # add image and caption kept together
                           P(f"Figure {_n['fig']}: {caption}", cap)]))  # (caption text with figure number)
    return True  # report that a figure was added


# ---- page decorations ---------------------------------------------------------
class Doc(BaseDocTemplate):  # class that sets up the PDF document and page headers
    last_page = 7  # default last page number

    def __init__(self, fn, last_page):  # set up the document
        super().__init__(fn, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,  # A4 page with 2 cm side margins
                         topMargin=2 * cm, bottomMargin=2 * cm,  # 2 cm top and bottom margins
                         title=TITLE, author="Arpita Tripathi, Arun Kumar, Amit Kumar, Alok Raj")  # set PDF title and author
        self.last = last_page  # remember the last page number
        W, H = A4  # get page width W and height H
        # text area 17 x 25 cm; header sits in the top margin region as in the template
        self.addPageTemplates([  # register the two page templates
            PageTemplate("first", frames=[Frame(2 * cm, 2.6 * cm, 17 * cm, 24.4 * cm - 0.8 * cm,  # first-page template with its text frame
                                                leftPadding=0, rightPadding=0, topPadding=0,  # (frame without padding)
                                                bottomPadding=0, id="f1")],  # (frame id)
                         onPage=self.first_page),  # call first_page when drawing this page
            PageTemplate("later", frames=[Frame(2 * cm, 2.0 * cm, 17 * cm, 25 * cm - 1.4 * cm,  # template for later pages with its frame
                                                leftPadding=0, rightPadding=0, topPadding=0,  # (frame without padding)
                                                bottomPadding=0, id="f2")],  # (frame id)
                         onPage=self.later_page),  # call later_page when drawing these pages
        ])  # close the list of templates

    def first_page(self, c, d):  # draw the header and footnote of the first page
        W, H = A4  # page width and height
        c.saveState()  # save the drawing state
        c.setFont("T-I", 10)  # set italic 10 pt font
        c.drawCentredString(W / 2, H - 1.6 * cm,  # draw the journal line centered at the top
                            f"Electronic Letters on Computer Vision and Image Analysis 0(0):1-{self.last}, 2026")  # (text of the journal line)
        # footnote block
        c.setLineWidth(0.5)  # set line thickness
        c.line(2.2 * cm, 2.45 * cm, 8.2 * cm, 2.45 * cm)  # draw the short line above the footnote
        c.setFont("T", 9)  # set 9 pt font for the footnote
        y = 2.1 * cm  # starting y position of the footnote
        for line in ["Correspondence to: su-24023@sitare.org",  # loop over the footnote lines
                     "",  # (correspondence line)
                     "Recommended for acceptance by Chhavi Sharma and Muskan Katiyar"]:  # (acceptance line)
            c.drawString(2.2 * cm, y, line)  # draw one footnote line
            y -= 0.38 * cm  # move down for the next line
        c.restoreState()  # restore the drawing state
        self._next_later()  # call a placeholder method

    def later_page(self, c, d):  # draw the header of the later pages
        W, H = A4  # page width and height
        n = d.page  # current page number
        c.saveState()  # save the drawing state
        c.setFont("T-I", 10)  # set italic 10 pt font
        txt = f"{SHORT} / Electronic Letters on Computer Vision and Image Analysis 0(0):1-{self.last}, 2026"  # header text with short author and journal
        c.drawCentredString(W / 2, H - 1.6 * cm, txt)  # draw the header text centered
        c.setFont("T", 10)  # set normal 10 pt font
        if n % 2 == 0:  # even page: number on the left
            c.drawString(2 * cm, H - 1.6 * cm, str(n))  # draw the number on the left
        else:  # odd page: number on the right
            c.drawRightString(W - 2 * cm, H - 1.6 * cm, str(n))  # draw the number on the right
        c.restoreState()  # restore the drawing state

    def _next_later(self):  # placeholder method that does nothing
        pass  # do nothing


def story():  # function that builds the list of report content
    s = []  # empty list for the content
    _n["fig"], _n["tab"] = 0, 3  # reset the figure and table counters
    s.append(Spacer(1, 1.1 * cm))  # add a blank space at the top
    s.append(P(TITLE, st_title))  # add the title
    s.append(P(AUTHORS, st_auth))  # add the authors
    s.append(P(AFFIL, st_aff))  # add the affiliation
    s.append(P(RECEIVED, st_recv))  # add the received line
    s.append(Spacer(1, 4))  # add a small space
    s.append(Rule())  # add a horizontal line
    s.append(Spacer(1, 6))  # add a small space
    s.append(P("Abstract", st_abs_h))  # add the Abstract heading
    # Abstract paragraphs: problem, method and results, and augmentation findings.
    abstract = [  # list of abstract paragraphs
        "Deepfake detectors often perform very well on the manipulation method they were trained on, "  # text of the report (continued)
        "but fail on methods they have never seen. This paper measures that gap with a "  # text of the report (continued)
        "leave-one-manipulation-out evaluation on FaceForensics++ (c23).",  # text of the report (continued)
        "We train one EfficientNet-B0 detector per manipulation method (DeepFakes, Face2Face, FaceSwap, "  # text of the report (continued)
        "NeuralTextures) and test each on all four methods and on FaceShifter, a method used for testing only. "  # text of the report (continued)
        "The mean same-method AUC is 0.906, but the mean AUC on unseen methods is only 0.571, close to chance, "  # text of the report (continued)
        "and several models score below 0.5.",  # text of the report (continued)
        "We then test whether blending-based augmentation, with JPEG and blur augmentation, reduces the gap. "  # text of the report (continued)
        "It raises the mean unseen-method AUC to 0.603 (+0.032) but lowers the mean same-method AUC to 0.844 "  # text of the report (continued)
        "(&minus;0.062). All results use a single seed and a training subset, so they are indicative.",  # text of the report (continued)
    ]  # continuation of the statement above
    for a in abstract:  # loop over the abstract paragraphs
        s.append(P(a, st_abs))  # add each paragraph with the abstract style
    s.append(Spacer(1, 8))  # add some space
    # Key words line.
    s.append(P("<i>Key Words</i>: Deepfake Detection, Generalisation, FaceForensics++, Pattern Recognition, "  # add the key words line
               "Data Augmentation, EfficientNet, Image Forensics.", st_kw))  # text of the report (continued)
    s.append(Spacer(1, 8))  # add some space
    s.append(Rule())  # add a horizontal line

    # 1 Introduction
    s.append(P("1 Introduction", h1))  # add the Introduction heading
    # Introduction paragraph.
    s.append(P("Deepfake detectors often perform very well on the manipulation method they were trained on, "  # add a paragraph to the report
               "but fail on methods they have never seen. This project measures that gap using "  # text of the report (continued)
               "leave-one-manipulation-out evaluation on FaceForensics++ (c23), and tests whether "  # text of the report (continued)
               "blending-based augmentation reduces it."))  # text of the report (continued)

    # 2 Background
    s.append(P("2 Background", h1))  # add the Background heading
    # Background paragraph about FaceForensics++.
    s.append(P("FaceForensics++ (R&ouml;ssler et al., ICCV 2019) [1] contains 1,000 original videos and four "  # add a paragraph to the report
               "manipulation methods: DeepFakes, Face2Face, FaceSwap and NeuralTextures. The paper trains an "  # text of the report (continued)
               "XceptionNet detector on all four methods together and reports results on compressed (c23 and c40) videos."))  # text of the report (continued)
    # Background paragraph about the Celeb-DF dataset.
    s.append(P("Celeb-DF (Li et al., CVPR 2020) [2] is a second-generation deepfake dataset with higher visual "  # add a paragraph to the report
               "quality. Its evaluation shows that detectors trained on earlier datasets lose a large amount of "  # text of the report (continued)
               "accuracy when tested on new data. Its Table 4 reports frame-level AUC for XceptionNet (c23) trained "  # text of the report (continued)
               "on FaceForensics++: 99.7% on FF-DF, 85.9% on DFD, 72.2% on DFDC and 65.3% on Celeb-DF."))  # text of the report (continued)
    # Background paragraph explaining how this project differs.
    s.append(P("Those experiments test cross-dataset transfer with a model trained on all four methods. "  # add a paragraph to the report
               "This project tests cross-method transfer, where each model sees only one method during training."))  # text of the report (continued)

    # 3 Data
    s.append(P("3 Data", h1))  # add the Data heading
    # Bullet list describing the data (source, split, frames, subset, test methods, data gap).
    s.append(bullets([  # add an item to the report
        "<b>Source:</b> FaceForensics++ c23, obtained as a public Kaggle copy (ff-faces-224). Used under the "  # text of the report (continued)
        "FaceForensics++ Terms of Use for research only; not redistributed.",  # text of the report (continued)
        "<b>Split:</b> the official FaceForensics++ split: 720 training, 140 validation and 140 test videos. "  # text of the report (continued)
        "Each fake video is in the same split as its source video. No video appears in more than one split.",  # text of the report (continued)
        "<b>Frames:</b> 5 frames per video, face crops from MTCNN with a 1.3&times; margin, resized to 224 &times; 224.",  # text of the report (continued)
        "<b>Training subset:</b> 200 training videos per method, plus the original videos used for each training "  # text of the report (continued)
        "run. Validation and test use all 140 videos.",  # text of the report (continued)
        "<b>Test methods:</b> DeepFakes, Face2Face, FaceSwap, NeuralTextures, and FaceShifter. FaceShifter is used "  # text of the report (continued)
        "only for testing, as an extra unseen method.",  # text of the report (continued)
        "<b>Known data gap:</b> DeepFakes validation has 698 frames instead of 700. Two frames were dropped during "  # text of the report (continued)
        "face detection.",  # text of the report (continued)
    ]))  # continuation of the statement above
    add_fig(s, "pipeline.png", "Preprocessing pipeline from video to 224 &times; 224 face crop.", 15 * cm)  # add the preprocessing pipeline figure
    add_fig(s, "split.png", "Official FaceForensics++ split (720/140/140); fake and source videos share a split.", 11 * cm)  # add the data split figure
    add_fig(s, "sample_faces.png", "Example test crops: original and each manipulation method.", 15 * cm)  # add the sample faces figure
    add_fig(s, "dataset_stats.png", "Number of frames per method and split.", 11 * cm)  # add the dataset statistics figure

    # 4 Method
    s.append(P("4 Method", h1))  # add the Method heading
    # Method paragraph: the model.
    s.append(P("<b>Model:</b> EfficientNet-B0 [3] from timm [4], ImageNet pre-trained, with one output for a "  # add a paragraph to the report
               "real/fake score."))  # text of the report (continued)
    # Method paragraph: training settings.
    s.append(P("<b>Training:</b> AdamW, learning rate 1e-4, weight decay 1e-4, batch size 32, 5 epochs, mixed "  # add a paragraph to the report
               "precision, random horizontal flip. Positive weight set from the class balance. The epoch with the "  # text of the report (continued)
               "highest validation AUC is kept."))  # text of the report (continued)
    # Method paragraph: baseline.
    s.append(P("<b>Baseline:</b> four models, one per training method (DeepFakes, Face2Face, FaceSwap, NeuralTextures)."))  # add a paragraph to the report
    # Method paragraph: augmented models.
    s.append(P("<b>Augmented:</b> the same four models and settings, with blending-based augmentation, JPEG and "  # add a paragraph to the report
               "blur augmentation added during training. The only difference from the baseline is the augmentation flag."))  # text of the report (continued)
    # Method paragraph: test rule and metrics.
    s.append(P("<b>Test rule:</b> for each test method, positives are that method's fake frames and negatives are "  # add a paragraph to the report
               "the original frames from the same test videos. Metrics are frame-level AUC and equal error rate (EER)."))  # text of the report (continued)
    # Method paragraph: random seeds.
    s.append(P("<b>Seed:</b> 42 for the main results" + (", with seeds 43 and 44 for variance (Section 5.5)."  # add the seed paragraph, mentioning extra seeds only if their figure exists
              if os.path.exists(os.path.join(ROOT, "figures", "seed_variance.png")) else ".")))  # (the else part gives a plain full stop)
    add_fig(s, "experiment.png", "Experiment design: one model per training method, tested on five methods.", 15 * cm)  # add the experiment design figure
    add_fig(s, "augmentation_examples.png", "Training augmentations applied to real crops: blended pseudo-fake, blur and JPEG.", 11 * cm)  # add the augmentation examples figure

    # 5 Results
    s.append(P("5 Results", h1))  # add the Results heading
    cols = ["Trained on", "DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]  # table header names
    cw = [3.0 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 3.3 * cm, 2.3 * cm]  # column widths of the results tables
    base = [cols,  # baseline AUC rows (header first)
            ["DeepFakes", "0.943", "0.607", "0.398", "0.728", "0.570"],  # one table row
            ["Face2Face", "0.728", "0.901", "0.562", "0.671", "0.436"],  # one table row
            ["FaceSwap", "0.516", "0.576", "0.926", "0.468", "0.475"],  # one table row
            ["NeuralTextures", "0.738", "0.635", "0.445", "0.854", "0.575"]]  # one table row
    aug = [cols,  # augmented AUC rows (header first)
           ["DeepFakes", "0.878", "0.631", "0.549", "0.661", "0.568"],  # one table row
           ["Face2Face", "0.703", "0.838", "0.608", "0.630", "0.493"],  # one table row
           ["FaceSwap", "0.605", "0.600", "0.866", "0.555", "0.570"],  # one table row
           ["NeuralTextures", "0.716", "0.674", "0.553", "0.794", "0.536"]]  # one table row

    s.append(P("5.1 Baseline matrix (AUC)", h2))  # add the 5.1 subsection heading
    s.append(KeepTogether([  # keep table and caption together
        grid_table(base, cw, bold_diag=True),  # build the baseline table
        # Caption of Table 1.
        P("Table 1: Baseline AUC. Rows are the training method, columns the test method. "  # continuation of the statement above
          "Bold cells are the same method for training and testing.", cap)]))  # text of the report (continued)
    s.append(P("5.2 Augmented matrix (AUC)", h2))  # add the 5.2 subsection heading
    s.append(KeepTogether([  # keep table and caption together
        grid_table(aug, cw, bold_diag=True),  # build the augmented table
        P("Table 2: AUC with blending, JPEG and blur augmentation. Same layout as Table 1.", cap)]))  # continuation of the statement above

    s.append(P("5.3 Summary", h2))  # add the 5.3 subsection heading
    summ = [["Measure", "Baseline", "Augmented", "Change"],  # rows of the summary table
            ["Mean AUC, same method (diagonal)", "0.906", "0.844", "&minus;0.062"],  # one table row
            ["Mean AUC, unseen methods (off-diagonal, incl. FaceShifter)", "0.571", "0.603", "+0.032"]]  # one table row
    s.append(KeepTogether([  # keep table and caption together
        grid_table(summ, [8.2 * cm, 2.9 * cm, 2.9 * cm, 3.0 * cm]),  # build the summary table
        P("Table 3: Summary of mean AUC for baseline and augmented models.", cap)]))  # continuation of the statement above
    # Short paragraph introducing the heat maps.
    s.append(P("The per-cell AUC matrices are shown below as heat maps, followed by the per-cell "  # add a paragraph to the report
               "difference between augmented and baseline."))  # text of the report (continued)
    s.append(fig("baseline_auc.png", "Baseline AUC matrix."))  # add the baseline AUC heat map
    s.append(fig("augmented_auc.png", "Augmented AUC matrix."))  # add the augmented AUC heat map
    s.append(fig("auc_difference.png", "AUC difference (augmented minus baseline) for each cell."))  # add the AUC difference heat map

    s.append(P("5.4 Largest changes", h2))  # add the 5.4 subsection heading
    s.append(bullets([  # add a bullet list of largest changes
        # Bullet list of the biggest gains and losses.
        "<b>Gains:</b> FaceSwap-trained model on DeepFakes (0.516 &rarr; 0.605) and on FaceShifter "  # text of the report (continued)
        "(0.475 &rarr; 0.570); DeepFakes-trained model on FaceSwap (0.398 &rarr; 0.549).",  # text of the report (continued)
        "<b>Losses:</b> DeepFakes-trained model on NeuralTextures (0.728 &rarr; 0.661); NeuralTextures-trained "  # text of the report (continued)
        "model on FaceShifter (0.575 &rarr; 0.536).",  # text of the report (continued)
    ]))  # continuation of the statement above

    # 5.5 optional extra analyses (only the parts whose files exist)
    extra = [("seen_vs_unseen.png", "Mean AUC on the training method and on unseen methods.", 9 * cm),  # list of optional figures: file, caption, width
             ("roc_curves.png", "ROC curves of baseline models trained on FaceSwap and DeepFakes (AUC in legend).", 15 * cm),  # one figure entry: file, caption, width
             ("training_curves.png", "Validation AUC per epoch for each training method.", 15 * cm),  # one figure entry: file, caption, width
             ("seed_variance.png", "Mean AUC over seeds (error bars: standard deviation).", 9 * cm),  # one figure entry: file, caption, width
             ("gradcam.png", "Grad-CAM heat maps; rows are models, columns are test methods.", 15 * cm)]  # one figure entry: file, caption, width
    if any(os.path.exists(os.path.join(ROOT, "figures", e[0])) for e in extra):  # only continue if at least one of these figures exists
        s.append(P("5.5 Additional analyses", h2))  # add the 5.5 subsection heading
        for name, c, w in extra:  # loop over the optional figures
            add_fig(s, name, c, w)  # add each figure if its file exists
        eer_csv = os.path.join(ROOT, "results", "eer_table.csv")  # path of the EER table file
        if os.path.exists(eer_csv):  # only continue if the file exists
            import csv  # import the csv module
            rows = list(csv.reader(open(eer_csv)))  # read all rows of the csv file
            hdr = ["Trained on"] + rows[0][2:]  # table header: training method plus test methods
            for tag in ("baseline_eer", "augmented_eer"):  # loop over baseline and augmented tags
                body_rows = [[r[1]] + r[2:] for r in rows[1:] if r[0] == tag]  # rows of this tag only, without the tag column
                s.append(KeepTogether([grid_table([hdr] + body_rows, cw, bold_diag=True),  # add the EER table with its caption
                                       P(f"Table {_n['tab'] + (1 if tag.startswith('base') else 2)}: {tag.split('_')[0].capitalize()} equal error rate (EER, lower is better).", cap)]))  # (caption with table number and tag name)

    # 5.6 statistical + error analysis (from report/analysis.py)
    s.append(P("5.6 Statistical and error analysis", h2))  # add the 5.6 subsection heading
    # Paragraph explaining the bootstrap method.
    s.append(P("To check which changes are larger than test-sample noise, we computed a 95% bootstrap confidence "  # add a paragraph to the report
               "interval for the AUC change of every cell. Videos, not frames, were resampled (1,000 repeats, "  # text of the report (continued)
               "positives and negatives separately, same resample for both models), because frames of one video are "  # text of the report (continued)
               "correlated. The interval covers test-set sampling only, not training randomness, which a single seed "  # text of the report (continued)
               "cannot measure."))  # text of the report (continued)
    # Paragraph with the bootstrap results and Wilcoxon test.
    s.append(P("Of the 20 cells, 15 changes are outside their 95% interval. All four same-method drops are significant "  # add a paragraph to the report
               "(&minus;0.060 to &minus;0.065). Among the 16 unseen-method cells, 8 improved significantly and 3 "  # text of the report (continued)
               "dropped significantly (DeepFakes&rarr;NeuralTextures, Face2Face&rarr;NeuralTextures, "  # text of the report (continued)
               "NeuralTextures&rarr;FaceShifter); the remaining 5 are not distinguishable from zero. The largest "  # text of the report (continued)
               "gain is DeepFakes&rarr;FaceSwap (+0.150, interval 0.108 to 0.191). A Wilcoxon signed-rank test on the "  # text of the report (continued)
               "16 unseen cells gives p = 0.079, so the average gain of 0.032 is suggestive but not significant at the "  # text of the report (continued)
               "5% level; the gains are concentrated in the FaceSwap-related cells."))  # text of the report (continued)
    add_fig(s, "diff_ci.png", "AUC change per cell (augmented minus baseline) with 95% video-bootstrap intervals.", 10 * cm)  # add the bootstrap interval figure
    # Paragraph with the error analysis at threshold 0.5.
    s.append(P("<b>Error analysis at threshold 0.5.</b> The FaceSwap-trained baseline labels only 11.6% of real test "  # add a paragraph to the report
               "frames as fake, but misses 83% to 90% of the fake frames of DeepFakes, NeuralTextures and FaceShifter, "  # text of the report (continued)
               "so its accuracy on them is about 50%. With augmentation it misses fewer fakes (52% to 60%), but it also "  # text of the report (continued)
               "flags 31.7% of real frames as fake. The same trade-off holds for all four models: the share of real frames "  # text of the report (continued)
               "flagged as fake rises from 11.6% to 23.0% (baseline) to 28.7% to 43.6% (augmented). The blended pseudo-fakes "  # text of the report (continued)
               "therefore make the detector more suspicious of real faces, which explains part of the lost "  # text of the report (continued)
               "same-method accuracy. Frame-level examples of the most confident errors are listed in "  # text of the report (continued)
               "<font face=\"T-I\">results/top_errors.csv</font>."))  # text of the report (continued)

    # 6 Discussion
    s.append(P("6 Discussion", h1))  # add the Discussion heading
    # Discussion paragraph: transfer is weaker than same-method results.
    s.append(P("<b>Cross-method transfer is much weaker than same-method performance.</b> The average same-method "  # add a paragraph to the report
               "AUC is 0.906, but the average on unseen methods is 0.571, close to chance. A detector that looks "  # text of the report (continued)
               "strong on its own training method can rank unseen manipulations poorly."))  # text of the report (continued)
    # Discussion paragraph: models below chance.
    s.append(P("<b>Some models score below chance on unseen methods.</b> The FaceSwap-trained baseline scores 0.516 "  # add a paragraph to the report
               "on DeepFakes, 0.468 on NeuralTextures and 0.475 on FaceShifter. An AUC below 0.5 means the model "  # text of the report (continued)
               "ranks fake frames as more real than real frames. This suggests the model learned artefacts specific "  # text of the report (continued)
               "to FaceSwap, which are absent or reversed in other methods. We report these values as measured."))  # text of the report (continued)
    # Discussion paragraph: augmentation trade-off.
    s.append(P("<b>Augmentation improves transfer on average but reduces same-method accuracy.</b> Mean "  # add a paragraph to the report
               "unseen-method AUC rises by 0.032, while mean same-method AUC falls by 0.062. The largest gains are on "  # text of the report (continued)
               "the FaceSwap-trained and DeepFakes-trained rows, where the baseline was weakest. The gains are not "  # text of the report (continued)
               "uniform: several cells dropped, including NeuralTextures-trained on FaceShifter. The confidence intervals in "  # text of the report (continued)
               "Section 5.6 show that 8 of the 16 unseen-method gains and all four same-method losses are larger than "  # text of the report (continued)
               "test-sample noise, but the overall average gain is not significant (p = 0.079)."))  # text of the report (continued)
    # Discussion paragraph: comparison with the literature.
    s.append(P("<b>Comparison with the literature.</b> The Celeb-DF paper [2] reports that XceptionNet trained on "  # add a paragraph to the report
               "FaceForensics++ loses accuracy across datasets (99.7% on FF-DF down to 65.3% on Celeb-DF). Our results "  # text of the report (continued)
               "show a similar pattern across manipulation methods, but they are not directly comparable: our models "  # text of the report (continued)
               "are smaller, trained on a subset, use one method each, and are evaluated with a different test rule."))  # text of the report (continued)

    # 7 Limitations
    s.append(P("7 Limitations", h1))  # add the Limitations heading
    s.append(numbered([  # add a numbered list of limitations
        # Numbered list of the limitations.
        "<b>Single seed.</b> All results use seed 42. The differences between baseline and augmented cells are often "  # text of the report (continued)
        "0.01 to 0.03, which may be within run-to-run variation. Only the large changes (such as the FaceSwap rows) "  # text of the report (continued)
        "should be treated as likely real. Seeds 43 and 44 would be needed to estimate variance. The bootstrap intervals in "  # text of the report (continued)
        "Section 5.6 cover test-set sampling only, not training randomness.",  # text of the report (continued)
        "<b>Training subset.</b> Each model used 200 training videos per method, not the full training set. Absolute "  # text of the report (continued)
        "scores are therefore likely lower than with full data.",  # text of the report (continued)
        "<b>Five frames per video.</b> Frame sampling is coarse, and video-level results may differ.",  # text of the report (continued)
        "<b>Split independence.</b> The split is video-disjoint, but the same person may appear in more than one "  # text of the report (continued)
        "video, so identity overlap across splits is possible.",  # text of the report (continued)
        "<b>Data gap.</b> DeepFakes validation has 698 frames instead of 700. This affects model selection only slightly.",  # text of the report (continued)
        "<b>Third-party data copy.</b> The dataset was obtained from a public Kaggle upload, not directly from the "  # text of the report (continued)
        "FaceForensics++ authors.",  # text of the report (continued)
        "<b>Paper comparison.</b> Table 4 of the Celeb-DF paper tests cross-dataset transfer with models trained on "  # text of the report (continued)
        "all four methods, so it cannot be used to check our numbers directly.",  # text of the report (continued)
    ]))  # continuation of the statement above

    # 8 Future work / 9 Ethics / 10 Conclusion
    s.append(P("8 Future Work", h1))  # add the Future Work heading
    s.append(bullets([  # add a bullet list of future work
        "Repeat all runs with several seeds and the full training set to measure training variance.",  # text of the report (continued)
        "Test a transformer-based detector (ViT) and compare its transfer with EfficientNet-B0.",  # text of the report (continued)
        "Evaluate on Celeb-DF v2, to measure cross-dataset transfer for models trained on one method.",  # text of the report (continued)
        "Run the stricter leave-one-manipulation-out setting: train on three methods and test on the fourth.",  # text of the report (continued)
        "Tune the share of blended pseudo-fakes, since it also raises false alarms on real faces.",  # text of the report (continued)
    ]))  # continuation of the statement above
    s.append(P("9 Ethical Considerations", h1))  # add the Ethics heading
    # Ethics paragraph.
    s.append(P("FaceForensics++ was used under its Terms of Use for research only. The processed face crops are kept in "  # add a paragraph to the report
               "a private dataset and are not redistributed; only a few example crops appear in this report. "  # text of the report (continued)
               "Deepfake detectors that fail on unseen manipulations can give a false sense of security, so the "  # text of the report (continued)
               "results here should not be read as evidence that any detector is safe to rely on in practice."))  # text of the report (continued)

    # 8 Conclusion
    s.append(P("10 Conclusion", h1))  # add the Conclusion heading
    # Conclusion paragraph.
    s.append(P("Detectors trained on a single FaceForensics++ manipulation method generalise poorly to other "  # add a paragraph to the report
               "methods. The average AUC drops from 0.906 on the training method to 0.571 on unseen methods, and "  # text of the report (continued)
               "some models fall below chance. Blending-based augmentation with JPEG and blur augmentation increases "  # text of the report (continued)
               "average unseen-method AUC to 0.603, at the cost of lower same-method accuracy. With one seed and a "  # text of the report (continued)
               "training subset, these results are indicative. Multiple seeds and the full training set are the next steps."))  # text of the report (continued)

    s.append(P("References", ParagraphStyle("rh", parent=h1)))  # add the References heading
    refs = [  # list of reference entries
        # Reference entries 1 to 4 (authors, title, venue, year).
        "[1] A. R&ouml;ssler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Nie&szlig;ner, "  # text of the report (continued)
        "\"FaceForensics++: Learning to detect manipulated facial images\", <i>Proc. of IEEE/CVF Intl. Conf. on "  # text of the report (continued)
        "Computer Vision (ICCV)</i>, 2019. arXiv:1901.08971.",  # text of the report (continued)
        "[2] Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu, \"Celeb-DF: A large-scale challenging dataset for DeepFake "  # text of the report (continued)
        "forensics\", <i>Proc. of IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)</i>, 2020. "  # text of the report (continued)
        "arXiv:1909.12962.",  # text of the report (continued)
        "[3] M. Tan, Q. Le, \"EfficientNet: Rethinking model scaling for convolutional neural networks\", "  # text of the report (continued)
        "<i>Proc. of Intl. Conf. on Machine Learning (ICML)</i>, 2019.",  # text of the report (continued)
        "[4] R. Wightman, <i>PyTorch Image Models (timm)</i>, "  # text of the report (continued)
        "https://github.com/huggingface/pytorch-image-models.",  # text of the report (continued)
    ]  # continuation of the statement above
    for r in refs:  # loop over the references
        s.append(P(r, ref))  # add each reference with the reference style
    return s  # give back the finished content list


def build(last):  # function that builds the PDF and returns the page count
    doc = Doc(OUT, last)  # create the document
    st = story()  # get the report content
    # after the first page switch to the "later" template
    from reportlab.platypus import NextPageTemplate  # import NextPageTemplate
    st.insert(0, NextPageTemplate("later"))  # switch to the later template after page 1
    doc.build(st)  # build the PDF file
    return doc.page  # return the number of pages


if __name__ == "__main__":  # run only when this script is started directly
    n = build(7)  # first build with a guess of 7 pages
    n = build(n)  # second build with the real page count
    print("pages:", n, "->", OUT)  # print the page count and output path
