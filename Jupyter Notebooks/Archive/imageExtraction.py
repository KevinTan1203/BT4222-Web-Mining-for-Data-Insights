"""
This file can be used to extract images from a pdf

Have the following installed:
    - PyMuPDF (version 1.16.14) (i.e pip install PyMuPDF==1.16.14)
    - fitz
"""

import fitz

# Source: https://stackoverflow.com/questions/2693820/extract-images-from-pdf-without-resampling-in-python
doc = fitz.open("Apple_Environmental_Progress_Report_2021.pdf")
count = 0
for i in range(len(doc)):
    for img in doc.getPageImageList(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n < 5:       # this is GRAY or RGB
            pix.writePNG("extractedImgs/p%s-%s.png" % (i, xref))
        else:               # CMYK: convert to RGB first
            pix1 = fitz.Pixmap(fitz.csRGB, pix)
            pix1.writePNG("extractedImgs/p%s-%s.png" % (i, xref))
            pix1 = None
        pix = None
        count += 1
print(f'Number of images extracted: {count}')