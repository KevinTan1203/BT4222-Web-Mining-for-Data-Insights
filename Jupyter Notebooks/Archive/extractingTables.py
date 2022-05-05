import tabula
import os


# Read PDF file
tables = tabula.read_pdf(
    "Apple_Environmental_Progress_Report_2021.pdf", pages="all")
    
for i in range(len(tables) - 1):
    print(tables[i])
    print('\n\n')
