import xlrd
import csv

def csv_from_excel():
    wb = xlrd.open_workbook('data/nsk_multilabel.xlsx')
    sh = wb.sheet_by_name('Sheet1')
    nsk_xlsx = open('data/nsk_decisions.csv', 'w', encoding='utf-8')
    wr = csv.writer(nsk_xlsx, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    nsk_xlsx.close()

csv_from_excel()
