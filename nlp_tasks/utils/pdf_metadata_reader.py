# -*- coding: utf-8 -*-


import PyPDF4
import optparse
from PyPDF4 import PdfFileReader


def printMeta(filename):
    pdfFile = PdfFileReader(open(filename, 'rb'))
    docInfo = pdfFile.getDocumentInfo()
    print('[*] PDF MetaData For: {}'.format(filename))
    for metaItem in docInfo:
        print('[+] {0} : {1}'.format(metaItem, docInfo[metaItem]))


def main():
    parser = optparse.OptionParser('usage %prog + -F <PDF file name>')
    parser.add_option('-F', dest='filename', type='string', help='specify PDF file name')
    (options, args) = parser.parse_args()
    fileName = options.filename
    if fileName == None:
        print(parser.usage)
        exit(0)
    else:
        printMeta(fileName)


if __name__ == '__main__':
    main()
