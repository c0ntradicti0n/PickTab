picktab
-------

It extracts first the tabular layout and goes on to the OCR reading.

Based on a script from [pdftabextract](https://github.com/WZBSocialScienceCenter/pdftabextract), Markus Konrad

The reason for this is, that tesseract makes a lot of mistakes, when processing layouted tables, e.g. ignoring some columns or cells.
One can either change the cell borders to readable chars or take subimages. I tried the latter.

Therefore this script uses the layout recognition of pdftabextract, and creates little images for the cells to do OpticalCharacterRecognition.

References
----------
[pdftabextract](https://github.com/WZBSocialScienceCenter/pdftabextract)

[other approach: Camelot](https://hackernoon.com/an-open-source-science-tool-to-extract-tables-from-pdfs-into-excels-3ed3cc7f22e1)