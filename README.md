picktab
-------

Based on a script from [pdftabextract](https://github.com/WZBSocialScienceCenter/pdftabextract),
I share my code. 

It extracts first the tabular layout and goes on to the OCR reading.

It's a kind of slow, but data quality matters.

The reason for this is, that tesseract makes a lot of mistakes, when processing layouted tables, e.g. ignoring some columns or cells.
That has nothing to do with the quality of training. 

Therefore this script uses the layout recognition of pdftabextract, which would also be able to adapt to skewed layouts,
to read ocr from pictures of the single cells.

For more documentation and testing see the module itself.

