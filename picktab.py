import logging

from helpers.time_tools import timeit_context

logging.getLogger().setLevel(logging.INFO)

import numpy as np
import os
import re
from addict import addict

from pdftabextract import imgproc
from pdftabextract.clustering import (find_clusters_1d_break_dist)
from pdftabextract.extract import make_grid_from_positions

from tesserocr import PyTessBaseAPI
from PIL import Image

from Levenshtein._levenshtein import ratio
from autocorrect import spell



class PickTab:
    """ Reads the text from a table layout

     That is hard, because OCR fails to work within such cells, neither it can't forget the half of the table sometimes
     nor it can recognize the layout
     solution: the picture is splitted into cells by the houch-line-algorithm

    It works this way
    -----------------

    1. generate a black-white image, that is optimezed for text recognition, when colored monochrome
     tabular borders and text have to be black, text white to work with tesseract
     e.g. with the textcleaner script of Fred Weinhaus (http://www.fmwconcepts.com/imagemagick/textcleaner/index.php)

    2. generate the grid
     with pdftabextract of Markus Konrad (https://github.com/WZBSocialScienceCenter/pdftabextract)

    3. run tesseract on the grid cells by splitting the numpy array picture into the grid cells
     To be less slowly, use the libtesseract-API and not a python module.

    4. apply a little textcleaning to eradicate misplaced tabular lines with '|' in the beginning.

    """

    def __init__(self, n_cols, min_col_width, min_row_width):
        self.N_COL_BORDERS = n_cols
        self.MIN_COL_WIDTH = min_col_width  # <- very important! minimum width of a column in pixels, measured in the scanned pages
        self.MIN_ROW_WIDTH = min_row_width  # <- very important! minimum width of a row in pixels, measured in the scanned pages

    vertical_lines_clusters = {}
    pages_image_scaling = {}  # scaling of the scanned page image in relation to the OCR page dimensions for each page
    images = {}

    def process(self, path, *args, **kwargs):
        """ Make grid, read ocr, postprocess

        :param path: where to find the black and white (!) picture
        :return: table
        """
        grid = self.create_grid(path)
        table = self.ocr_on_cells(path, grid)
        table = self.post_process_table(table, *args, **kwargs)
        return table

    def miss_out_logo_in_corner(self, table, logo_string, cell_coords=(-1, -1)):
        ''' right bottom corner often you can read the logo: 'differencebetween.net with a wrong detected  tab border,
        delete it by fuzzy matching on the end of the string. '''
        right_bottom = table[cell_coords[0]][cell_coords[1]]
        if ratio(right_bottom[-30:], logo_string) > 0.5:
            ratios = [ratio(right_bottom[-r:], logo_string) for r in range(35)]
            element_to_key = lambda i: ratios[i]
            arg_max_ratios = max(range(len(ratios)), key=element_to_key) + 1
            table[cell_coords[0]][cell_coords[1]] = right_bottom.replace(right_bottom[-arg_max_ratios:], "").strip()
            return table
        else:
            return table

    def correct_this(self, text):
        ''' spell correct the text. '''
        if text.startswith('I ') or text.startswith('| ') or text.startswith('l '):
            text = text[2:]
        return " ".join([spell(s) for s in text.split()])

    def post_process_table(self, table, *args, **kwargs):
        """ If column is totally empty, it woun't be a column, but a double placed column border.
        The postprocessing collapses cell content to single strings, if there is content in some
        of the three cells. Erases the '|' in front of the strings.

        :param table: a unclean table
        :return: clean table
        """

        pattern_header = re.compile(r'[^@]+ vs [^@]+')
        clean_table = addict.Dict()
        # header_found = False
        for i, row in enumerate(table.items()):
            all_texts = [cell[1] for cell in row[1].items()]
            all_texts = [self.correct_this(s) for s in all_texts]

            # if not header_found and any(pattern_header.match(cell) for cell in all_texts):
            #    header = ''.join(all_texts)
            #    logging.info(header)
            #    header_found = True

            if any(s for s in row[1].values()):
                clean_table[i] = all_texts

        for col in range(len(table['0']))[::-1]:
            all_texts = [row[col] for k, row in clean_table.items()]

            if not any(all_texts):
                logging.info('empty column %d' % col)

                for k, row in clean_table.items():
                    del row[col]

        clean_table = [row for row in clean_table.values()]

        self.miss_out_logo_in_corner(clean_table, *args, **kwargs)
        return clean_table

    api = PyTessBaseAPI(oem=1)

    def create_grid(self, path, paint=True):
        ''' create a grid by detecting the tabular borders

        :param path: where to find the image
        :param paint: if one should paint a test picture
        :return:
        '''
        # path = blackwhitify(path)
        imgfile = path

        # create an image processing object with the scanned page
        exists = os.path.isfile(path)
        if not exists:
            logging.info("%s not found, passing" % path)
            return None
        try:
            image_to_process = imgproc.ImageProc(imgfile)
        except OSError:
            logging.info("%s is damaged" % path)
            return None

        # detect the lines
        logging.info("detecting lines in image file '%s'..." % (imgfile))
        with timeit_context('line detecting'):

            with timeit_context('hlines'):
                lines_hough = image_to_process.detect_lines(canny_low_thresh=900, canny_high_thresh=1030,
                                                            canny_kernel_size=3,
                                                            hough_rho_res=0.2,
                                                            hough_theta_res=np.pi / 20,
                                                            hough_votes_thresh=round(0.4 * image_to_process.img_w))
                logging.info("found %d lines at all" % len(lines_hough))

            with timeit_context('hcluster'):
                vertical_clusters = image_to_process.find_clusters(imgproc.DIRECTION_VERTICAL,
                                                                   find_clusters_1d_break_dist,
                                                                   dist_thresh=self.MIN_COL_WIDTH / 2)
            logging.info("thereof %d vertical clusters" % len(vertical_clusters))
            horizontal_clusters = image_to_process.find_clusters(imgproc.DIRECTION_HORIZONTAL,
                                                                 find_clusters_1d_break_dist,
                                                                 dist_thresh=self.MIN_ROW_WIDTH / 2)
            logging.info("thereof %d horizontal clusters" % len(horizontal_clusters))

        vertical_lines = [x[1][0] for x in vertical_clusters]
        horizontal_lines = [x[1][0] for x in horizontal_clusters]
        grid = make_grid_from_positions(vertical_lines, horizontal_lines)  # line_positions[p_num])
        n_rows = len(grid)
        n_cols = len(grid[0])
        logging.info("grid with %d rows, %d columns" % (n_rows, n_cols))

        return grid

    def ocr_on_cells(self, path, grid):
        ''' Reopens the picture, because that works faster than taking it from pdftab by the tesseract api and reads
        every single cell.

        :param path: path to pic
        :param grid: computed grid
        :return: addict-dict table
        '''

        table = addict.Dict()

        with timeit_context('ocr on table cells'):
            image = Image.open(path)
            self.api.SetImage(image)
            #  self.api.SetImage(Image.fromarray(image_to_process.input_img)) # It's faster to reload the picture
            for x, row in enumerate(grid):
                for y, col in enumerate(row):
                    cell = grid[x][y]

                    box = {'x': int(cell[0][0]),
                           'y': int(cell[0][1]),
                           'w': int(cell[1][0] - cell[0][0]),
                           'h': int(cell[1][1] - cell[0][1])}

                    self.api.SetRectangle(box['x'], box['y'], box['w'], box['h'])

                    ocrResult = self.api.GetUTF8Text()
                    table[str(x)][str(y)] = ocrResult.replace('\n', ' ').strip()

        return table

    # difference between net
    convert_args = "-filter lanczos -resize 350% -negate  +dither -colors 3 -monochrome  " \
                   "-channel A  -fuzz 0% -transparent black -fuzz 0% -transparent white -negate +channel   " \
                   "-filter lanczos -resize 250%  +dither -colors 2 -monochrome"

    # difference between com
    # convert_args = "-filter lanczos -resize 150% -negate  +dither -colors 10 -monochrome  " \
    #               "-channel A  -fuzz 0% -transparent black -fuzz 0% -transparent white -negate +channel   " \
    #               "-filter lanczos -resize 150%  +dither -colors 2 -monochrome"

    def blackwhitify(self, path):
        ''' external convertion to blackwhite picture, more or less obsolete

        :param: path to the pic
        :return: path to black and white picture, created at the same location as the data
        '''
        bw_path = path + ".png"
        convert_command = (
            "convert  -units PixelsPerInch {input} {convert_args}   {output}  ".format(
                input=path, output=bw_path, convert_args=self.convert_args))
        os.system(convert_command)
        return bw_path

    def analyse_pic_data(self, pic_input_dir, input_image_file_filter):
        ''' run recursively through a folder and read every picture as a table, creating a text-file with the content
        to avoid rereading of that

        :param pic_input_dir: where to search for the pictures
        :param input_image_file_filter: wildcarded glob-path to pictures
        :return: addict-dict of the table
        '''

        for path in get_files_from_recursive_path(pic_input_dir + input_image_file_filter):
            logging.info("table for %s" % path)

            try:
                with open(path + '.txt', 'w') as f:
                    f.write(str(self.process(path)))

            except BaseException as error:
                logging.warning('Error: %s occured with %s ' % (str(error), path))
            else:
                print("Everything looks great!")


import unittest


class TestPickTab(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPickTab, self).__init__(*args, **kwargs)

        self.t = PickTab(n_cols=4, min_col_width=300, min_row_width=60)

    def run_picktab(self, path, gold):
        res = self.t.process(path, logo_string='[Dap| DifferenceBetween.net')
        import pprint
        pprint.pprint(res)

        for y, r in enumerate(res):
            for x, c in enumerate(r):
                try:
                    self.assertGreater(ratio(res[y][x].lower(), gold[y][x].lower()), 0.96)
                except:
                    print(res[y][x], " <!=> ", gold[y][x])
                    raise

    def test_vs1(self):
        path = "./tests/picktab/Amazon-vs-Google.jpg.bw.png" #http://www.differencebetween.net/technology/internet/difference-between-amazon-and-google/
        gold = [['Amazon', 'Google'],
                ['Amazon is the market leader in the online retail landscape spearheaded by '
                 'CEO Jeff Bezos.',
                 'Sunday Pichai-led Google is the undisputed leader of the search realm'],
                ['Prime Video is the Amazons own online streaming service',
                 'YouTube offers everything from hottest music videos to trending games movie '
                 'trailers and more'],
                ['Amazon Echo is a family of voice-based smart speakers developed by Amazon',
                 'Google Home is the family of smart speakers developed by Googled'],
                ['Alexa is Amazons voice-based digital assistant',
                 'Google Assistant is Googles own voice assistant'],
                ['AWS is a comprehensive cloud computing platform from Amazon',
                 'Google Cloud is a suite of public cloud computing services offered by '
                 'Google']]
        self.run_picktab(path, gold)

    def test_vs2(self):
        path = "./tests/picktab/AWS-VERSUS-Google-Cloud.jpg.bw.png" # http://www.differencebetween.net/technology/difference-between-aws-and-google-cloud/
        gold = [['AWS', 'Google Cloud'],
                ['AWS is a secure cloud service platform developed and managed by Amazon',
                 'Google Cloud Platform is a suite of googles public cloud computing '
                 'resources and services'],
                ['It dominates the public cloud market with its broad range of cloud-based '
                 'products and services',
                 'It specializes in high compute offerings like Big Data but is fairly new to '
                 'the cloud scene'],
                ['The flagship compute service of AWS is Elastic Compute Clouds or ECHO',
                 'googles primary compute service is Compute Engine'],
                ['its on a little higher end in terms of compute and storage costs',
                 'its a clear winner with its competitive pricing as compared to other cloud '
                 'providers']]
        self.run_picktab(path, gold)

    def test_vs3(self):
        path = "./tests/picktab/Anointing-vs-Holy-Spirit.jpg.bw.png" # http://www.differencebetween.net/miscellaneous/religion-miscellaneous/difference-between-anointing-and-holy-spirit/
        gold = [['Characteristics', 'Anointing', 'Holy Spirit'],
                ['Meaning',
                 'Anointing means smearing pilon a person body to mark a religious belief It '
                 'also means being sanctified by the Holy Spirit to live in the ways of the '
                 'Holy Spirit',
                 'One of the parts making up the Trinity the Father Son Holy Spirit'],
                ['Nature',
                 'a verb referring to the process of getting someone to follow the ways of '
                 'Cod through the Holy Spirit a Anointing is also exclusive',
                 'A person making the Holy Trinity a Holy Spirit is also inclusive']]
        self.run_picktab(path, gold)

    def test_VERSUS(self):
        path = "./tests/picktab/Capex-VERSUS-Opex.jpg.bw.png" # http://www.differencebetween.net/business/difference-between-capex-and-opex/
        gold = [['Meaning',
                 'Costs Incurred when Buying Assets',
                 'Expenses Incurred in Daily Running of an Entity'],
                ['Costs Involved', 'Costly', 'Relatively Costly'],
                ['Accounting Treatment',
                 'Depreciated and Amortized',
                 'Deducted in the same Accounting Period'],
                ['Profits Earned', 'Slow and Gradual', 'Earned for Shorter Time'],
                ['Examples',
                 'Machinery Equipment and Patents',
                 'Rent Utilities and Administrative Costs'],
                ['Sources of Finance', 'Lending Institutions', 'Personal Savings']]
        self.run_picktab(path, gold)


if __name__ == '__main__':
    unittest.main()

