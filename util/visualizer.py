import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        # self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        # if self.use_html:
        if opt.isTrain:
            self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        else:
            self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_{}/images'.format(opt.which_iter))
        print('create img_dir %s...' % self.img_dir)
        util.mkdirs([self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, step):

        for label, image_numpy in visuals.items():
            img_path = os.path.join(self.img_dir, '%s_iter%d.jpg' % (label,step))
            # print('type: ', image_numpy.dtype)
            util.save_image(image_numpy, img_path)


    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, i, errors, t):
        message = '(iters: %d, time: %.3f) ' % (i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.6f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()

        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)


