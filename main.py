import skimage.io
from PySide6 import QtWidgets, QtGui, QtCore

import os
import widgets as wid
import weka as wk
import resources as res

import numpy as np

from skimage import data, segmentation, feature, future
import matplotlib.pyplot as plt


class PixelCategory:
    def __init__(self):
        self.nb_roi = 0
        self.item_list = []
        self.roi_list = []
        self.color = None
        self.name = ''

class WEKAWindow(QtWidgets.QMainWindow):
    """
    Main Window class for the Pointify application.
    """

    def __init__(self, parent=None):
        """
        Function to initialize the class
        :param parent:
        """
        super(WEKAWindow, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        basename = 'segment'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        print(uifile)
        wid.loadUi(uifile, self)

        # define categories
        self.categories = []
        self.active_category = None

        self.image_array = []
        self.image_path = ''

        self.training_labels = None

        # add actions to action group
        ag = QtGui.QActionGroup(self)
        ag.setExclusive(True)
        ag.addAction(self.actionRectangle_selection)
        ag.addAction(self.actionHand_selector)
        ag.addAction(self.actionBrush)


        # Add icons to buttons TODO: update to google icons
        self.add_icon(res.find('img/label.png'), self.pushButton_addCat)

        self.add_icon(res.find('img/load.png'), self.actionLoad_image)
        self.add_icon(res.find('img/rectangle.png'), self.actionRectangle_selection)
        self.add_icon(res.find('img/hand.png'), self.actionHand_selector)
        self.add_icon(res.find('img/brush.png'), self.actionBrush)
        self.add_icon(res.find('img/test.png'), self.actionTest)
        self.add_icon(res.find('img/magic2.png'), self.actionRun)

        self.image_loaded = False

        # tools initialization
        self.rect_active = False

        self.viewer = wid.PhotoViewer(self)
        self.horizontalLayout.addWidget(self.viewer)

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        # create connections (signals)
        self.create_connections()

    def add_icon(self, img_source, pushButton_object):
        """
        Function to add an icon to a pushButton
        """
        pushButton_object.setIcon(QtGui.QIcon(img_source))

    def create_connections(self):
        self.pushButton_addCat.clicked.connect(self.add_cat)
        self.actionLoad_image.triggered.connect(self.get_image)
        self.actionRectangle_selection.triggered.connect(self.rectangle_selection)
        self.actionRun.triggered.connect(self.go_segment)
        self.actionTest.triggered.connect(self.generate_multi_outputs)

        self.viewer.endDrawing.connect(self.add_roi)
        self.comboBox_cat.currentIndexChanged.connect(self.on_cat_change)

    def on_cat_change(self):
        self.active_i = self.comboBox_cat.currentIndex()
        if self.categories:
            self.active_category = self.categories[self.active_i]
            print(self.active_category)

    def go_segment(self):
        # load image
        img = self.image_array
        self.training_labels = wk.generate_training(img, self.categories)

        results = wk.weka_segment(img, self.training_labels)
        dest_path = self.image_path[:-4] + 'segmented.jpg'

        #results = skimage.color.label2rgb(results)
        skimage.io.imsave(dest_path, results)

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
        ax[0].imshow(segmentation.mark_boundaries(img, results, mode='thick'))
        ax[0].contour(self.training_labels)
        ax[0].set_title('Image, mask and segmentation boundaries')
        ax[1].imshow(results)
        ax[1].set_title('Segmentation')
        fig.tight_layout()
        plt.show()

    def generate_multi_outputs(self):
        img = self.image_array
        results = []
        test_edges = [True, False]
        test_sigma_min = [0.5, 2]
        test_sigma_max = [4, 16]

        if self.training_labels == None:
            self.training_labels = wk.generate_training(img, self.categories)

        for test_e in test_edges:
            for test_s_min in test_sigma_min:
                for test_s_max in test_sigma_max:
                    result = wk.weka_segment(img, self.training_labels, edges=test_e, sigma_min = test_s_min,
                                                  sigma_max = test_s_max)
                    results.append(result)

        fig, ax = plt.subplots(1, 8, sharex=True, sharey=True, figsize=(12, 4))
        for i,a in enumerate(ax):
            img = results[i]
            a.imshow(img, interpolation='none')
        fig.tight_layout()
        plt.show()

    def add_cat(self):
        text, ok = QtWidgets.QInputDialog.getText(self, 'Text Input Dialog',
                                                  'Enter name of category:')
        if ok:
            self.comboBox_cat.addItem(text)
            self.comboBox_cat.setEnabled(True)
            if self.image_loaded:
                self.actionRectangle_selection.setEnabled(True)

            # add header to ROI list
            self.add_item_in_tree(self.model, text)
            cat = PixelCategory()
            cat.name = text

            self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Categories')

            # add color
            color = QtWidgets.QColorDialog.getColor()
            cat.color = color

            # add classification category to list of categories
            self.categories.append(cat)


            # select combobox
            nb_cat = len(self.categories)
            self.comboBox_cat.setCurrentIndex(nb_cat-1)

            self.on_cat_change()

    def add_roi(self, nb):
        # add roi item
        # find name in model
        rect_item = self.model.findItems(self.active_category.name)

        cat_from_gui = self.viewer.getCurrentCat()
        self.active_category.nb_roi = cat_from_gui.nb_roi
        self.active_category.roi_list = cat_from_gui.roi_list
        nb_roi = self.active_category.nb_roi
        desc = 'rect_zone' + str(nb_roi)

        self.add_item_in_tree(rect_item[0], desc)
        self.categories[self.active_i] = self.active_category

        print(self.active_category.roi_list)

        # switch back to hand tool
        self.actionHand_selector.setChecked(True)

        self.actionRun.setEnabled(True)
        self.actionTest.setEnabled(True)


    def rectangle_selection(self):
        if self.actionRectangle_selection.isChecked():
            # transmit categories
            self.viewer.setCat(self.active_category, self.categories)

            # define color of rectangle selection
            color = self.active_category.color

            # activate drawing tool
            self.viewer.pen.setColor(color)
            self.viewer.rect = True
            self.viewer.toggleDragMode()

    def get_image(self):
        img = QtWidgets.QFileDialog.getOpenFileName(self, u"Ouverture de fichiers","", "Image Files (*.png *.jpg *.bmp)")
        if not img:
            return
        self.load_image(img[0])

    def load_image(self, path):
        print(path)
        self.image_path = path
        self.image_array = skimage.io.imread(path)
        self.viewer.setPhoto(QtGui.QPixmap(path))
        self.image_loaded = True

        self.actionHand_selector.setEnabled(True)
        self.actionHand_selector.setChecked(True)

        if self.comboBox_cat.count() > 0:
            self.actionRectangle_selection.setEnabled(True)

    def add_item_in_tree(self, parent, line):
        item = QtGui.QStandardItem(line)
        parent.appendRow(item)

def main(argv=None):
    """
    Creates the main window for the application and begins the \
    QApplication if necessary.

    :param      argv | [, ..] || None

    :return      error code
    """

    # Define installation path
    install_folder = os.path.dirname(__file__)

    app = None

    # create the application if necessary
    if (not QtWidgets.QApplication.instance()):
        app = QtWidgets.QApplication(argv)
        app.setStyle('Fusion')

    # create the main window

    window = WEKAWindow()
    window.show()

    # run the application if necessary
    if (app):
        return app.exec_()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))