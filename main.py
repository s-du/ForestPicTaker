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

        # Add icons to buttons TODO: update to google icons
        self.add_icon(res.find('img/load.png'), self.pushButton_main_load)
        self.add_icon(res.find('img/rectangle.png'), self.pushButton_rect)
        self.add_icon(res.find('img/circle.png'), self.pushButton_circle)
        self.add_icon(res.find('img/label.png'), self.pushButton_addCat)
        self.add_icon(res.find('img/hand.png'), self.pushButton_hand)
        self.add_icon(res.find('img/magic2.png'), self.pushButton_run)

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
        # 'Simplify buttons'
        self.pushButton_main_load.clicked.connect(self.get_image)
        self.pushButton_addCat.clicked.connect(self.add_cat)
        self.pushButton_rect.clicked.connect(self.rectangle_selection)
        self.pushButton_run.clicked.connect(self.go_segment)
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
        training_labels = np.zeros(img.shape[:2], dtype=np.uint8)

        for i, cat in enumerate(self.categories):
            for roi in cat.roi_list:
                start_x = int(roi[0].x())
                start_y = int(roi[0].y())
                end_x = int(roi[1].x())
                end_y = int(roi[1].y())

                training_labels[start_y:end_y, start_x:end_x] = i+1

        results = wk.weka_segment(img, training_labels)
        dest_path = self.image_path[:-4] + 'segmented.jpg'

        skimage.io.imsave(dest_path, results*10)

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
        ax[0].imshow(segmentation.mark_boundaries(img, results, mode='thick'))
        ax[0].contour(training_labels)
        ax[0].set_title('Image, mask and segmentation boundaries')
        ax[1].imshow(results)
        ax[1].set_title('Segmentation')
        fig.tight_layout()
        plt.show()

    def add_cat(self):
        text, ok = QtWidgets.QInputDialog.getText(self, 'Text Input Dialog',
                                                  'Enter name of category:')
        if ok:
            self.comboBox_cat.addItem(text)
            self.comboBox_cat.setEnabled(True)
            if self.image_loaded:
                self.pushButton_rect.setEnabled(True)
                self.pushButton_circle.setEnabled(True)

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
        self.pushButton_hand.setChecked(True)

    def rectangle_selection(self):
        if self.pushButton_rect.isChecked():
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

        self.pushButton_hand.setEnabled(True)
        self.pushButton_hand.setChecked(True)

        if self.comboBox_cat.count() > 0:
            self.pushButton_rect.setEnabled(True)
            self.pushButton_circle.setEnabled(True)



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
        app.setStyle('QtCurve')

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