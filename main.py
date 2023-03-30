# imports
from PySide6 import QtWidgets, QtGui, QtCore
from skimage import color, io, data, segmentation, feature, future
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('qtagg') # for avoiding problems with pyinstaller
import os

# custom libraries
import widgets as wid
import weka as wk
import resources as res


class PixelCategory:
    """
    Class to describe a segmentation category
    """
    def __init__(self):
        self.nb_roi_rect = 0
        self.nb_roi_brush = 0
        self.item_list_rect = []
        self.item_list_brush = []
        self.roi_list_rect = []
        self.roi_list_brush = []
        self.color = None
        self.name = ''

class AboutDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('What is this app about?')
        self.layout = QtWidgets.QVBoxLayout()

        about_text = QtWidgets.QLabel('This app was made by Buildwise, to simplify the use of one simple, '
                                      'yet efficient, machine learning algorithm. Start by loading an image, '
                                      'then adding a category.')
        self.layout.addWidget(about_text)

        self.setLayout(self.layout)

class WEKAWindow(QtWidgets.QMainWindow):
    """
    Main Window class for the ForestPicTaker application.
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

        self.image_array = []
        self.image_path = ''
        self.image_loaded = False

        # define categories
        self.categories = []
        self.active_category = None
        self.training_labels = None
        self.model_available = False

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        # add actions to action group
        ag = QtGui.QActionGroup(self)
        ag.setExclusive(True)
        ag.addAction(self.actionRectangle_selection)
        ag.addAction(self.actionHand_selector)
        ag.addAction(self.actionBrush)

        # Add icons to buttons
        self.add_icon(res.find('img/label.png'), self.pushButton_addCat)

        self.add_icon(res.find('img/load.png'), self.actionLoad_image)
        self.add_icon(res.find('img/rectangle3.png'), self.actionRectangle_selection)
        self.add_icon(res.find('img/hand.png'), self.actionHand_selector)
        self.add_icon(res.find('img/brush.png'), self.actionBrush)
        self.add_icon(res.find('img/test.png'), self.actionTest)
        self.add_icon(res.find('img/forest.png'), self.actionRun)
        self.add_icon(res.find('img/reset.png'), self.actionReset_all)
        self.add_icon(res.find('img/settings.png'), self.actionParameters)
        self.add_icon(res.find('img/folder.png'), self.actionApply_to_folder)
        self.add_icon(res.find('img/info.png'), self.actionInfo)

        self.viewer = wid.PhotoViewer(self)
        self.horizontalLayout.addWidget(self.viewer)

        # create connections (signals)
        self.create_connections()

    def reset_parameters(self):
        """
        Reset all model parameters (image and categories)
        """
        self.image_array = []
        self.image_path = ''
        self.image_loaded = False

        # define categories
        self.categories = []
        self.active_category = None
        self.training_labels = None

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        # clean graphicscene
        self.viewer.clean_scene()

        # clean combobox
        self.comboBox_cat.clear()

    def reset_roi(self):
        # clean tree view
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        # clean roi in each cat
        for cat in self.categories:
            cat.nb_roi_rect = 0
            cat.nb_roi_brush = 0
            cat.item_list_rect = []
            cat.item_list_brush = []
            cat.roi_list_rect = []
            cat.roi_list_brush = []

            self.add_item_in_tree(self.model, cat.name)
            self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Categories')

        # clean graphicscene
        self.viewer.clean_scene()

    def add_icon(self, img_source, pushButton_object):
        """
        Function to add an icon to a pushButton
        """
        pushButton_object.setIcon(QtGui.QIcon(img_source))

    def create_connections(self):
        """
        Link signals to slots
        """
        self.pushButton_addCat.clicked.connect(self.add_cat)
        self.actionLoad_image.triggered.connect(self.get_image)
        self.actionRectangle_selection.triggered.connect(self.rectangle_selection)
        self.actionBrush.triggered.connect(self.brush_selection)
        self.actionRun.triggered.connect(self.go_segment)
        self.actionTest.triggered.connect(self.generate_multi_outputs)
        self.actionReset_all.triggered.connect(self.reset_roi)
        self.actionApply_to_folder.triggered.connect(self.apply_to_folder)
        self.actionInfo.triggered.connect(self.show_info)

        self.viewer.endDrawing_rect.connect(self.add_roi_rect)
        self.viewer.endDrawing_brush.connect(self.add_roi_brush)
        self.comboBox_cat.currentIndexChanged.connect(self.on_cat_change)

    def show_info(self):
        dialog = AboutDialog()
        if dialog.exec_():
            pass

    def apply_to_folder(self):
        if self.model_available:
            # dialog folder selection
            folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

            # analyse images
            if not folder == "":  # if user cancel selection, stop function
                self.main_folder = folder
                self.app_folder = os.path.join(folder, 'ForestPicTaker_outputs')

                if not os.path.exists(self.app_folder):
                    os.mkdir(self.app_folder)

                img_list = os.listdir(folder)
                img_paths = []

                for img_file in img_list:
                    if img_file.endswith('.jpg') or img_file.endswith('.JPG'):
                        img_paths.append(os.path.join(folder, img_file))

                print(img_paths)
                for i, path in enumerate(img_paths):
                    img_array = io.imread(path)
                    img_array = wk.rgba2rgb(img_array)
                    self.training_labels = wk.generate_training(img_array, self.categories)
                    features_new = self.feat_func(img_array)
                    results_new = future.predict_segmenter(features_new, self.clf)
                    results = color.label2rgb(results_new)

                    dest_path = os.path.join(self.app_folder, f'segmented_{i}.jpg')
                    io.imsave(dest_path, results)

    def on_cat_change(self):
        """
        When the combobox to choose a segmentation category is activated
        """
        self.active_i = self.comboBox_cat.currentIndex()
        if self.categories:
            self.active_category = self.categories[self.active_i]
            print(f"the active segmentation category is {self.active_category}")

    def go_segment(self):
        """
        Launch the segmentation
        """
        # load image
        img = self.image_array
        img = wk.rgba2rgb(img)
        # generate training data
        self.training_labels = wk.generate_training(img, self.categories)

        self.clf, self.feat_func, results = wk.weka_segment(img, self.training_labels)
        dest_path = self.image_path[:-4] + 'segmented.jpg'

        #results = skimage.color.label2rgb(results)

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
        ax[0].imshow(segmentation.mark_boundaries(img, results, mode='thick'))
        ax[0].contour(self.training_labels)
        ax[0].set_title('Image, mask and segmentation boundaries')
        ax[1].imshow(results)
        ax[1].set_title('Segmentation')
        fig.tight_layout()
        plt.show()

        self.model_available = True
        self.actionApply_to_folder.setEnabled(True)

    def generate_multi_outputs(self):
        """
        Launch an analysis with different segmentation parameters
        """
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
        """
        Add a segmentation category (eg. 'wood', 'bricks', ...)
        """
        text, ok = QtWidgets.QInputDialog.getText(self, 'Text Input Dialog',
                                                  'Enter name of category:')
        if ok:
            # add color
            color = QtWidgets.QColorDialog.getColor()
            print(color.rgb())
            if color.isValid():
                # add category to combobox
                self.comboBox_cat.addItem(text)
                self.comboBox_cat.setEnabled(True)

                # add header to ROI list
                self.add_item_in_tree(self.model, text)
                self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Categories')

                # create category class
                cat = PixelCategory()
                cat.name = text
                cat.color = color

                # add classification category to list of categories
                self.categories.append(cat)

                # activate tools
                self.actionBrush.setEnabled(True)
                self.actionRectangle_selection.setEnabled(True)

                # select new cat in combobox
                nb_cat = len(self.categories)
                self.comboBox_cat.setCurrentIndex(nb_cat-1)
                self.on_cat_change()

    def add_roi_brush(self, nb):
        """
        Add a region of interest coming from the brush tool
        :param nb: number of existing roi's
        """
        # find the parent item in the treeview (the active category)
        brush_item = self.model.findItems(self.active_category.name)
        cat_from_gui = self.viewer.get_current_cat()

        self.active_category.nb_roi_brush = cat_from_gui.nb_roi_brush
        self.active_category.roi_list_brush = cat_from_gui.roi_list_brush
        nb_roi = self.active_category.nb_roi_brush
        desc = 'brush_zone' + str(nb_roi)

        self.add_item_in_tree(brush_item[0], desc)
        self.categories[self.active_i] = self.active_category
        self.treeView.expandAll()

        # switch back to hand tool
        self.hand_pan()

        self.actionRun.setEnabled(True)
        self.actionTest.setEnabled(True)
        self.actionReset_all.setEnabled(True)

    def add_roi_rect(self, nb):
        """
        Add a region of interest coming from the rectangle tool
        :param nb: number of existing roi's
        """
        # add roi item
        # find name in model
        rect_item = self.model.findItems(self.active_category.name)

        cat_from_gui = self.viewer.get_current_cat()
        self.active_category.nb_roi_rect = cat_from_gui.nb_roi_rect
        self.active_category.roi_list_rect = cat_from_gui.roi_list_rect
        nb_roi = self.active_category.nb_roi_rect
        # create description name
        desc = 'rect_zone' + str(nb_roi)

        # get size of roi

        self.add_item_in_tree(rect_item[0], desc)
        self.categories[self.active_i] = self.active_category
        self.treeView.expandAll()

        # switch back to hand tool
        self.hand_pan()

        # enable actions
        self.actionRun.setEnabled(True)
        self.actionTest.setEnabled(True)
        self.actionReset_all.setEnabled(True)

    def hand_pan(self):
        # switch back to hand tool
        self.actionHand_selector.setChecked(True)
        # activate combobox
        self.comboBox_cat.setEnabled(True)

    def brush_selection(self):
        if self.actionBrush.isChecked():
            self.viewer.set_cat(self.active_category, self.categories)

            # desactivate combobox
            self.comboBox_cat.setEnabled(False)

            # define color of rectangle selection
            color = self.active_category.color

            # change cursor
            self.viewer.change_to_brush_cursor()

            # activate drawing tool
            self.viewer.pen.setColor(color)
            self.viewer.painting = True
            self.viewer.toggleDragMode()

    def rectangle_selection(self):
        if self.actionRectangle_selection.isChecked():
            # transmit categories
            self.viewer.set_cat(self.active_category, self.categories)

            # define color of rectangle selection
            color = self.active_category.color

            # activate drawing tool
            self.viewer.pen.setColor(color)
            self.viewer.rect = True
            self.viewer.toggleDragMode()

    def get_image(self):
        """
        Get the image path from the user
        :return:
        """
        try:
            img = QtWidgets.QFileDialog.getOpenFileName(self, u"Ouverture de fichiers","", "Image Files (*.png *.jpg *.bmp)")
            print(f'the following image will be loaded {img[0]}')
        except:
            pass
        if img[0] != '':
            # load and show new image
            self.load_image(img[0])

    def load_image(self, path):
        """
        Load the new image and reset the model
        :param path:
        :return:
        """
        self.reset_parameters()

        self.image_path = path
        self.image_array = io.imread(path)
        self.viewer.setPhoto(QtGui.QPixmap(path))
        self.image_loaded = True

        # enable action
        self.pushButton_addCat.setEnabled(True)
        self.actionHand_selector.setEnabled(True)
        self.actionHand_selector.setChecked(True)


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