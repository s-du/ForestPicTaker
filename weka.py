from skimage import io,data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import numpy as np


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


def generate_training(img_array, categories):
    training_labels = np.zeros(img_array.shape[:2], dtype=np.uint8)

    for i, cat in enumerate(categories):
        for roi in cat.roi_list_rect:
            start_x = int(roi[0].x())
            start_y = int(roi[0].y())
            end_x = int(roi[1].x())
            end_y = int(roi[1].y())

            training_labels[start_y:end_y, start_x:end_x] = i + 1
        for roi in cat.roi_list_brush:
            for j in range(len(roi[:, 1])):
                # take line by line, each line is a coordinate couple
                coord = roi[j, :]

                print(len(training_labels[:,1]))
                print(len(training_labels[1, :]))
                if coord[0] >= len(training_labels[:,1]):
                    coord[0] = len(training_labels[:,1])-1

                if coord[1] >= len(training_labels[1,:]):
                    coord[1] = len(training_labels[1,:])-1

                training_labels[coord[0], coord[1]] = i + 1


    return training_labels


def weka_segment(img_array, training_labels, sigma_min=1, sigma_max=16,edges=False, texture=True):
    # Build an array of labels for training the segmentation.
    # Here we use rectangles but visualization libraries such as plotly
    # (and napari?) can be used to draw a mask on the image.
    """
    training_labels = np.zeros(img.shape[:2], dtype=np.uint8)
    training_labels[:130] = 1
    training_labels[:170, :400] = 1
    training_labels[600:900, 200:650] = 2
    training_labels[330:430, 210:320] = 3
    training_labels[260:340, 60:170] = 4
    training_labels[150:200, 720:860] = 4
    """

    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=edges, texture=texture,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            channel_axis=-1)
    features = features_func(img_array)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(training_labels, features, clf)
    result = future.predict_segmenter(features, clf)

    return clf, features_func, result