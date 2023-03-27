
import skimage.io
from skimage import io,data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial



def weka_segment(img_array, training_labels):
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

    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            channel_axis=-1)
    features = features_func(img_array)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(training_labels, features, clf)
    result = future.predict_segmenter(features, clf)

    return result