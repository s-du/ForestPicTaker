<p align="center">
    <a href="https://ibb.co/mynnMm0"><img src="https://i.ibb.co/3FxxLwv/forest-Pic-Taker2.png" alt="forest-Pic-Taker2" border="0"></a>
</p>

## Introduction

ForestPicTaker is a Pyside6 application for using random forests segmentation algorithm (from scikit-learn). 
:evergreen_tree: :deciduous_tree:

**The project is still in pre-release, so do not hesitate to send your recommendations or the bugs you encountered!**

<p align="center">
    <a href="https://ibb.co/wwDnS2v"><img src="https://i.ibb.co/F0GQ5dP/weka.jpg" alt="weka" border="0"></a>
    
    GUI for random forests image segmentation
</p>


## Principle
This concept is based on this tutorial: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html
We decided to add a simple graphical user interface for making the labelling process easier!


### Step 1: Importing an image
Simply choose an image from your HDD

### Step 2: Add classes
Add one or several classes and give them names. Note that the random-forest based segmentation approach uses local features based on local intensity, edges and textures at different scales. It is not a semantic-based approach!

### Step 3: Label image
With the rectangular, or the simple 'brush' tool, you can label the image with the defined classes.
When the labelling is finished, simply click on the 'tree' icon to get a result!

<p align="center">
    <a href="https://ibb.co/CwvHZ5f"><img src="https://i.ibb.co/7SV15Jq/weka-2.jpg" alt="weka-2" border="0"></a>
    
    Result of the segmentation process
</p>

## Upcoming key features:

- **Choosing segmentation parameters**
- **Export/import models**
- **Processing batch of imges**:
    - batch can then be used as input for photogrammetry reconstructions
- **Integrated WebODM support**

## Installation instructions

1. Clone the repository:
```
git clone https://github.com/s-du/ForestPicTaker
```

2. Navigate to the app directory:
```
cd ForestPicTaker
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Run the app:
```
python main.py
```

## User manual
(coming soon)

## Contributing

Contributions to the IRMapper App are welcome! If you find any bugs, have suggestions for new features, or would like to contribute enhancements, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request describing your changes.

## Acknowledgements
This project was made possible thanks to subsidies from the Brussels Capital Region, via Innoviris.
Feel free to use or modify the code, in which case you can cite Buildwise and the Pointify project!

## TO DO

- [ ] Add other segmentation algorithm (eg. Segment anything)
- [ ] Add integrated photogrammetric reconstruction for batch of images (with associated point cloud segmentation based on image segmentation)

