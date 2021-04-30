## Machine learning for cell classification and neighborhood analysis in glioma tissue

[![bioRxiv shield](https://img.shields.io/badge/bioRxiv-10.1101/2021.02.26.433051-red.svg)](https://www.biorxiv.org/content/10.1101/2021.02.26.433051v1)

The code shows how we preprocessed our data, how we created our neworks and how we made use of them. However this is not intended as a generic pipeline. To use it in your own data you have to make the modifications. We distribute this code with a GNU GPLv3 license meaning you can use the code as long as you credit us. This code is provided without warranty. The authors or license can not be held liable for any damages inflicted by the code.

The image data of the 21 cores is available upon reasonable request directly to the authors from the Department of Oncology-Pathology, Karolinska Institutet, Sweden. 

# 1. Prerequisites

We use several libraries, to be able to use all the methods in this pipeline. This is a list yo can put in a requirements.txt
+ numpy = "*"
+ scikit-image = "*"
+ jupyter = "*"
+ pandas = "*"
+ xgboost = "*"
+ python-bioformats = "*"
+ torch = "*"
+ torchvision = "*"
+ openslide-python = "*"
+ seaborn = "*"
+ adabelief-pytorch = "*"
+ pytorch-metric-learning = "*"
+ tensorflow = "==1.15.0"
+ tensorflow-gpu = "==1.15.0"
+ networkx = "==2.4"
+ matplotlib = "==3.0.3"
+ stellargraph = "==0.8.1"
+ scipy = "==1.3.1"
+ scikit-learn = ">=0.21.3"
+ tqdm = "==4.36.1"
+ umap-learn = "==0.3.10"
+ scanpy = "==1.4.4"
+ leidenalg = "==0.7.0"
+ h5py = "==2.10.0"
+ loompy = "==3.0.6"
+ jupyter-contrib-nbextensions = "*"

You need your own multiplex immunofluorescence (mIF) images and the formats. We intentionally separated cores into their own tiff files, but often the images come out of the microscope in their own formats, which is why we include libraries like python-bioformats.

You need to know which channels represent which markers. You need to understand a little bit about pandas dataframes, CSVs, JSON files

# 2. Overview

![overview image](https://github.com/wahlby-lab/celltypeneighbor/blob/bd5bf6ce4c0c76f64635d81aa5cb6d193fa84c7e/misc/overview.png)

The overview shows an example of a piece of a core. Features are computed from each segmented cell, such as marker composition and our suggested d90s features. A vector containing these features is the input to an ensemble of FNNs. The output is a class for a cell. Once all cells have a class, we define the neighborhoods as cells being closer than a certain distance, we chose 90th percentile of the minimum distance between any two cells in our dataset. Any two cells at this distance will be connected by an edge and a graph is created which is inputed to a spage2vec (GNN) model to obtain a neighborhood descriptor. Of course both FNN and GNN have to be trained.

In our code we preprocess both images and CSV files to obtain all features and create a dataframe that is used through the rest of the project.
We also include a file fcnn.py which contains the model and the training procedures as we explained in the manuscript.

The method spage2vec is found in this same lab at [wahlby-lab/spage2vec](https://github.com/wahlby-lab/spage2vec) which has many examples of training and inference and we adapted for our work.

We include utilities for display and visualization.

# 3. Specifics

## Nuclei segmentation
So you have mIF images, what to do next?

First you need to have a way to segment nuclei and get a polygon around each cell. Since our collaborators use [QuPath](https://qupath.github.io/) we decided to stick to it and we do the nuclei segmentation in it.

By script this is done like so:

```runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImage": "DAPI",  "requestedPixelSizeMicrons": 0.5,  "backgroundRadiusMicrons": 8.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.5,  "watershedPostProcess": true,  "cellExpansionMicrons": 2.5,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}')```

You don't need to use QuPath if you don't want to, but for our next steps we will asume that it is used.

QuPath offers a simple export where a CSV can be obtained with the measurements such as marker quantification and others. Custom quantifications are not avilable directly and the measurementes do not include the 90th percentile of a marker. So to obtain the 90th percentile, we want to have the polygon so we can look in the image ourselves. For this we created the script `exportcellstojson.groovy`

## Preparing the data 
In the notebook [``1-MakeMainDataFrame.ipynb``](https://github.com/wahlby-lab/ML-celltype-neighbor/blob/main/1-MakeMainDataFrame.ipynb) we create all the segmented nuclei as images with all their channels and compute the D90s

## Training and using the CNN

In the python file [``2-useFCNN.py ``](https://github.com/wahlby-lab/ML-celltype-neighbor/blob/main/2-useFCNN.py) you can train a network ensemble and then use it with [``3-UseNewFCNNensemble.ipynb ``](https://github.com/wahlby-lab/ML-celltype-neighbor/blob/main/3-UseNewFCNNensemble.ipynb).

## Plotting and utilities

The file utils.py  contains functions we created to plot from a pandas data frame and also to do color conversions and create palettes of N different colors which comes in handy when you have N clusters to show.

## Spage2vec

Coming soon. We refer you to the [original work of spage2vec in our lab](https://github.com/wahlby-lab/spage2vec). We simply cloned the notebooks and adapted them


