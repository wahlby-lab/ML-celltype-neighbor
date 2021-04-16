# Machine learning for cell classification and neighborhood analysis in glioma tissue

This is the code repository for our manuscript (under review) available in [BioRxiv](https://www.biorxiv.org/content/10.1101/2021.02.26.433051v1)

The code shows how we preprocessed our data, how we created our neworks and how we made use of them. However this is not intended as a generic pipeline. To use it in your own data you have to make the modifications. We distribute this code with a GNU GPLv3 license meaning you can use the code as long as you credit us. This code is provided without warranty. The authors or license can not be held liable for any damages inflicted by the code.

The image data of the 21 cores is available upon reasonable request directly to the authors from the Department of Oncology-Pathology, Karolinska Institutet, Sweden. 

1. Prerequisites

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

You need your own multiplex immunofluorescence images and the formats. We intentionally separated cores into their own tiff files, but often the images come out of the microscope in their own formats, which is why we include libraries like python-bioformats.

You need to know which channels represent which markers. You need to understand a little bit about pandas dataframes, CSVs, JSON files

