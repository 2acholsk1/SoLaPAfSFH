# SoLaPAfSFH
**S**egmentation **o**f **L**awns **a**nd **P**avings **A**rea **f**or **S**ingle **F**amily **H**ouses
# ZPO Project - Your Deepness Model
This is a template for a Project.
Please follow the structure below and address all required points.
Please put it in a public repository.

AT THE TOP OF THIS README ADD AN IMAGE/GIF WITH EXAMPLE MODEL PREICTION, AS A BANNER

We reecommend making this README pleaseant to read, you can later use it as portfolio `:)`

Main goal of the project was creating a model for segmentation suburbs areas into 3 categories:
- Lawns
- Pawing areas
- The rest


## Dataset
### Data Source
Unfortunately, no pre-existing datasets were available for our specific purpose. Therefore, we created our own dataset by extracting images of suburban areas in Poznań from the "Poznan 2022 aerial ortophoto high resolution" available in the QGIS plugin QuickMapServices.

Using the [Deepness](https://github.com/PUTvision/qgis-plugin-deepness) plugin, these data were processed and exported. The dataset consists of approximately 500 images of suburban areas, each measuring 512x512 pixels.

### Dataset Details
The dataset is available in this repository under data/SoLaPAfSFH.v4i.png-mask-semantic.zip.
It is divided into two subsets: train and test.

### Image and Mask Association
Each photo in the dataset has an associated mask with the same name, suffixed with _mask. For example:
- Image: image1.png
- Mask: image1_mask.png

Each photo have an unique name.

### Mask Encoding
The segmentation masks are encoded as follows:
- 0 - Background
- 1 - Lawn
- 2 - Pawing Areas
The information about the encoding can be also found in the dataset in .csv file.

## Training
- what network, how trained, what parameters
- what augmentation methods used
- what script to run the training
- remember to have a fully specified Python environemnt (Python version, requirements list with versions)
- other instructions to reproduce the training process

## Results
- Example images from dataset (diverse), at least 4 images
- Examples of good and bad predictions, at least 4 images
- Metrics on the test and train dataset

## Trained model in ONNX ready for `Deepness` plugin

To export model in ONYX format the script in solapafsfh\cli\convert_onxx.py is provided.

- model uploaded to XXX and a LINK_HERE
- model have to be in the ONNX format, including metadata required by `Deepness` plugin (spatial resolution, thresholds, ...)
- name of the script used to convert the model to ONNX and add the metadata to it


## Demo instructions and video
- a short video of running the model in Deepness (no need for audio), preferably converted to GIF
- what ortophoto to load in QGIS and what physical place to zoom-in. E.g. Poznan 2022 zoomed-in at PUT campus
- showing the results of running the model

## People
- Piotr Zacholski
- Bruno Maruszczak
- Witold Szerszeń 

## Other information
Feel free to add other information here.