# Age_by_Voice
Determining the age of a person by the features of their voice

# Folder Structure

## Notebooks

All Jupyter notebooks present in this repository attend a unique task, indicated by their name.
Notebooks "_result" in their name are built for a quick overview for each task (gender- and age estimation).

## src

This folder contains a small python libary with additional source code, including:
- a 3-D plot server for voice clusters
- dataset parsers and their base classes and data models
- a dataset preparator, which was not used in the final model development

## data, build and notes

Custom folders can, or have to be made and created when trying out the code:

- The python environment (either venv or conda env)
- the feature and voice set, see parse_datasets.ipynb
- the audio cluster for the audio cluster server, see audio_cluster.ipynb


only with a working features.csv and voices.csv file the other notebooks, especially the results will work
