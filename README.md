# Age_by_Voice
Determining the age of a person by the features of their voice

# Paper

Before jumping into the code itself I highly recommend reading the paper [Age_by_Voice.pdf](Age_by_Voice.pdf)

# Folder Structure

## notebooks

All Jupyter notebooks present in this repository attend a unique task, indicated by their name.
Notebooks "_result" in their name are built for a quick overview for each task (gender- and age estimation).

## src/age_by_voice

This folder contains a small python libary with additional source code, including:
- a 3-D plot server for voice clusters
- dataset parsers and their base classes and data models
- a dataset preparator, which was not used in the final model development

## data, build, notes.txt

Custom files and folders can, or have to be made when trying out the code:

- The python environment (either venv or conda env)
- the feature and voice set, see [parse_datasets.ipynb](notebooks/parse_datasets.ipynb)
- the audio cluster for the [audio cluster server](src/age_by_voice/server/voice_cluster_server.py), see [audio_cluster.ipynb](notebooks/audio_cluster.ipynb)


Only with a working features.csv and voices.csv file the other notebooks, especially the results will work!