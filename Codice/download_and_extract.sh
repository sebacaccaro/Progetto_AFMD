#!/bin/bash
kaggle datasets download - d baltacifatih/turkish-lira-banknote-dataset
mkdir dataset
unzip turkish-lira-banknote-dataset.zip -d dataset