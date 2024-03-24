# CSSR Model Code Repository

This repository contains the implementation of the CSSR (Cross-Slice Super-Resolution) model described in the following publication:

**Paper Title:** CSSR: 3D reconstruction of the uterus based on super-resolution of thick-slice MRI

The dataset utilized for training the model originates from a dataset paper crafted by our research team. Details are as follows:

**Dataset Paper:** Large-scale uterine myoma MRI dataset covering all FIGO types with pixel-level annotations

**Journal:** Scientific Data

**Status:** Accepted for publication(Manuscript Sent to Production)

**Expected Publication Date:** Forthcoming

The dataset link is referenced in the accompanying cover letter.

## Repository Structure

- `train.py`: The main script for initiating the CSSR model training process. 
- `pred.py`: The script for performing inference with the trained CSSR model and generating 3d models.
