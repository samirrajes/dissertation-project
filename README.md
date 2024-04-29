# A Comparative Analysis of Generative Paradigms for Text to Image Synthesis

**University of Warwick CS310 - Final Year Project**

## Introduction
This project explores various generative models for synthesizing images from textual descriptions. It particularly focuses on comparing two major paradigms: Diffusion Models (DDPM) and Generative Adversarial Networks (GANs). Both unconditional and conditional approaches are analyzed using FID metrics on the FashionMNIST dataset.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
  - [Unconditional DDPM](#unconditional-ddpm)
  - [Conditional DDPM](#conditional-ddpm)
  - [Unconditional DCGAN](#unconditional-dcgan)
  - [Conditional DCGAN](#conditional-dcgan)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

## Models

### Unconditional DDPM
The `ddpm.ipynb` notebook explores the Unconditional Denoising Diffusion Probabilistic Model (DDPM). This model generates images without conditioning on any additional external information.

### Conditional DDPM
The `cfg-ddpm.ipynb` notebook details the Conditional Denoising Diffusion Probabilistic Model. It conditions the image generation process on class labels to guide the synthesis process.

### Unconditional DCGAN
The `dcgan.ipynb` notebook implements an Unconditional Deep Convolutional GAN (DCGAN). This model learns to generate new images from noise without any conditional input.

### Conditional DCGAN
The `cdcgan.ipynb` notebook covers the Conditional Deep Convolutional GAN (DCGAN). It enhances the generative capabilities of DCGAN by conditioning the generation process on class label information.

## Usage
Open the notebook of choice within any Jupyter environment and execute the cells sequentially as per the instructions provided within the notebook.

## Dependencies
To ensure proper execution of the notebooks, the following libraries and frameworks are required:
- Python 3.8+
- NumPy
- Matplotlib
- tqdm
- PyTorch
- torchvision
- torch-fidelity (for DDPM notebooks)
- PIL (for GAN notebooks)

## Acknowledgements
We extend our gratitude to Brian Pulfer for his U-Net implementation, which significantly aided in establishing our baseline for the Unconditional DDPM. His code can be found at [Brian Pulfer's GitHub](https://github.com/BrianPulfer/PapersReimplementations/blob/main/src/cv/ddpm/models.py). Additionally, we appreciate the educational resources provided by Jason Brownlee at Machine Learning Mastery, which offered an accessible introduction to programming Deep Convolutional GANs. His tutorial can be accessed at [Machine Learning Mastery's guide to DCGANs](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/).
