# Brain-Breast Tumor Classification and Segmentation

This is a keras implementation of single super resolution algorithms: [EDSR](https://arxiv.org/abs/1707.02921), [SRGAN](https://arxiv.org/abs/1609.04802), [SRFeat](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf), [RCAN](https://arxiv.org/abs/1807.02758), [ESRGAN](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf) and [ERCA](https://drive.google.com/open?id=1GFEMT8rCR7SovhudMWFP_lvP_DrtHoTP) (ours). This project aims to improve the performace of the baseline (SRFeat).

To run this project you need to setup the environment, train and test the network models. I will show you step by step to run this project and i hope it is clear enough.

## Prerequiste

I tested my project in Corei7, 16G RAM, GPU RTX2070-super.

## Environment

I recommend you using virtualenv to create a virtual environments. You can install virtualenv (which is itself a pip package) via

```
pip install virtualenv
```

Create a virtual environment called venv with python3, one runs

```
virtualenv -p python3 .env
```

Activate the virtual enviroment:

```
source .env/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

## Dataset

I used a custom dataset which you can find in the repository ([here](https://github.com/Gooda97/Brain-Breast-tumor-classification-and-segmentation/tree/main/Dataset))
The dataset contains set of radiology images for brain in which you can find tumor and normal radiologies. And for the tumor images you can find their masks.
It also contains three classes for breast radiologies (Normal, Benign, Malignant) and for both of benign and malignant classes you will find their masks.

For the first task (Brain breast classification) I used the same dataset but with different hierarchy which you can find [here](https://github.com/Gooda97/Brain-Breast-tumor-classification-and-segmentation/tree/main/Brain_breast_model/Dataset_task_1)

## Training

We have here 5 models to train:

- Brain-Breast classifier
- Brain-Tumor Classification
- Breast-Tumor Classification
- Brain-Tumor segmentation
- Breast-Tumor segmentation

Each of them has its own folder and its own notebook.

You can find the hierarchy and the model metrics in the PDF file [here](<https://github.com/Gooda97/Brain-Breast-Tumor-Classification-and-Segmentation/blob/main/Brain%2C%20Breast%20Classification%20%26%20Segmentation%20(1).pdf>)

## Deployment

We made the deployment on a Flask server that you acn run using the file [test.py](https://github.com/Gooda97/Brain-Breast-Tumor-Classification-and-Segmentation/blob/main/Deploy/test.py)