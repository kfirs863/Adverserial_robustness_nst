## Environment Setup
1. Install [Python 3.10](https://www.python.org/downloads/release/python-3100/)
2. Install required packages
```bash
pip install -r requirements.txt
```
## Dataset
1. Set paths in dataset_preprocessing.py
2. Run the script
```bash
python dataset_preprocessing.py
```
## Models
1. Clone Adain-Pytorch into the project directory
```bash
git clone https://github.com/naoto0804/pytorch-AdaIN.git
```

## Weights
1. Download RealESRGAN weights from [here](https://drive.google.com/drive/folders/16PlVKhTNkSyWFx52RPb2hXPIQveNGbxS)
2. Place the weights in the `weights` directory

## Logging to Comet
1. Create a Comet account
2. Set the API key in the `baseline_model_legacy.py` file
```python
    comet_logger = CometLogger(
        api_key="COMET_API_KEY",...)
```

## Inferencing
### Option 1
Run the Jupiter Notebook
```bash
jupyter notebook
```
Open the notebook `Image_Classification_using_pre_trained_models.ipynb`
Run the notebook

### Option 2
Run the full model pipeline
1. Set the paths in the `baseline_model_legacy.py` file to the adversarial images dataset
```python
    dataset = ImageFolder(root='PATH TO SAVE THE ADVERSARIAL IMAGES',transform=transforms.ToTensor())
```

2.  Run the script
```bash
python baseline_model_legacy.py
```
3. See the results in the Comet dashboard
```