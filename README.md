# Fashion-MNIST Image Classification — ML, Deep Learning & Generative AI

End-to-end data mining project on the **Fashion-MNIST** dataset (70,000 grayscale 28×28 clothing images, 10 classes). The project covers exploratory data analysis, custom feature engineering, classical ML with ensembling, a convolutional neural network, model evaluation, and a bonus generative AI layer that creates product descriptions and powers semantic search over a synthetic product catalog.

## Results

| Model | Accuracy | F1-Score (Macro) |
|---|---|---|
| **CNN (Deep Learning)** | **0.9008** | **0.9016** |
| Ensemble (Voting: Random Forest + Logistic Regression) | 0.8942 | 0.8931 |
| Logistic Regression | 0.8890 | 0.8883 |
| Random Forest | 0.8705 | 0.8688 |

## Key Findings

- **Shirt, Coat, and Pullover** are the most confused classes — visible in both the confusion matrices and the per-class average images, since they share similar torso silhouettes.
- **Central pixels carry the most discriminative variance.** A spatial variance map showed the middle of each image concentrates most of the signal, motivating the use of structured features over raw pixel vectors for the classical pipeline.
- **Engineered features** — Sobel edge detection, Histogram of Oriented Gradients (HOG), horizontal/vertical symmetry, and PCA-reduced pixel intensities — gave the classical models meaningful structure to learn from.
- **CNN outperformed the classical ensemble by ~1 point of accuracy**, but the ensemble is far cheaper to train and serve — a real trade-off worth weighing, not just "deep learning wins."

## Pipeline

1. **EDA** — class balance, pixel intensity distributions, per-class average images, PCA 2D class separability
2. **Feature Engineering** — spatial variance maps, symmetry features, Sobel edge filter, HOG, PCA dimensionality reduction
3. **Classical ML** — Random Forest, Logistic Regression, soft-voting Ensemble
4. **Deep Learning** — CNN (2 convolutional blocks with dropout + dense classifier head), trained 10 epochs with Adam optimizer
5. **Evaluation** — accuracy / F1 comparison across all models, confusion matrices (classical vs. deep learning)
6. **Generative AI (bonus)** — TinyLlama-1.1B-Chat for auto-generated e-commerce product descriptions from synthetic metadata; semantic product search using `all-MiniLM-L6-v2` sentence embeddings + cosine similarity

## Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `TensorFlow / Keras` · `scikit-image` (HOG) · `scipy.ndimage` (Sobel) · `transformers` (TinyLlama) · `sentence-transformers`

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale images (60,000 train / 10,000 test), 10 balanced classes, 784 pixel features + label per image.

The dataset files (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`, `fashion_mnist_metadata.json`) are **not included in this repository** — `fashion-mnist_train.csv` alone is ~127 MB, over GitHub's 100 MB per-file limit, so it's excluded via `.gitignore`. See **Setup** below for how to get them.

## Repository Structure

```
fashion-mnist-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── fashion-mnist-classification.ipynb
└── data/                          # not tracked by git — see Setup
    ├── fashion-mnist_train.csv
    ├── fashion-mnist_test.csv
    └── fashion_mnist_metadata.json
```

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/fashion-mnist-classification.git
cd fashion-mnist-classification
```

**2. Create an environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Get the dataset**

Create a `data/` folder in the project root and place these files inside it:

- `fashion-mnist_train.csv` and `fashion-mnist_test.csv` — download from [Kaggle (Zalando Research)](https://www.kaggle.com/datasets/zalando-research/fashionmnist) or the [official Fashion-MNIST repo](https://github.com/zalandoresearch/fashion-mnist)
- `fashion_mnist_metadata.json` — synthetic product metadata used only for the Generative AI section; if you don't have this file, you can skip the last two cells of the notebook (everything up to and including the model evaluation will still run fine)

```
data/
├── fashion-mnist_train.csv
├── fashion-mnist_test.csv
└── fashion_mnist_metadata.json
```

**4. Run the notebook**

```bash
jupyter notebook fashion-mnist-classification.ipynb
```

Run the cells from top to bottom. The notebook reads data from the local `data/` folder by default (`FOLDER_PATH = "data/"` in the first code cell) — no Google Drive or Colab account needed.

> Want to run it on Google Colab instead? Upload the three data files to your own Google Drive, then in the first code cell replace `FOLDER_PATH = "data/"` with a Drive mount + path, e.g.:
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
> FOLDER_PATH = "/content/drive/MyDrive/<your-folder>/"
> ```

## Authors

- Gerard Bogunyà
- Alejandro Lachner
