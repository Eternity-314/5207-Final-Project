
# Local, Global, and Sequential: Mechanistic Comparison of Deep Learning Paradigms Under Multimodal Noise Corruption

This project is a cross-modal deep learning model evaluation benchmark designed to horizontally compare the performance of classic architectures (MLP, CNN, ResNet) and cutting-edge architectures (ViT, Transformer, Mamba, KAN) across various tasks and extreme scenarios. The evaluation covers three core dimensions: **Image Robustness**, **Time Series Forecasting**, and **Extreme Long Text Memory**.

## Code Base Structure

```text
├── README.md                # This documentation
├── notebooks/               # Core execution scripts 
│   ├── 1_Image_Robustness.ipynb        
│   ├── 2_TimeSeries_Forecasting.ipynb 
│   └── 3_LongText_Memory.ipynb         
├── outputs/                            
│   ├── image_results/                  
│   ├── time_series_results/            
│   └── long_text_results/              
└── saved_models/
└── AI - Gemini.pdf                      
```

## Dataset Preparation

The advantage of this project lies in its fully automated data preparation process, eliminating the need to manually download or clean massive data files:

1. **Image Dataset (CIFAR-10):** When running `1_Image_Robustness.ipynb`, `torchvision.datasets` automatically downloads CIFAR-10 to the local `./data` directory and automatically applies custom perturbation transformations (salt-and-pepper noise, spatial occlusion, and sketch stylization).
2. **Time Series Dataset (Yahoo Finance):** When running `2_TimeSeries_Forecasting.ipynb`, the `yfinance` library fetches real historical trading data for the S&P 500 index (`^GSPC`) from the past 10 years in real-time via API, and completes sliding window feature engineering in-memory (constructing features such as MA5 and MA20).
3. **Long Text Dataset (PG19):** When running `3_LongText_Memory.ipynb`, HuggingFace's `datasets` library loads the PG19 book dataset in a streaming manner, automatically extracting text blocks exceeding 50,000 words for the "needle in a haystack" experiment without taking up massive amounts of local hard drive space.

## Reproducing Results

Launch the Jupyter environment：
```bash
jupyter notebook
```
Open the `notebooks/` folder in your browser and run them in the following order or according to your research needs:

### Experiment 1: Image Robustness and Interpretability Testing
* **Run Script**: `notebooks/1_Image_Robustness.ipynb`
* **Core Objective**: Test the performance of lightweight models (~1.7M parameters) such as MLP, CNN, and ViT under four scenarios (clean, noise, occlusion, and sketch).
* **Reproduce Core Results**: 
  1. After running, it will output a dictionary file `evaluation_results.json` (containing Acc, F1, and ECE calibration error).
  2. Automatically generate a multi-dimensional chart **`comprehensive_robustness_analysis.png`** (bar chart, radar chart, and efficiency scatter plot).
  3. Automatically generate a feature attention heatmap matrix **`xai_saliency_all_scenarios.png`**.

### Experiment 2: Financial Time Series Trend Forecasting
* **Run Script**: `notebooks/2_TimeSeries_Forecasting.ipynb`
* **Core Objective**: Compare the convergence capabilities of TS_MLP, TS_CNN, TS_Transformer, TS_Mamba, and TS_KAN on time series data.
* **Reproduce Core Results**: 
  1. The script will output a precise model parameter count comparison table.
  2. Automatically calculate the **Mean Squared Error (MSE)** and **Directional Accuracy (DA%)** for each model on the test set.

### Experiment 3: Extreme Long Text "Needle in a Haystack" Memory Test
* **Run Script**: `notebooks/3_LongText_Memory.ipynb`
* **Core Objective**: Utilize a text pixelation strategy to unify input dimensions and test the feature retrieval engine performance of PureMamba, ViT, ResNet50, and MLP across length spans up to 16,384.
* **Reproduce Core Results**: 
  1. The console will output a detailed "PG19 Long Text Memory Test Experiment Report" (including specific hit rate percentages).
  2. Automatically generate a memory decay curve chart **`memory_accuracy_curve.png`** locally.
```
