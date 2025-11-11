# Temperature-Prediction-Using-Apache-Spark-MLib-

## Machine Learning on Weather Data Using Apache Spark MLib (PySpark + Dataproc)

This project implements a distributed end-to-end ML pipeline for predicting surface air temperature from the NOAA Integrated Surface Dataset (ISD). We use Apache Spark MLlib on Google Cloud Dataproc with data in Google Cloud Storage (GCS) to perform scalable preprocessing, feature engineering, model training, and hyperparameter tuning. 

The workflow evaluates three regressors—Generalized Linear Regression (GLR), Random Forest (RF), and Gradient Boosted Trees (GBT)—to compare accuracy, scalability, and interpretability in a distributed setting. After distributed preprocessing and tuning, GBT delivered the strongest performance, effectively capturing non-linear temperature patterns while remaining computationally efficient.

---

## Requirements

### Create a Bucket in Google Cloud Storage (GCS) to be used to store both raw and processed data

gs://dsa5208-mllib-proj/

This bucket serves as the central data repository for:
	•	Input meteorological datasets
	•	Preprocessed Parquet shards
	•	Model outputs and checkpoints


### Google Dataproc Cluster

Cluster configuration used for distributed training:

| **Component**       | **Specification**                                                                 |
|----------------------|-----------------------------------------------------------------------------------|
| Master Node          | 1 × n2-standard-4                                                                |
| Worker Nodes         | 2 × n2-standard-4                                                                |
| Optional             | 2–4 preemptible workers for elastic scaling during model tuning                  |
| Notebook Access      | Jupyter Notebook Component Gateway enabled for interactive PySpark development and visualization |

### Packages Used
Install the following Python dependencies on your Dataproc cluster or local environment before running the notebooks:

```bash
pip install pyspark pandas pyarrow numpy matplotlib seaborn
```

Optional (for Jupyter-based development):
```bash 
pip install notebook jupyterlab
```

---

## Project Files

| File | Description |
|------|--------------|
| `[Final] Processing of Raw Data for Machine Learning.ipynb` |Cleans the raw NOAA data and produces the preprocessed dataset (in parquet format) used for all models. |
| `[Final] GLR_Model.ipynb` | Implements the Generalized Linear Regression model with a full Spark ML pipeline. |
| `[Final] RandomForestRegressor.ipynb` | Implements the Random Forest Regression model with a full Spark ML pipeline. |
| `[Final] GBT Model.ipynb` | Implements the Gradient Boosted Trees model with a full Spark ML pipeline. |

---

## Step-by-Step Execution

### Step 0 — Set up storage & cluster (one-time)
	1.	Create a GCS bucket (stores raw data, cleaned Parquet, and outputs): 
        gs://dsa5208-mllib-proj/
  
    2. Start a Dataproc cluster (with Jupyter Component Gateway enabled):
	•	1 × master (n2-standard-4)
	•	2 × workers (n2-standard-4)
	•	(Optional) 2–4 preemptible workers for tuning

### Step 1 — Preprocess the NOAA ISD CSV datasets

Run the preprocessing notebook to clean and structure the raw NOAA ISD data.
This removes sentinel values, handles missing data, and performs feature engineering (e.g., date_numeric).

In Jupyter (Component Gateway)
	1.	Open: [Final] Processing of Raw Data for Machine Learning.ipynb
	2.	Run all cells.
	3.	Output: gs://dsa5208-mllib-proj/processed/df_cleaned.parquet

This will:
	•	Parse and split multi-field columns (e.g., WND/CIG/VIS/SLP)
	•	Remove sentinel tokens (e.g., 99, 9999, +9999)
	•	Impute missing values where applicable
	•	Create feature columns (dew_cel, slp_hpa, wnd_speed, vis_dist, cig_height, date_numeric)
	•	Save cleaned shards to Parquet for fast distributed reads
    
### Step 2 — Run the (1) Generalised Linear Regression Model, (2) Random Forest Regression Model, (3) Gradient Boosted Tree Model with PySpark

For each model, in Jupyter
	1.	Open the named model file; i.e. [Final] GLR_Model.ipynb OR [FINAL]RandomForestRegressor.ipynb OR [Final] GBT Model.ipynb
	2.	For each notebook, load in the processed/cleaned Parquet via df = spark.read.parquet("gs://dsa5208-mllib-proj/processed/df_cleaned.parquet")
    3.	Run all cells.

---

## Outputs

Each model notebook (.ipynb) generates the following outputs to enable comparison and evaluation of the best-performing model type for predicting surface air temperature from the NOAA Integrated Surface Dataset (ISD):
	•	Test metrics: RMSE, MAE, and R²
	•	RMSE heatmap: Visualizing model performance across hyperparameter combinations (e.g., regParam × maxIter)
	•	Top predictors bar chart: Showing the five most influential features and their impact on air temperature
	•	Residual diagnostics plot: Assessing model fit, bias, and error distribution
    
---
**Author:** Akshat Atul Bhargava, Choo Li Ying, Goh Chi Min
**Course:** DSA5208 – Scalable Distributed Computing for Data Science  
**Institution:** National University of Singapore  




