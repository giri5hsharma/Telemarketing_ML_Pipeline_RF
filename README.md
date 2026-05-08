# Bank Telemarketing Conversion Prediction & Optimization

> An end-to-end ML pipeline using Random Forest to predict telemarketing conversions, neutralizing an 88:12 class imbalance to optimize campaign targeting.

This repository contains an end-to-end Machine Learning classification pipeline designed to optimize banking telemarketing campaigns. By leveraging a Random Forest classifier and probability-ranking algorithms, this system prioritizes high-value leads to maximize subscription rates while significantly cutting operational costs.

## 🚀 Business Impact
* **70% Cost Reduction:** By utilizing probability thresholding, the model allows the business to stop targeting the bottom 70% of low-probability leads, drastically reducing wasted call center resources.
* **85% Capture Rate:** The algorithm successfully captures approximately 85% of all potential subscribers by contacting only the top 30% of the ranked prospect list.

## 🧠 Technical Highlights
* **End-to-End Pipeline:** Automated data cleaning, sentinel value extraction (e.g., transforming the `pdays` column), and feature engineering using Python, Scikit-Learn, and Pandas.
* **Imbalance Handling & Leakage Prevention:** Addressed a severe 88:12 class imbalance using balanced class weights and Stratified Cross Validation. Strictly removed post-call data leakage features (like call duration) to ensure real-world predictive reliability.
* **Model Optimization:** Implemented rigorous hyperparameter tuning and decision threshold optimization to maximize the harmonic mean (F1-score) and precision-recall trade-offs.

## 📊 Performance Metrics
The model was evaluated against a 20% stratified holdout test set:
* **Test AUC-ROC:** 0.805
* **Average Precision (PR):** 0.454 (A 3.8x improvement over the 11.7% random baseline)
* **Optimized F1-Score:** 0.487 (at the optimal 0.51 decision threshold)

## 🛠️ Tools & Libraries
* **Language:** Python 3.10
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/giri5hsharma/Telemarketing_ML_Pipeline_RF.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook `RandomForest_Telemarketing.ipynb` to view the pipeline execution, evaluation metrics, and cumulative gains chart generation.

## 📂 Project Structure
* `RandomForest_Telemarketing.ipynb`: The main notebook containing the EDA, preprocessing, training, and evaluation logic.
* `bank_rf_model_report.png`: The consolidated visualization dashboard containing the ROC curve, Confusion Matrix, and Cumulative Gains chart.
