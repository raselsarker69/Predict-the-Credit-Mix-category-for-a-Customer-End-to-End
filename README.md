# Predicting Credit_Mix Category for Customers

---

## **Project Overview**
This project involves building a machine learning model to predict the **Credit_Mix** category for customers based on their financial data. The goal is to identify patterns in financial behavior and provide actionable insights to improve credit health. By evaluating and comparing multiple classification algorithms, we aim to select the most effective model for accurate predictions.

---

## **Dataset Description**
The dataset includes customer financial data, such as:

- **Credit_Mix**: Target variable (categorical).
- **Features**:
  - Credit history
  - Debt-to-income ratio
  - Account balances
  - Number of credit inquiries
  - Payment history

### **Data Preprocessing**
1. **Handling Missing Data**:
   - Missing values were imputed using mean or median imputation.
2. **Scaling and Normalization**:
   - Numerical features were scaled using MinMaxScaler.
3. **Encoding Categorical Variables**:
   - Categorical features were encoded using one-hot encoding.
4. **Splitting Data**:
   - Dataset split into 80% training and 20% testing.

---

## **Models Evaluated**
The following machine learning models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost Classifier**
5. **CatBoost Classifier**
6. **Gradient Boosting Classifier**

### **Hyperparameter Tuning**
- GridSearchCV and RandomizedSearchCV were used to optimize hyperparameters for all models.
- Example for Random Forest:
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

---

## **Model Performance**

### **Evaluation Metrics**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Cross-Validation Accuracy**

| Model                 | Main Accuracy | Main Precision | Main Recall | Main F1 Score | CV Accuracy |
|-----------------------|---------------|----------------|-------------|---------------|-------------|
| Logistic Regression   | 0.6329        | 0.5446         | 0.5949      | 0.5491        | 0.6294      |
| Decision Tree         | 0.6301        | 0.6124         | 0.6033      | 0.6077        | 0.6369      |
| Random Forest         | **0.7687**    | 0.6305         | **0.7204**  | 0.6506        | **0.7657**  |
| XGBoost               | 0.7596        | **0.6238**     | 0.7106      | **0.6404**    | 0.7568      |
| CatBoost              | 0.7549        | 0.6183         | 0.7074      | 0.6343        | 0.7567      |
| Gradient Boosting     | 0.7501        | 0.6033         | 0.7020      | 0.6273        | 0.7522      |

### **Visualization Highlights**
- Bar chart of model performance metrics illustrates that Random Forest outperformed all other models in terms of accuracy and cross-validation scores.

---

## **Best Performing Model**
### **Random Forest Classifier**
- **Best Parameters**:
  - `max_depth`: None
  - `min_samples_leaf`: 1
  - `min_samples_split`: 2
  - `n_estimators`: 50
- **Best Cross-Validation Score**: 0.9667
- **Evaluation Metrics**:
  - **Accuracy**: 1.00
  - **Precision**: 1.00
  - **Recall**: 1.00
  - **F1-Score**: 1.00
  
---

## **Recommendations**

1. **Model Selection**:
   - Deploy the Random Forest Classifier as it offers the best performance for this task.

2. **Actionable Insights**:
   - Focus on improving customers' debt-to-income ratio and payment history, as these features strongly impact credit health.

3. **Feature Engineering**:
   - Identify additional features like employment stability and loan repayment history to improve model performance.

4. **Real-World Implementation**:
   - Build a dashboard to monitor predictions and provide customers with tailored recommendations to improve their credit health.

5. **Future Work**:
   - Experiment with ensemble techniques like stacking and blending for further performance enhancement.
   - Evaluate the robustness of the model on unseen data from different financial institutions.

---

## **How to Run the Project**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name/credit-mix-prediction.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the Data**:
   - Run `data_preprocessing.py` to clean and prepare the dataset.

4. **Train the Model**:
   - Use `train_model.py` to train and evaluate the machine learning models.

5. **Generate Predictions**:
   - Run `predict.py` to predict the Credit_Mix category for new customers.

6. **Visualize Results**:
   - Use `visualizations.py` to generate performance comparison charts.

---


## Contributing

Contributions are welcome! If you have ideas to improve this repository or want to add more projects, please feel free to:

1. Fork the repository.
2. Make your changes.
3. Submit a pull request.

---

## License
This repository is licensed under the MIT License. Feel free to use and modify the code as needed.

---

## Author
**Md. Rasel Sarker**  
Email: [rasel.sarker6933@gmail.com](mailto:rasel.sarker6933@gmail.com)  

<br>
<h1 align="left">
 <h2><img src = "https://media2.giphy.com/media/QssGEmpkyEOhBCb7e1/giphy.gif?cid=ecf05e47a0n3gi1bfqntqmob8g9aid1oyj2wr3ds3mg700bl&rid=giphy.gif" width=30px valign="bottom"> 🌐 Connect with Me:</h2>
</h1>

<p align="center">
  <a href="mailto:rasel.sarker6933@gmail.com"><img src="https://img.shields.io/badge/Email-rasel.sarker6933@gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/raselsarker69"><img src="https://img.shields.io/badge/GitHub-%40Raselsarker-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://www.linkedin.com/in/rasel-sarker-405160227/"><img src="https://img.shields.io/badge/LinkedIn-Rasel%20Sarker-blue?style=flat-square&logo=linkedin"></a>
  <a href="https://www.facebook.com/mdrasel.sarker.7773631"><img src="https://img.shields.io/badge/Facebook-%40Raselsarker-blue?style=flat-square&logo=facebook"></a>
  <a href="https://www.kaggle.com/mdraselsarker"><img src="https://img.shields.io/badge/Kaggle-%40Raselsarker-blue?style=flat-square&logo=kaggle"></a>
  <a href="https://www.youtube.com/@raselsarker69"><img src="https://img.shields.io/badge/YouTube-Rasel%20Sarker-red?style=flat-square&logo=youtube"></a>
  <a href="https://www.facebook.com/groups/832585175685301"><img src="https://img.shields.io/badge/Facebook%20Group-Rasel%20Sarker%20Group-blue?style=flat-square&logo=facebook"></a>
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801581528651-green?style=flat-square&logo=whatsapp">
</p>
 

---

<div align="center">

Thank you for visiting my repository. I hope these projects inspire and guide your learning journey!

---

Feel free to explore, learn, and build upon these projects. Happy coding!<br>

&copy; 2025 Machine Learning Projects

</div>
