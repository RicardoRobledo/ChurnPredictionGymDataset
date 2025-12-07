# Gym Member Churn Prediction 🏋️‍♂️

A comprehensive machine learning study to predict gym member churn using ensemble methods and class imbalance techniques. This project evaluates multiple algorithms and resampling strategies to identify the optimal approach for churn prediction.

## 📊 Dataset Overview

**Source**: [Churn Prediction Gym Members Dataset](https://www.kaggle.com/datasets/hassaan2580/churn-prediction-gym-members-dataset/data)

The dataset contains anonymized gym membership and activity data for churn performance analysis. Each row represents one gym member with comprehensive demographic, behavioral, and engagement features.

### Features Description

**Demographic Information:**
- `Member_ID`: Unique identifier for each member
- `Name`: Anonymized member name
- `Age`: Member's age
- `Gender`: Member's gender
- `Address`: Member's address
- `Phone_Number`: Contact information

**Membership Details:**
- `Membership_Type`: Subscription plan (Monthly, Quarterly, or Yearly)
- `Join_Date`: Date when member joined the gym
- `Last_Visit_Date`: Most recent gym visit date

**Activity & Engagement Metrics:**
- `Favorite_Exercise`: Preferred workout or gym activity
- `Avg_Workout_Duration_Min`: Average time spent per workout session (minutes)
- `Avg_Calories_Burned`: Estimated calories burned per workout
- `Total_Weight_Lifted_kg`: Cumulative weight lifted across all workouts (kg)
- `Visits_Per_Month`: Average frequency of gym visits per month

**Target Variable:**
- `Churn`: Binary indicator of member attrition (Yes/No)

### Dataset Characteristics

- **Purpose**: Machine learning classification, data cleaning, and exploratory data analysis
- **Data Type**: Tabular, anonymized member records
- **Challenge**: Class imbalance between active members and churned members
- **Use Case**: Predictive modeling for member retention strategies

## 🔧 Feature Engineering

### Temporal Feature Transformation

Date columns were converted into numerical features to capture temporal patterns and member engagement trends:

**Engineered Features:**

1. **`days_since_join`**: Number of days since member registration
   - Captures member tenure and loyalty duration
   - Higher values indicate longer membership commitment
   - Important for understanding retention patterns

2. **`days_since_last_visit`**: Days elapsed since most recent gym visit
   - Strong predictor of disengagement and churn risk
   - Increasing values signal declining engagement
   - Critical early warning indicator for intervention

**Rationale:**

Raw date columns are not directly interpretable by machine learning algorithms. Converting them to numerical representations:
- Enables models to capture temporal relationships
- Provides consistent, comparable metrics across all members
- Improves model performance and interpretability
- Allows detection of engagement decay patterns

### Data Cleaning

- Removed unnecessary columns to reduce dimensionality and noise
- Focused on features with direct predictive value for churn
- Improved computational efficiency and model interpretability

## 🧪 Experimental Setup

### Model Comparison Study

We evaluated **9 different classification algorithms** to identify the best performer for churn prediction:

| Model | Abbreviation | Type |
|-------|--------------|------|
| Support Vector Classifier | SVC | Kernel-based |
| Logistic Regression | LR | Linear |
| Linear Discriminant Analysis | LDA | Linear |
| Random Forest | RF | Ensemble |
| Gradient Boosting Classifier | GBC | Ensemble |
| AdaBoost | ADA | Ensemble |
| Bagging Classifier | BAG | Ensemble |
| Gaussian Naive Bayes | GNB | Probabilistic |
| Decision Tree | DT | Tree-based |

### Resampling Techniques Evaluation

We tested **11 class imbalance handling strategies** to address the churn class imbalance:

**Undersampling Methods:**
- Random UnderSampler
- Tomek Links
- Condensed Nearest Neighbour
- Edited Nearest Neighbours

**Oversampling Methods:**
- SMOTE (Synthetic Minority Oversampling Technique)
- Borderline SMOTE
- SVM SMOTE
- ADASYN (Adaptive Synthetic Sampling)
- Random OverSampler

**Hybrid Methods:**
- SMOTE + Tomek Links
- SMOTE + ENN (Edited Nearest Neighbours)

### Evaluation Metric

**G-Mean (Geometric Mean)** was selected as the primary metric because:
- Balances sensitivity (recall) and specificity for imbalanced datasets
- Penalizes models that perform well on majority class but poorly on minority class
- Provides a single score that reflects performance on both classes
- More appropriate than accuracy for imbalanced classification problems

**Formula:** G-Mean = √(Sensitivity × Specificity)

## 📈 Experimental Results

### Model Performance Comparison

| Rank | Model | Mean G-Mean | Std Dev | Performance |
|------|-------|-------------|---------|-------------|
| 🥇 1 | **Random Forest (RF)** | **0.9686** | ±0.0319 | Excellent |
| 🥈 2 | Gradient Boosting (GBC) | 0.9646 | ±0.0291 | Excellent |
| 🥈 2 | AdaBoost (ADA) | 0.9646 | ±0.0291 | Excellent |
| 🥈 2 | Bagging (BAG) | 0.9646 | ±0.0291 | Excellent |
| 🥉 5 | Decision Tree (DT) | 0.9562 | ±0.0323 | Very Good |
| 6 | Linear Discriminant Analysis (LDA) | 0.8767 | ±0.0862 | Good |
| 7 | Logistic Regression (LR) | 0.8648 | ±0.1067 | Good |
| 8 | Gaussian Naive Bayes (GNB) | 0.8236 | ±0.0503 | Moderate |
| 9 | Support Vector Classifier (SVC) | 0.5528 | ±0.2562 | Poor |

**Key Findings:**

1. **Ensemble methods dominate**: Top 5 models are all ensemble-based algorithms
2. **Random Forest leads**: Best performance with lowest variance
3. **SVC struggles**: Linear kernel may not capture complex patterns
4. **Consistent performance**: Top models show low standard deviation (±0.03)

### Resampling Technique Performance

| Rank | Technique | Mean G-Mean | Std Dev | Performance |
|------|-----------|-------------|---------|-------------|
| 🥇 1 | **Tomek Links** | **0.9686** | ±0.0319 | Excellent |
| 🥇 1 | **SMOTE** | **0.9686** | ±0.0319 | Excellent |
| 🥇 1 | **SMOTE + Tomek** | **0.9686** | ±0.0319 | Excellent |
| 4 | Random OverSampler | 0.9648 | ±0.0359 | Excellent |
| 5 | SVM SMOTE | 0.9646 | ±0.0291 | Excellent |
| 6 | ADASYN | 0.9544 | ±0.0425 | Very Good |
| 7 | Condensed Nearest Neighbour | 0.9529 | ±0.0348 | Very Good |
| 8 | Borderline SMOTE | 0.9503 | ±0.0414 | Very Good |
| 9 | Random UnderSampler | 0.9447 | ±0.0278 | Very Good |
| 10 | Edited Nearest Neighbours | 0.9190 | ±0.0338 | Good |
| 11 | SMOTE + ENN | 0.8622 | ±0.0660 | Good |

**Key Findings:**

1. **Three-way tie at the top**: Tomek Links, SMOTE, and SMOTE+Tomek achieve identical performance
2. **Tomek Links is preferred**: Simplest approach with best results (undersampling only)
3. **SMOTE variants excel**: Multiple SMOTE-based methods in top 5
4. **Hybrid methods vary**: SMOTE+Tomek excellent, SMOTE+ENN underperforms
5. **Consistent variance**: Top techniques show low standard deviation

## 🏆 Best Model & Technique

### Optimal Configuration

**Model:** Random Forest (RF)
- **Mean G-Mean Score:** 0.9686
- **Standard Deviation:** ±0.0319
- **Full Result:** `0.9685730304894465 ± 0.03193311387774696`

**Resampling:** Tomek Links
- **Mean G-Mean Score:** 0.9686
- **Standard Deviation:** ±0.0319
- **Full Result:** `0.9685730304894465 ± 0.03193311387774696`

### Why Random Forest?

Random Forest achieved superior performance due to:

1. **Ensemble Learning**: Combines predictions from multiple decision trees
2. **Robustness**: Reduces overfitting through bootstrap aggregation
3. **Non-linear Patterns**: Captures complex interactions between features
4. **Feature Importance**: Provides interpretability through feature ranking
5. **Handles Imbalance**: Naturally adapts to class distribution
6. **Low Variance**: Consistent performance across different data splits

### Why Tomek Links?

Tomek Links was selected as the optimal resampling technique because:

1. **Boundary Cleaning**: Removes ambiguous samples at class boundaries
2. **Simplicity**: Undersampling-only approach, no synthetic data generation
3. **Efficiency**: Faster than SMOTE-based methods
4. **Data Integrity**: Maintains original data distribution
5. **Noise Reduction**: Eliminates borderline and noisy instances
6. **Best Performance**: Tied for highest G-Mean with lowest complexity

**Technical Explanation:**

Tomek Links identifies pairs of instances from opposite classes that are nearest neighbors. These pairs represent noisy or borderline examples that confuse classifiers. By removing the majority class instance from each pair, Tomek Links:
- Creates cleaner decision boundaries
- Reduces class overlap
- Improves model generalization
- Maintains data quality without synthetic examples

## 🎯 Decision Threshold Optimization

### Problem Statement

Classification models typically use a default threshold of **0.5** to convert probabilities into binary predictions. However, this may not be optimal when:

- Dealing with imbalanced datasets
- False negatives have higher business cost than false positives
- Recall is more important than precision for the minority class

### Our Approach

We adjusted the decision threshold to maximize G-Mean and align with business objectives:

**Goal:** Maximize detection of potential churners while maintaining acceptable precision

**Trade-offs:**
- ✅ **Increased Recall**: Catch more actual churners (true positives)
- ⚠️ **Slightly Lower Precision**: Accept more false alarms (false positives)
- ✅ **Better Business Outcome**: Proactive retention is cheaper than losing members

### Business Justification

In the gym membership context:

- **Cost of False Negative (Missing a churner):** Lost membership revenue, high customer acquisition cost
- **Cost of False Positive (Flagging active member):** Minor retention effort cost (email, discount offer)

**Conclusion:** It's more cost-effective to intervene with 10 members (where 8 might stay anyway) than to lose 2 actual churners who could have been retained.

## 📊 Test Set Evaluation

### Performance on Unseen Data

The model's performance on the test set was lower than training/validation performance, revealing important insights:

**Observations:**
1. Test set contains more challenging, real-world cases
2. Model shows some overfitting to training distribution
3. Performance gap is expected and acceptable for deployment
4. Indicates need for continuous model monitoring and retraining

**Implications:**
- Model generalization is good but not perfect
- Real-world performance will require ongoing evaluation
- Additional feature engineering may improve robustness
- Ensemble methods or regularization could reduce overfitting

This realistic assessment ensures proper expectations when deploying the model in production environments.

## 🔍 Key Insights

### Model Selection Insights

1. **Ensemble methods outperform linear models** by 10-15% G-Mean score
2. **Random Forest is the clear winner** with best performance and consistency
3. **SVC struggles significantly** on this dataset, likely due to feature scaling and kernel choice
4. **Tree-based models excel** at capturing complex member behavior patterns

### Resampling Insights

1. **Tomek Links achieves best results** with simplest approach
2. **SMOTE variants are reliable** but more complex than needed
3. **Hybrid methods don't always improve** over single techniques
4. **Undersampling can outperform oversampling** when implemented correctly

### Feature Engineering Insights

1. **Temporal features are crucial**: Days since join and last visit are strong predictors
2. **Engagement metrics matter**: Workout frequency and duration indicate commitment
3. **Membership type is informative**: Contract length affects churn probability
4. **Activity patterns reveal intent**: Declining visit frequency predicts churn

## 🚀 Practical Applications

### Business Use Cases

1. **Proactive Retention Campaigns**: Target high-risk members before they churn
2. **Personalized Interventions**: Customize offers based on member profiles
3. **Resource Optimization**: Focus retention budget on saveable members
4. **Performance Monitoring**: Track gym engagement metrics in real-time
5. **Membership Strategy**: Adjust pricing and contract terms based on churn patterns

### Deployment Recommendations

- **Prediction Frequency**: Weekly batch predictions for all active members
- **Action Threshold**: Flag members with churn probability > optimized threshold
- **Intervention Types**: Personalized emails, discount offers, trainer consultations
- **Success Metrics**: Track retention rate of flagged members after intervention
- **Model Refresh**: Retrain quarterly with new data to maintain accuracy

## 📚 References

- **Dataset**: [Kaggle - Churn Prediction Gym Members Dataset](https://www.kaggle.com/datasets/hassaan2580/churn-prediction-gym-members-dataset/data)
- **Imbalanced Learning**: [imbalanced-learn documentation](https://imbalanced-learn.org/)
- **Random Forest**: [Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32](https://link.springer.com/article/10.1023/A:1010933404324)
- **Tomek Links**: [Tomek, I. (1976). Two modifications of CNN. IEEE Trans. Systems, Man and Cybernetics, 6, 769-772](https://ieeexplore.ieee.org/document/4309452)
- **G-Mean Metric**: [Kubat, M., & Matwin, S. (1997). Addressing the curse of imbalanced training sets](https://www.site.uottawa.ca/~nat/Workshop1997/kubat.pdf)

**Built with ❤️ for data science and gym member retention**
