Integration of machine learning XGBoost and SHAP models for NBA game outcome prediction and quantitative analysis

sample: NBA 2021-2023
Conducted a real-time prediction model for NBA game outcomes
integrating: ml XGBoost and SHAP algorithms

XGBoost algorithm was highly effective in predicting NBA game outcomes
KPI: FG%, DREB TO
First half assists affect the outcome of the game
Second half of games offensive rebounds and 3pt% were kpi affecting the outcome of games

Machine learning models a valuable tool for sports performance analysts
ML models are sometimes black box.

To overcome "black boxes:, The Shapley Additive exPlanations (SHAP) method is utilized to interpret machine learning models and visualize individual variable predictions.
construct a real time model
The innovation in there research lies in the use of a grouping stategy to build real time game prediction models.


This research introduces an innovative real-time prediction method for NBA game outcomes.

we designed two different approaches during the
model construction process: 
1. a real-time prediction model based on the technical performance indicators from the first two quarters and the first three quarters of the game, 
2. and a post-game prediction model constructed based on the full-game technical performance indicators.

XGBoost - stands for eXtreme Gradient Boosting, is an ensemble ml based on gradient boosting desicion trees. They use decision trees as weak learners, iteratively fitting the residuals of the predictions from the previous iteration, forming a strong learner through weighted aggregation. 

Machine learning algorithms have demonstrated excellent performance in predicting the outcomes of NBA games, but they also face the issue of being “interpretable”. We often cannot comprehend the decision-making process of machine learning algorithms, which is commonly referred to as the “black box” model.
SHAP interpretation algorithm, inspired by game theory. The purpose of this algorithm is to calculate the Shap Values for each feature, reflecting the contribution of different features to the model. This allows for the interpretation of the prediction mechanism of complex machine learning algorithm models from both local and global perspectives. 

4.1 Data Acquisition - basketball-reference.com
4.1.2 Reliability and validity of data - Two Basketball players watched 5 games each and verified a intraclass correlation coefficent (ICC) of 0.98
4.1.3 Data Preparation - The game outcome prediction problem was transformed into a binary classification issue. The target value result represented the binary classification label for the home team’s win or loss, with a home team win/loss converted to the numerical values 1/0, respectively. Additionally, the data from the first two quarters and the first three quarters of the game were summed up to create new feature variables, with the prefixes H2, H3, and game added to distinguish them. an approach was adopted where features representing identical technical indicators for both the home and away teams are subtracted from each other. This method helped to mitigate situations where the technical
statistics of the home and away teams were closely matched, reducing the interference of redundant information. Concurrently, it served to lower the dimensionality of the data, thereby enhancing the performance and efficiency of the predictive model.
Exploratory analysis was conducted by plotting a heatmap of feature correlations. The color of the heatmap indicates the correlation between two features: darker colors represent stronger
positive correlations, while lighter colors represent stronger negative correlations. The numerical values represent the correlation coefficients between corresponding features. The asterisks reflect the significance levels of the correlation coefficients: no asterisk denotes p >0.05, one asterisk denotes 0.01 <p <0.05, two asterisks denote 0.001 <p <0.01, and three asterisks denote p <0.001. Taking the full-game technical statistics as an example, the correlation heat-
map is shown in Fig 3.

Diiference In Scores was calculated as well as advanced stats and major stats differences between the home and away teams. 

To avoid the high correlation between feature variables affecting the model’s predictive performance, feature selection was conducted using a heatmap of feature correlations. The total rebounds were calculated as the sum of offensive and defensive rebounds, and the field goal percentage was determined as the ratio of field goal attempts to field goals made. Since the field goal percentage more accurately reflects the team’s strength and condition, features such as field goal makes, field goal attempts, two-pointer makes, two-pointer attempts, three-pointer makes, three-pointer attempts, free throw makes, free throw attempts, and total rebounds were removed. Additionally, irrelevant features from the basic game information were also eliminated, constructing a new set of features for the sample dataset. The describe function was used to perform descriptive statistics on the NBA sample data, and the analysis results are listed in Table 2 below.

To ensure the robustness of our predictive model and to enhance the reliability and
interpretability of our research findings, we employed logistic regression to test the significance of key performance Variables. Tables 3–5 present the logistic regression analysis results for different periods of the game. From the logistic regression results, it is evident that field goal percentage, two-point field goal percentage, three-point field goal percentage, free throw percentage, offensive rebounds, defensive rebounds, assists, personal fouls, and turnovers significantly impact the game outcome across different game periods. In contrast, blocks and steals do not significantly affect the game outcome. Additionally, personal fouls and turnovers have a negative impact on the
game outcome. Given the crucial role of blocks and steals in influencing game results and their confirmed importance in previous research, we decided to retain all 11 indicators [28, 33–35], including blocks and steals, across the three distinct time period datasets.

4.2 Model Training and Experimental Results

To enhance the practical significance and application value of the predictive model, this study grouped and merged the NBA game dataset according to the duration of the games, constructing datasets for the first two quarters, the first three quarters, and the full game to make real-time predictions of game outcomes at the end of the second and third quarters. An NBA game outcome prediction model was built based on the XGBoost algorithm. Hyperparameter tuning was conducted for the XGBoost algorithm and five other main-stream machine learning algorithms—KNN, LightGBM, SVM, Random Forest, Logistic Regression, and Decision Tree—using methods such as Bayesian optimization and grid search. The optimal predictive model architecture was obtained through a ten-fold cross-validation comparative
experiment, combined with evaluation metrics. Finally, the SHAP algorithm was introduced for interpretability analysis of the best model, to uncover the key factors that determine the outcome of the games.
This study employs five classification performance metrics—AUC, F1 Score, accuracy, precision, and recall—to evaluate the quality of the NBA game outcome prediction model. When a model demonstrates good performance across these metrics in comparative experiments, it is considered to have superior predictive capabilities. Accuracy, precision, and recall respectively reflect the performance of a predictive model in “being right”, “being precise”, and “being comprehensive”. The F1 Score combines precision and recall, reflecting the robustness of the model. The probabilistic meaning of the AUC value is the probability that, when ran-
domly selecting a pair of positive and negative samples, the positive sample will have a higher score than the negative sample. The AUC value ranges from 0 to 1, with a higher AUC indicating a higher predictive value of the model.

The classification prediction results for game outcomes are presented in the confusion matrix as shown in table 6. 
1. Games where the actual result was a home team win and were predicted as a home team win are True Positives (TP)
2. Games where the actual result was a home team loss but were predicted as a home team win are False Positives (FP)
3. Games where the actual result was a home team loss and were predicted as a home team loss are True Negative (TN)
4. Games where the actual result was a home team win but were predicted as a home team loss are False Negatives (FN)


accuracy = TP + TN / TP + FP + FN + TN
precision = TP / TP + FN
recall = TP / TP + FN
F1 score = 2 * preicision * recall / precision + recall

4.2.1 Ten-fold cross-validation comparative experiment - After hyperparameter tuning
using Bayesian optimization and grid search, the evaluation metric results of the predictive models for different time periods by various algorithms in the ten-fold cross-validation comparative experiments are listed in Tables 7–9 and shown in Fig 5: The XGBoost algorithm exhibits optimal performance in predicting the outcomes of NBA games. In terms of the AUC and F1 Score metrics, the XGBoost algorithm performed excellently, consistently ranking in the top 2 across the ten-fold cross-validation comparative experiments for the three different time periods. Regarding the accuracy and precision metrics, the XGBoost algorithm consistently showed the best performance. However, in terms of recall, the XGBoost algorithm
ranked 4th, 3rd, and 1st in the comparative experiments for the three different time periods, with recall values of 0.775, 0.807, and 0.939, respectively.

4.2.2 Analysis of factors influencing game outcomes at different time - SHAP provides powerful and diverse data visualization charts to demonstrate the interpretability of the model. Based on the XGBoost real-time game outcome prediction model discussed earlier, SHAP
quantifies and ranks the importance of features that influence the outcome of the games, as listed in Table 10. 
