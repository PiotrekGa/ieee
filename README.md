## Lessons learned

### General
* Spend more time on feature engineering
* Better local validation
* Use script except from Notebook

### Validation
* Use at most 6 folds in Cross-Validation
* Implement `Fast Cross Validation` algorithm in Python
* Use default Catboost as baseline
* Use negative downsampling to create fast new feature validation: [discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/108616#latest-634925)
* Perform all the 

### Data
* Minify data (function ready)

### Features
* Try engineering all the features: [discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/107441)
* Alternatively perform feature selection
* Use Target Encoding or Frequency encoding for categorical features
* Impute NA data with mean and add binary column indicating where it was added
* Remember to impute on training part only in CV
* Remove low frequency categories (experiment with that)
* Remove low / high occuring categories [kernel](https://www.kaggle.com/bogorodvo/lightgbm-baseline-model-using-sparse-matrix)
* Try removing correlated data (Spearman correlation for trees)
* Change pandas `Object` into `Category`

### Training
* Use GPU for gradient boosting (Kaggle / Colab / Cloud)
* Explore Catboost
