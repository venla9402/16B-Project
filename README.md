# 16B-Project


PIC16B Project Proposal
Group Members: Qi Yang, HongYe Zheng, Vieno Wu


**Introduction/Abstracts**

In the world of finance, the ability to predict stock prices accurately holds immense value for portfolio management, risk assessment, and investment strategy development. Stock price prediction remains a critical challenge due to the complex and dynamic nature of financial markets, influenced by myriad factors including economic indicators, company performance, political events, and market sentiment.

This project is an attempt to predict stock price patterns utilizing data analytics tools and thus   benefit investors by enhancing decision making processes. As a group of students with a background in financial mathematics and modeling, this project aligns well with the group's academic focus and personal interests. 

**Resources:**

Data: The cornerstone of this project is historical stock price data. Access to high-quality high-frequency data is crucial for the accuracy of predictive models in finance. For the purpose of this project, the group will utilize a primary dataset found on Kaggle: https://www.kaggle.com/datasets/rafsunahmad/apple-stock-price 
In addition, the group will utilize daily stock price data from the official NASDAQ website, which includes open, high, low, close prices and volume. As the stock exchange where our interested stocks are traded, it offers extensive historical price data for a range of stocks, which can be accessed directly or through APIs such as yfinance in Python. Both Kaggle and NASDAQ will serve as the primary dataset for training and testing of the predictive models.

Computing Power: Given the computational intensive nature of machine learning tasks, particularly when dealing with large datasets or complex models. The group has the following computing power tools: Intel i9 processor for efficient data processing and modeling, 16GB of RAM for handling large datasets in memory for data manipulation and model training, RTX 4080 for any intensive computations, and cloud platforms like Google Colab for any additional computations.

**Software and Tools**:

Python will be the primary programming language for this project due its support for data analysis and machine learning through libraries such as Panda, Numpy, Scikit-learn.
Tensorflow and Pytorch for building and training complex models such as LSTM networks.
Jupyter Notebook for code development, testing, and data visualization and analysis.
For additional data needs or alternative data sources, tools and libraries such as BeautifulSoup or Scrapy might be used to scrape financial news websites or other stock-related information from any website.

**Summarize Previous Work**:

https://www.kaggle.com/code/seokhyokang/apple-stock-price-prediction-w-tensorflow-lstm

In the Kaggle project titled “Apple Stock Price Prediction with TensorFLow LSTM” by Seokhyo Kang, an LSTM model is implemented to forecast Apple’s stock prices. The model utilizes historical data, including daily prices and trading volumes, which were normalized to facilitate neural network training. LSTM is chosen for its proficiency in handling time series data, capable of capturing long-term dependencies in stock price movements. The project assesses model performance using Root Mean Square Error (RMSE), highlighting its ability to effectively track and predict stock price trends. However, it also illustrates the challenges of deep learning in stock prediction, such as sensitivity to hyperparameters and the tendency for overfitting, pointing out the need for careful model tuning and validation.

https://www.geeksforgeeks.org/stock-price-prediction-using-machine-learning-in-python/

The GeeksforGeeks tutorial presents a straightforward approach to stock price prediction using Logistic , SVC and XGB Classifiers . The models involve fetching historical stock data, which typically includes dates and closing prices, and then applying data preprocessing techniques to prepare for model training. In addition to existing features of the stock prices, the project also focused on feature engineering where specific domain knowledge is integrated into the model. For example, there is creation of features “end-of-year” and “end-of-quarter”, as well as events like product launches and major corporate announcements, which are known to affect stock prices.Later, a ROC-AUC curve is utilized as the performance metric for model’s predictive accuracy. However, it is found that among the three models, XGBClassifier has the highest performance but it is pruned to overfitting. Later a confusion matrix was constructed and found that the accuracy achieved by the state of the art ML model is no better than simply guessing with a probability of 50%. The possible reasons may be the lack of data or using a very simple model to perform such a complex task like Stock Market prediction, which again illustrates the challenges in stock prediction.
 
**Description of Learning Opportunities**:

Time Series Analysis: The group will enhance the understanding of the time series data, particularly focusing on how to manage and analyze sequential data such as stock-prices that are time-dependent.
Exploring Machine Learning Models: The group will gain hands-on experience with several predictive modeling techniques, including LSTM, and other more advanced deep learning models like CNNs, along with their strengths and limitations in the context of financial data.
Data Handling and Processing: The group will have the learning opportunity to effectively clean, preprocess and engineer features from raw stock price data. Along with dealing missing values, outliers and deriving new variables that could potentially improves models accuracy
Statistical Analysis: statistical methods for time series forecasting, comparing their performance against modern machine learning approaches. 
Error Analysis and Model Evaluation: Evaluating models using various metrics such as MAE, RMSE and other financial performance metrics
Python and Libraries: The group will strengthening the skills in using Pandas, Numpy, Tensorflow and Scikit-learn

**Tentative Timeline**:

After two weeks (Week 4 - Week 6): Establish a robust dataset through kaggle and by fetching historical stock price data, complete all complex data visualizations. 
After four weeks (Week 6 - Week 8) : Learn and implement machine learning models to predict stock prices and understand model evaluation metrics.
After six weeks (Week 8 - Week 10): Develop a user interactive dashboard using Python libraries like Dash or Plotly. The dashboard can display real-time predictions, historical data analysis, and model accuracy metrics, allowing users to observe how the predictions change. 

**Role of Each Member**:

Qi Yang: Qi is going to learn and take care of customizing the visualization of data patterns and performances of various models.
Vieno Wu: Vieno is going to learn about the caveats for each different time series model and conduct data preprocessing to prepare the data for the different models.
Hongye Zheng: Hongye is going to discuss the implications of the outcomes of each model and the significance of our project.



