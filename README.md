# Pic16B Project: Time Series Forecasting with Deep Learning Models


Group Members: Qi Yang, HongYe Zheng, Vieno Wu

This project is about time series forecasting with deep learning models. We predicted the closing price
of Apple stock with FeedForward Neural Network, Convolutional Neural Network, Simple Recurrent Neural Network, 
LSTM, and GRU. The performances are detailed below:

FeedForward Neural Network:       Testing MSE: 32.007712738013936 

Convolutional Neural Network:     Training MSE: 0.0001688430995534914, 
                                  Testing MSE: 0.0005194461806446983
                                  
Simple Recurrent Neural Network: Training MSE: 0.00028258335874879585,
                                  Testing MSE: 0.0008784622312902721
                                  
GRU(Gated Recurrent Unit) :      Training MSE: 0.0001306275336723546,
                                  Testing MSE: 0.00039311261364430875
                                  
LSTM(Long Short Term Memory) :    Training MSE: 0.0005805156894650927,
                                  Testing MSE: 0.0013470245214053241
                                  
The results show that CNN, RNN, GRU, and LSTM all performed similarly well, indicating their ability to capture intricate details in non-stationary data. 
Despite these comprehensive efforts, overfitting persisted, which can be attributed to the inherent challenges of financial time series data. Stock prices are highly volatile and influenced by numerous external factors, making them difficult to predict accurately. The available historical data may not be sufficient to capture long-term trends, and financial data often contains noise and anomalies that hinder model performance.  In conclusion, while significant strides were made to address overfitting, it remains a persistent challenge in stock price prediction. Future research should explore more advanced models, incorporate alternative data sources, and enhance feature engineering techniques. Collaboration with domain experts could provide deeper insights into market dynamics, aiding in the development of more robust predictive models.


 

                         
                            


