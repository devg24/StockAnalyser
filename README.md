# StockAnalyser

This is an ongoing project I'm working on to learn more about stock market prices and investing

Since there are numerous extremely volatile factors that effect the share market, I've decided to begin with evaluating relatively non valatile factors and their influence on the stock market.

This program basically gauges the effect of the main economic indicators of a country like unemployment rate, GDP etc on the prices of different stocks and then decides which sector to invest in using that information

## Components

- The program uses company tickers and exchange symbols dataset I found on kaggle to get the sectors and names of over 20,000 companies
- The program then uses **Alpha Vintage** API to get the monthly Adjusted closing price of each company over a certain period of time
- **FRED** API's python source **full_fred** is used to make calls to the API and get main economic indicators of the country which consists of more than 6000 metrics
- **PCA** is performed for dimension reduction of those metrics to get enough components to capture 90% of the variance in the data.
- The data is then put into a neural network to learn the effect and/or weights of the factors and understand their influence on stock prices. 

## Current Progress

- The neural network is able to perofrm at a 92% accuracy.

## Future plans

- Include visualisations to better see the relationships between the inputs and outputs
- Add a Frontend to make the program more user friendly
- Classify the output based on sectors.


