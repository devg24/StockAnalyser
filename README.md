# StockAnalyser

This is an ongoing project I'm working on to learn more about stock market prices and investing

This program basically gauges the effect of external market factors like unemployment rate, GDP etc on the prices of different stocks and then decides which sector to invest in using that information

## Components and Current Road-Map

- The program will use company tickers and exchange symbols dataset I found on kaggle to get the sectors and names of over 20,000 companies
- The program will then use **yfinance** API to get the Adjusted closing price of each company over a certain period of time
- **FRED** API's python source **fredpy** will be used to make calls to the API and get market data
- **PCA** will be performed for dimension reduction of those external factors.
- Probably a **Machine Learning Model** will be used to learn the effect and/or weights of the factors to understand their influence on stock prices. 


