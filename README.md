# Real Time Stock Market Forecasting

This repository contians an implementation of ensemble deep learning models to forecast or predict stock price. We use Alpha Vantage API
to pull stock data(open,high,low,close,volume) and scrape news headlines from inshorts to perform setiment analysis.

The code and the images of this repository are free to use as regulated by the licence and subject to proper attribution:
```
Shah, Raj and Tambe, Ashutosh and Bhatt, Tej and Rote, Uday, Real-Time Stock Market Forecasting using Ensemble Deep Learning and Rainbow DQN. 
Available at SSRN: https://ssrn.com/abstract=3586788 or http://dx.doi.org/10.2139/ssrn.3586788.
```

## Getting Started

It would be a better idea to create a conda environment and work in isolation 

- Create a virtual environment
```
conda create -n envname python=3.6.8 anaconda 
conda activate envname
```
Use ```conda deactivate``` to deactivate the environment
- Clone this repository
```
git clone --depth 1 https://github.com/THINK989/Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN.git && cd Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN
```
- Install the requirements
```
pip install -r requirements.txt
```
- Run python script

To vizualize the forecast 
```
python animate.py
```
To get heatmap visualization for correlation analysis on ^NSEI(Nifty50)
```
python heatmap.py
```

## License 

This repository is distributed under [MIT License](LICENSE)

and 

respective LICENSE under directories
```
forecast/LICENSE
or 
rainbow/LICENSE
```
for thier independent code usage. 

## Acknowledgement

- [@huseinzol05](https://github.com/huseinzol05/) for deep learning models
- [@sentdex](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ) for plotting tutorials
- [Vader Sentiment](https://github.com/cjhutto/vaderSentiment)
- [Alpha Vantage API](https://www.alphavantage.co/) for stock data
- [Inshorts](inshorts.com) for news headlines



























