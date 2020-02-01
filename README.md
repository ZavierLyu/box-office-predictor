# Box Office Predictor
A project of box office predictor for EIE3280 in CUHKSZ.



## Baseline Structure 

![structure](./assets/eie3280.png)

## Report
Here is the report of this project where you can find more details.
[Box Office Predictor Project](report\Group5_report.pdf)

### NLP tools
- [Awesome Chinese NLP](https://github.com/crownpku/Awesome-Chinese-NLP)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)

## Plotly Dash Application
Here we wanna build a dash application for the visualization of the predictions.

### Steps to execute
Steps to run in local:
```shell
# Enter the directory
cd box-office-predictor
# Install the env management tool pipenv
pip install pipenv
# Install dependencies
pipenv install
# Enter the virtual environment
pipenv shell
# Open the jupyter book
python application.py
```
### How to deploy on AWS?
We use AWS Elastic Beanstalk service to deploy the web application. We recommend [Elastic Beanstalk CLI](https://docs.amazonaws.cn/elasticbeanstalk/latest/dg/eb-cli3-install.html). 

Steps to deploy:
```shell
eb init -p python-3.6 box-office-predictor #Initialize the repository
eb init # Set the configurations
...
eb create box-office-predictor-env # Deploy the web, upload whole application files
eb open # Open the web on browser
```

## Materials
- [Dash Tutorial](https://pythonprogramming.net/data-visualization-application-dash-python-tutorial-introduction/)
- [Dash Official Doc](https://dash.plot.ly/)
- [Python Machine Learning Materials](https://github.com/ageron/handson-ml)
- [Social Sentiment](https://github.com/Sentdex/socialsentiment)

## Reference

- [Prediction of Movie Success using Sentiment
Analysis of Tweets](http://www.jscse.com/papers/vol3.no3/vol3.no3.46.pdf)

- [Dynamic Box Office Forecasting Based on Microblog Data](https://www.jstor.org/stable/pdf/24899494.pdf?refreqid=excelsior%3Aa3a89bc298c2c0a9c141ce4b02f3cead)