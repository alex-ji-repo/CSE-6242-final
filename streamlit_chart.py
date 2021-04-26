# import all libraries
import yfinance as yf
import numpy as np
from fbprophet import Prophet
import pandas as pd
from datetime import date, timedelta, datetime
from textblob import TextBlob
import streamlit as st
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

df_bitcoin = yf.download(tickers="BTC-USD", period="10y", interval="1d")
st.write(
    """# Team 077 Final Project
    By:
    - Peter Chang
    - Alex Ji
    - Jose Lozano
    - Jonhua Qin
"""
)

# add date slider
time_box = st.slider(
    "Select your forecast period:",
    value=(date(2020, 7, 1), date(2021, 3, 17)),
    min_value=date(2020, 7, 1),
    max_value=date(2021, 3, 17),
)
print(time_box[0].strftime("%Y-%m-%d"))

df_bitcoin["pct_change"] = df_bitcoin["Adj Close"].pct_change()
df_bitcoin = df_bitcoin[1:]
# subset past daily dataframe
df_train = df_bitcoin[df_bitcoin.index <= pd.to_datetime("2020-12-31")]
# create future daily dataframe
df_test = df_bitcoin[
    (df_bitcoin.index >= pd.to_datetime(time_box[0].strftime("%Y-%m-%d")))
    & (df_bitcoin.index <= pd.to_datetime(time_box[1].strftime("%Y-%m-%d")))
]
time_lst = df_test.index.tolist()
df_future = pd.DataFrame(time_lst)
df_future.columns = ["ds"]


def get_forecast_model(
    df,
    df_future,
    daily_seasonality=False,
    weekly_seasonality=False,
    seasonality_mode="additive",
):
    df_train = df["pct_change"].reset_index()
    df_train.columns = ["ds", "y"]
    df_train["ds"] = pd.to_datetime(df_train["ds"])
    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        seasonality_mode=seasonality_mode,
    )
    model.add_country_holidays(country_name="US")
    model.fit(df_train)
    df_forecast = model.predict(df_future)
    return model, df_forecast


model_daily, forecast_daily = get_forecast_model(
    df_train, df_future, daily_seasonality=True
)
fig1 = model_daily.plot(forecast_daily)


# time series charts
st.write(fig1)

# add descriptino of the time series charts
st.write(
    """
    Figure 1: Using the Facebook Prophet time series library, with adjustment for daily seasonality and holidays,
    the 0% prediction is consistently within the 95% confident interval. Therefore, the traditional time series 
    analysis is not able to statistically diffentiate the up or down next day prediction.
"""
)

df_class = pd.read_csv("CODE/classification_result.csv")
df_class["stack_model"] = df_class.apply(
    lambda x: 1
    if (
        x["Logistic Regression"]
        + x["Random Forest"]
        + x["Gradient Boosting"]
        + x["KNN"]
        + x["Decision Tree"]
    )
    > 3
    else 0,
    axis=1,
)


dict_lst = []
for col_name in df_class.columns[1:-3].tolist() + ["stack_model"]:
    acc_dict = {}
    auc_dict = {}

    acc_dict["metric_type"] = "accuracy"
    acc_dict["model_type"] = col_name
    acc_dict["score"] = accuracy_score(df_class["Actual Y"], df_class[col_name])

    auc_dict["metric_type"] = "roc_auc"
    auc_dict["model_type"] = col_name
    auc_dict["score"] = roc_auc_score(df_class["Actual Y"], df_class[col_name])

    dict_lst.append(acc_dict)
    dict_lst.append(auc_dict)

df_result = pd.DataFrame.from_dict(dict_lst)
st.dataframe(df_result, 500, 500)

# add table description
st.write(
    """
    Table 1: Using a a list of classification algorithms with hyperparameter tuning and and cross validation, we are seeing a fairly narrow range of accuracy and AUC metrics.
"""
)


model_selects = ["stack_model"] + df_class.columns[1:7].tolist()
option = st.selectbox(
    "Which model would you like to compare baseline against?", model_selects
)

df_class["Buy and Hold Return"] = (1 + df_class["Return"]).cumprod() - 1
df_class["Classification Model Return"] = (
    1 + df_class["Return"] * df_class[option]
).cumprod() - 1

st.write(df_class["Buy and Hold Return"][-2:-1])
st.write(df_class["Classification Model Return"][-2:-1])
df_line_chart = df_class[["Buy and Hold Return", "Classification Model Return"]]
st.line_chart(df_line_chart)

# Add final description
st.write(
    """
    Figure 2: Using the buy and hold strategy as baseline, we compare against a range of classification models
"""
)


# add final thoughts
st.write(
    """# Final Thought:
    Clearing using daily entry point for buying and selling Bitcoin with classification model is less optimal than the baseline buy and hold strategy. 
    However, given the AUC for the stacked model is better than 50%, we hypothesize there is a possibility to fine tune the algorithm to a shorter time frame 
    in order to capture potential shorter term gains in the bitcoin price with Reddit data.
"""
)
