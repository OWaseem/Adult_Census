import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error

@st.cache()
def prediction(car_df, car_width, engine_size, horse_power, drive_wheel,car_company_buick):
    X = car_df.iloc[:, :-1]
    y = car_df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)  # Trainning the model with train df
    score = lin_reg.score(X_train, y_train)     # Sxcore of Trainning  model


    y_test_pred = lin_reg.predict(X_test)   # Giving new questions and checking for error values
    test_r2_score = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_msle = mean_squared_log_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    price = lin_reg.predict([[car_width, engine_size, horse_power, drive_wheel,car_company_buick]])  # Prediction done with what the user has given as input
    #price = price[0]



    return price, score, test_r2_score, test_mae, test_msle, test_rmse

def app(car_df):

	st.markdown("<p style='color:red;font-size:40px'> This APP uses <b> Linear Regression </b> </p>", unsafe_allow_html = True)
	st.sidebar.subheader("Select your values:")
	rw = st.sidebar.slider("Input Car Width", float(car_df['carwidth'].min()), float(car_df['carwidth'].max()))
	es = st.sidebar.slider("Input Engine Size", int(car_df['enginesize'].min()), int(car_df['enginesize'].max()))
	hp = st.sidebar.slider("Input Horse Power", int(car_df['horsepower'].min()), int(car_df['horsepower'].max()))

	dw = st.sidebar.radio('Is the car forward driving?', ('Yes', 'No'))
	cb = st.sidebar.radio('Is the car manufactured by BUICK?', ('Yes', 'No'))

	if dw == 'Yes':
		dw = 1
	else:
		dw = 0

	if cb == 'Yes':
		cb = 1
	else:
		cb = 0

	if st.button('Predict'):
		price, score, test_r2_score, test_mae, test_msle, test_rmse = prediction(car_df, rw, es, hp, dw, cb)
		st.subheader('Prediction Results')
		st.success('Predicted Price of Car: ${:,}'.format(int(price)))
		st.error('MAE: {:2.2%}'.format(test_mae))
		st.error('MSLE: {:2.2%}'.format(test_msle))
		st.error('RMSE: {:2.2%}'.format(test_rmse))
		st.info('R2 Score: {:2.2%}'.format(test_r2_score))
		st.info('Score: {:2.2%}'.format(score))


