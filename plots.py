import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def app(car_df):

	  st.set_option('deprecation.showPyplotGlobalUse', False)
	  st.subheader("Correlation Heatmap")
	  plt.figure(figsize = (15, 5))
	  sns.heatmap(car_df.corr(), annot = True)
	  st.pyplot()

	  for feature in car_df.columns[:-1]:
	    st.subheader(f"Scatter plot between {feature} and Price")
	    plt.figure(figsize = (12, 6))
	    sns.scatterplot(x = feature, y = 'price', data = car_df)
	    st.pyplot()

	  for feature in ['carwidth', 'enginesize', 'horsepower']:
	  	st.subheader(f'Histogram for {feature}')
	  	plt.figure(figsize = (15, 5))
	  	plt.hist(car_df[feature])
	  	st.pyplot()


	  for feature in ['carwidth', 'enginesize', 'horsepower']:
	  	st.subheader(f"Box plot of {feature}")
	  	plt.figure(figsize = (15, 5))
	  	sns.boxplot(data = car_df, x = feature, orient = 'h')
	  	st.pyplot()