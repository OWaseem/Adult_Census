import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def app(car_df):
	st.header('View Data')
	with st.expander('Load DataFrame'):
		st.table(car_df)

	st.subheader('Column Description')
	if st.checkbox('Show Summary'):
		st.table(car_df.describe())

	col1, col2, col3 = st.columns(3)

	with col1:
		if st.checkbox('Show All Column Names'):
			st.table(car_df.columns)
	with col2:
		if st.checkbox('View Column Datatype'):
			st.table(car_df.dtypes)
	with col3:
		if st.checkbox('View Column Values'):
			column = st.selectbox('column', tuple(car_df.columns))
			st.table(car_df[column].head())

