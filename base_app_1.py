"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd


# Vectorizer
news_vectorizer = open("resources/count_vector_1.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate change Tweet Classification")
	
	
	




	


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":

		st.subheader("Select Classification Model")
	

		select_model = st.radio("",('Logistic Regresion model','Linear SVC model','Random Forest Regressor model','Multinomial baive Bayes model'))
		if select_model == 'Logistic Regresion model':
			model = "resources/LR_model_1.pkl"
		elif select_model == 'Linear SVC model':
			model = "resources/svc_model_1.pkl"
		elif select_model == 'Random Forest Regressor model':
			model = "resources/Forest_model_1.pkl"
		elif select_model == 'Multinomial baive Bayes model':
			model = "resources/svc_model_1.pkl"

		st.subheader("Single tweet analysis")
		st.info("The tweet entered below will be used to make a sentiment analysis prediction according to the selected model above.")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(model),"rb"))
			prediction = predictor.predict(vect_text)
			

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Sentimanet analysis: {}".format(prediction))

		st.subheader("Sentiment Info")
		st.info("1 Pro: the tweet supports the belief of man-made climate change"+
		"\n\r0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change"+
		"\n\r-1 Anti: the tweet does not believe in man-made climate change"+
		"\n\r2 News: the tweet links to factual news about climate change")
	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
