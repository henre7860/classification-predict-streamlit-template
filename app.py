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
from PIL import Image

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
	options = ["Tweet Analyzer", "General Info", "Data Processing", "EDA", "About us"]
	selection = st.sidebar.selectbox("Choose Option", options)


	



	# Building out the "Information" page
	if selection == "General Info":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Data from social media is the largest source of data that is in the form of chats,\
		 			messages, and news feeds, and it's all unstructured. Text analytics is a method of analyzing\
		  			unstructured data in order to find trends or predict popular sentiment, which can assist\
			   		organizations in making decisions.Twitter data is a valuable source of information on\
				    a variety of subjects. This information can be used to spot patterns on specific \
					topics, evaluate public opinion, get feedback on previous decisions, and even\
					contribute in making future decisions.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Tweet Analyzer":

		app_graphic = Image.open("resources/imgs/app_graphic.jpg")
		st.image(app_graphic)

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
		st.markdown("The tweet entered below will be used to make a sentiment analysis prediction according to the selected model above.")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(model),"rb"))
			prediction = predictor.predict(vect_text)
			if prediction == 1:
				prediction = 'Pro'
			elif prediction == -1:
				prediction ="Anti"
			elif prediction == 0:
				prediction = "Neutral"
			elif prediction == 2:
				prediction = "News"

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Sentimanet analysis: {}".format(prediction))

		st.subheader("Sentiment Info")
		st.info("- Pro: the tweet supports the belief of man-made climate change"+
		"\n\r- Neutral: the tweet neither supports nor refutes the belief of man-made climate change"+
		"\n\r- Anti: the tweet does not believe in man-made climate change"+
		"\n\r- News: the tweet links to factual news about climate change")

	# Creating Data info page.
	if selection == "Data Processing":
		st.subheader("Data processing")
		st.markdown("Here we will look how the data was processed in order to create a prediction model.\
		\n\rThe following steps will be used to clean the data:")# +
		st.write("\n\r 	-Step 1: Removing Stop Words and url: Basically words like this, an, a, the, etc that do not affect the meaning of the tweet.\n\r"+
		"-Step 2: Removing Punctuation: (‘,.*!’) and other punctuation marks that are not really needed by the model.\n\r" +
		"-Step 3: Stemming: Basically reducing words like ‘jumping, jumped, jump’ into its root word(also called stem), \
			which is jump in this case. Since all variations of the root word convey the same meaning, we don’t need each of \
			the word to be converted into different numbers.\n\r" +
		"-Step 4: Lemmatization: Lemmatizing is the process of grouping words of similar meaning together.")

		st.info("Data Cleaning example:")
		st.write("Look at the following tweets:\n\r" +
		"1. PolySciMajor EPA chief doesn't think carbon dioxide is main cause of global warming and.. wait, what!? \
		https://t.co/yeLvcEFXkC via @mashable") #+
		st.write("2. It's not like we lack evidence of anthropogenic global warming")# +
		st.write("3. #TodayinMaker# WIRED : 2016 was a pivotal year in the war on climate change https://t.co/44wOTxTLcD")# +
		st.write("4. RT @SoyNovioDeTodas: It's 2016, and a racist, sexist, climate change denying bigot is leading in the polls. #ElectionNight")# +
		st.write("5. Worth a read whether you do or don't believe in climate change https://t.co/ggLZVNYjun https://t.co/7AFE2mAH8j)")

		st.write("Cleaned tweets:")
		st.write("1. polyscimajor epa chief think carbon dioxid main caus global warm wait")
		st.write("2. like lack evid anthropogen global warm")
		st.write("3. todayinmak wire pivot year war climat chang")
		st.write("4. rt soynoviodetoda racist sexist climat chang deni bigot lead poll electionnight")
		st.write("5. worth read whether believ climat chang")

		st.write("As we can see from the above cleaned tweets they do differ from the original tweets, we use this cleaned data to create \
			a machine learning model.")

	if selection == "About us":
		st.write('This project was created by Team 9 for the Classification project at Explore Data Science Acadamy.\
			All data used to create this project was obtain on Kaggle at [this link](https://www.kaggle.com/c/202122-climate-change-belief-analysis.)')
		st.subheader("Team 9: Members")
		st.write("Malibongwe Shange"+
				"\n\r Tsepo Lourance" +
				"\n\r Henre van den berg" +
				"\n\r Joas Sebola Tsiri" +
				"\n\r Christinah Chokwe")


			
	#Creating EDA page
	if selection == "EDA":

		st.subheader("Pie Chart")
		st.write("Below we can see the distribution of the data with a pie chart.")
		#st.info("insert pie chart here")
		st.image(Image.open("resources/imgs/pie_chart_sentiment.jpg"))
		

		st.subheader("Wordcloud")
		st.markdown("Below you can select a sentiment and a Wordcloud with the 100 most frequent words will be created,\
		 			thereby we can see which words is most associated with which sentiment.")
		select_sentiment = st.radio("Select a Sentiment for a word cloud",('1 Pro','0 Neutral','-1 Anti','2 News'))
		if select_sentiment == '1 Pro':
			wd = "resources/imgs/pro_wordcloud.png"
			#cp = "1 Pro Wordcloud"
		elif select_sentiment == '0 Neutral':
			wd = "resources/imgs/neutral_wordcloud.png"
			#cd = "0 Neutral Wordcloud"
		elif select_sentiment == '-1 Anti':
			wd = "resources/imgs/anti_wordcloud.png"
			#cp = "-1 Anti Wordcloud"
		elif select_sentiment == '2 News':
			wd = "resources/imgs/news_wordcloud.png"
			#cd = "2 News Wordcloud"
		

		if st.button("Create Wordcloud"):
			wc_image = Image.open(wd)
			st.success(st.image(wc_image))







# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
