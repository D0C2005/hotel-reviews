# Sentiment analysis and topic modeling of hotel reviews

The Python files in this repository are the codes used in the research for the paper 

"_From Reviews to Actionable Insights: Leveraging NLP for Sentiment and Topic Analysis in Hospitality_" 

(in the process of publishing)
by:

- Cristina Fleșeriu (Faculty of Business, Babeș-Bolyai University, Cluj-Napoca, Romania)
- Smaranda Adina Cosma (Faculty of Business, Babeș-Bolyai University, Cluj-Napoca, Romania)
- Katarina Kostelic (Faculty of Informatics, Juraj Dobrila University of Pula, Pula, Croatia)
- Vlad Bocăneț (Faculty of Industrial Engineering, Robotics and Production Management, Technical University of Cluj-Napoca, Cluj-Napoca, Romania)

The original data was obtained from: _https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe_

The files have the following contents:

- **01. Reviews_analysis_preparation.py** - the preparation of the original data.
- **02. Naive_Bayes_model.py** - the training and testing of the Naive Bayes model for Sentiment Analysis
- **03. LSTM_model.py** - the training and testing of the LSTM model for Sentiment Analysis
- **04. BERT_model_w_att_heatmaps.py** - the inference script for the BERT model for Sentiment Analysis using attention heatmaps for model explainability.
- **05. GPT_model_BatchAPI_Structured_output.py** - the inference script for the GPT model for Sentiment Analysis using OpenAI API with batch API and structured output. An openAI API key is required.
- **06. LDA_topic_modeling.py** - the preparation of data and usage of the LDA model for topic analysis
- **Training_data.zip** - the training data used for training models for Sentiment Analysis
- **Test_data.zip** - the test data used for inference of models for Sentiment Analysis

