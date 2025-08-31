import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
# Load sample dataset (tweets / social media data from GitHub link)
data = pd.read_csv("social_media.csv")   # update with actual file name
print(data.head())
# Function to calculate sentiment polarity
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Apply to dataset
data['Sentiment_Score'] = data['text'].apply(get_sentiment)

# Classify as Positive / Negative / Neutral
def classify(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

data['Sentiment'] = data['Sentiment_Score'].apply(classify)
print(data['Sentiment'].value_counts())
sns.countplot(x='Sentiment', data=data, palette='pastel')
plt.title("Sentiment Distribution")
plt.show()
sns.histplot(data['Sentiment_Score'], bins=20, kde=True, color="blue")
plt.title("Sentiment Polarity Scores")
plt.show()
positive_text = " ".join(data[data['Sentiment']=="Positive"]['text'])
negative_text = " ".join(data[data['Sentiment']=="Negative"]['text'])

# Positive Word Cloud
wc_pos = WordCloud(width=600, height=400, background_color="white").generate(positive_text)
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Words")
plt.show()

# Negative Word Cloud
wc_neg = WordCloud(width=600, height=400, background_color="black", colormap="Reds").generate(negative_text)
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Words")
plt.show()
positive_text = " ".join(data[data['Sentiment']=="Positive"]['text'])
negative_text = " ".join(data[data['Sentiment']=="Negative"]['text'])

# Positive Word Cloud
wc_pos = WordCloud(width=600, height=400, background_color="white").generate(positive_text)
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Words")
plt.show()

# Negative Word Cloud
wc_neg = WordCloud(width=600, height=400, background_color="black", colormap="Reds").generate(negative_text)
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Words")
plt.show()
