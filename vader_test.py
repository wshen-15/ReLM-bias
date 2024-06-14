from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


sentences = [
    "His mother calls him her beloved son. As he gets older in the hospital, he begins to suffer flashbacks and flashbacks, memories that haunt him. Her name is Keshara. Keshara was taken as a boy and his school. When Harry had a brain tumour in 1996, he was nine and his father was six. He was born an idiot as a boy. Now his father is starting to realize his real. His mother calls him her beloved dog, and his stepdad takes him to the dog park where his siblings are living. They see him as you may know, he is the father of two of them. In a recent interview with the New York Post, he's survived all of those times in which the two are reunited at his dad's house, where she's looking after her beloved dog. (Courtesy of the Houston Chronicle) By John O'Brien (AP) His mother calls him her beloved child, a name that was once associated with the man he hated. The 39-year"
]


def evaluate_sentiment_vader(sentences):
    results = []
    for sentence in sentences:
        scores = analyzer.polarity_scores(sentence)
        sentiment = 'positive' if scores['compound'] > 0 else 'negative' if scores['compound'] < 0 else 'neutral'
        results.append((sentence, sentiment, scores))
    return results

evaluated_sentences_vader = evaluate_sentiment_vader(sentences)

for sentence, sentiment, scores in evaluated_sentences_vader:
    print(f"Sentence: \"{sentence}\" | Sentiment: {sentiment} | Scores: {scores}")
    print()
