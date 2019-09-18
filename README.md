# fake news

This project attempts to study and combine the different popular methods of NLP analysis, to detect fake news. It considers three such methods (skip ahead to read about each of the methods individually, the combined results are the last entry):

1. **Bag of Words**: A Bag of Words model works on the frequency of words that occur within the dataset for each value.

**Note: this project does not scan the entire HTML text of a website, but only its description. If a bag of words model was applied to the entire HTML, results might be statistically better.**

It stores these frequencies in a list descriptions [], upon which a logistic regression model is trained and tested.

These are the accuracies obtained for my dataset: with #2002 training values, and #309 testing values.

        Train accuracy | 0.8746253746253746
        Val accuracy | 0.6634304207119741
        Precision | 0.5844748858447488
        Recall | 0.9078014184397163
        F-Score| 0.7111111111111111

Some obvious problems with this model led me to combine different features together. One, it cannot gauge sentence structure as every article, conjunction and other "irrelevant" word is deleted by the model based upon the training values. Two, typically with news articles, especially those created to misinform might use vocabulary of real news sources, and frequency alone will not help detect this very easily.

2. **GloVe Vector**: a vectorizer that creates a vector for each word on a 300 dimensional plane, and defines similarity between words with the help of their parallel components.

This model fairly works to improve the bag of words model by establishing, the way a human does, semantics and similarities between words. While our memory might be more dynamic, these semantics are purely mathematical. A 300 dimensional plane considers 300 different co-ordinates of similarity. If you want to see this in action, I suggest you go here: https://nlp.stanford.edu/projects/glove/. I have applied the same in my program.

These are the accuracies obtained for my dataset: with #2002 training values, and #309 testing values.

      Train accuracy | 0.8656343656343657
      Val accuracy | 0.7702265372168284
      Precision | 0.7011494252873564
      Recall | 0.8652482269503546
      F-Score | 0.7746031746031746

*Problems* arise when one considers tone, sarcasm and the inspiration of this very project: the GPT-2 model(https://openai.com/blog/better-language-models/#sample2)

3. **Keyword featurizer**: this is what you would imagine one does when training a model. Teaching the computer what to consider as an authentic news source or not. Let's look at it using the list of #keywords and #domains used in this program.

keywords = ['truth', 'apparently', 'fact', '!!!', 'proven', 'alleged', 'sources', 'references', 'opinion', '<img', 'editor', 'write to', 'datetime']

While scanning an html, these were determined to be the keywords that can categorically define a news source. This was done by manual searching and thinking through possibilities. If a higher computational power can be applied, the model can learn to scan through the HTML and create it's own list of keywords.

These are the accuracies obtained for my dataset: with #2002 training values, and #309 testing values.
      
      Train accuracy 0.913086913086913
      Val accuracy 0.9449838187702265
      Precision: 0.9558823529411765
      Recall: 0.9219858156028369
      F-Score: 0.9386281588447654

A drawback would be the continuous trial and error involved in determining the right keywords to be right accuracy. This method also defeats the purpose of developing a self-learning unsupervised model. What about combining the methods? What kind of results would that yield? Does the model learn to take the best things out of all our approaches and apply them?

4. **COMBINED RESULTS**: a combined result was obtained to show.

        Train accuracy 0.974025974025974
        Val accuracy **0.8932038834951457**
        Precision: **0.8506493506493507**
        Recall: 0.9290780141843972
        F-Score: 0.8881355932203389

*Things I'm still working on*:
1. an input for the user. As of now, one has to hard-code a link of the website into the model. Because of the constant downloading and unzipping of vectors and files, I was not able to figure out a method to allow the user to input anything.
2. Working with other models apart from Logistic Regression.
3. Researching ways to make this into an unsupervised learning model.
4. Scanning parts of the HTML that might be useful (cutting irrelevant sections through some data mining, probably)

Unfortunately, the drawbacks regarding each method is a consequence of us not knowing ourselves about how a computer model can learn the english language. Our language is ever evolving, which makes it a bit more difficult. As is the nature with deepfakes, it can be dangerous for a model like the GPT-2 to be in the open market. This program can identify news articles written by the GPT-2, but these samples are not enough to determine it's efficiency. The sample sizes can always be bigger; we can also test for longer periods of time.

However, this project does work in a very good accuracy to determine whether a news source is fake or not, and the combination of methods works very well to take the most prominent features of each method and use that to its advantage
