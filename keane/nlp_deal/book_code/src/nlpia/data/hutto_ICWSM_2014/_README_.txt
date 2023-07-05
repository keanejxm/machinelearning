=======

Introduction

This README file describes the dataset of the paper:
  VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text
  C.J. Hutto and Eric Gilbert
  Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
 
For questions, please contact: 
C.J. Hutto
Georgia Institute of Technology, Atlanta, GA 30032 
cjhutto@gatech.edu
  
=======

Citation Information

If you use either the dataset or any of the VADER sentiment analysis tools (VADER sentiment lexicon or Python code for rule-based sentiment analysis engine) in your research, please cite the above paper. For example:

  Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

=======

Resource and Dataset Descriptions

This zip file includes:

PRIMARY RESOURCES:
1. vader_icwsm2014_final.pdf
    The original paper for the data set, see citation information (above).
2. vader_sentiment_lexicon.txt
       Empirically validated by multiple independent human judges, VADER incorporates a
    "gold-standard" sentiment lexicon that is especially attuned to microblog-like contexts. 
    The VADER sentiment lexicon is sensitive both the polarity and the intensity of sentiments 
	expressed in social media contexts, and is also generally applicable to sentiment analysis 
	in other domains.
	   Manually creating (much less, validating) a comprehensive sentiment lexicon is 
	a labor intensive and sometimes error prone process, so it is no wonder that many 
	opinion mining researchers and practitioners rely so heavily on existing lexicons 
	as primary resources. We are pleased to offer ours as a new resource.
	   We begin by constructing a list inspired by examining existing well-established 
	sentiment word-banks (LIWC, ANEW, and GI). To this, we next incorporate numerous 
	lexical features common to sentiment expression in microblogs, including 
	 - a full list of Western-style emoticons, for example, :-) denotes a smiley face 
	   and generally indicates positive sentiment)
	 - sentiment-related acronyms and initialisms (e.g., LOL and WTF are both examples of 
	   sentiment-laden initialisms)
	 - commonly used slang with sentiment value (e.g., nah, meh and giggly). 
	This process provided us with over 9,000 lexical feature candidates. Next, we assessed 
	the general applicability of each feature candidate to sentiment expressions. We 
	used a wisdom-of-the-crowd13 (WotC) approach (Surowiecki, 2004) to acquire a valid 
	point estimate for the sentiment valence (intensity) of each context-free candidate 
	feature. We collected intensity ratings on each of our candidate lexical features 
	from ten independent human raters (for a total of 90,000+ ratings). Features were 
	rated on a scale from "[�4] Extremely Negative" to "[4] Extremely Positive", with 
	allowance for "[0] Neutral (or Neither, N/A)". 
	   We kept every lexical feature that had a non-zero mean rating, and whose standard 
	deviation was less than 2.5 as determined by the aggregate of ten independent raters. 
	This left us with just over 7,500 lexical features with validated valence scores that 
	indicated both the sentiment polarity (positive/negative), and the sentiment intensity 
	on a scale from �4 to +4. For example, the word "okay" has a positive valence of 0.9, 
	"good" is 1.9, and "great" is 3.1, whereas "horrible" is �2.5, the frowning emoticon :( 
	is �2.2, and "sucks" and it's slang derivative "sux" are both �1.5. 
3. vaderSentiment.py
    The Python code for the rule-based sentiment analysis engine. Implements the 
	grammatical and syntactical rules described in the paper, incorporating empirically 
	derived quantifications for the impact of each rule on the perceived intensity of 
	sentiment in sentence-level text. Importantly, these heuristics go beyond what would 
	normally be captured in a typical bag-of-words model. They incorporate **word-order 
	sensitive relationships** between terms. For example, degree modifiers (also called 
	intensifiers, booster words, or degree adverbs) impact sentiment intensity by either 
	increasing or decreasing the intensity. Consider these examples:
	   (a) "The service here is extremely good" 
	   (b) "The service here is good"
	   (c) "The service here is marginally good"
	From Table 3 in the paper, we see that for 95% of the data, using a degree modifier
    increases the positive sentiment intensity of example (a) by 0.227 to 0.36, with a 
	mean difference of 0.293 on a rating scale from 1 to 4. Likewise, example (c) reduces 
	the perceived sentiment intensity by 0.293, on average.

DATASETS AND TESTING RESOURCES:
4. tweets_GroundTruth.txt
	FORMAT: the file is tab delimited with ID, MEAN-SENTIMENT-RATING, and TWEET-TEXT
    DESCRIPTION: includes "tweet-like" text as inspired by 4,000 tweets pulled from Twitter�s public timeline, plus 200 completely contrived tweet-like texts intended to specifically test syntactical and grammatical conventions of conveying differences in sentiment intensity. The "tweet-like" texts incorporate a fictitious username (@anonymous) in places where a username might typically appear, along with a fake URL ( http://url_removed ) in places where a URL might typically appear, as inspired by the original tweets. The ID and MEAN-SENTIMENT-RATING correspond to the raw sentiment rating data provided in 'tweets_anonDataRatings.txt' (described below).
5. tweets_anonDataRatings.txt
    FORMAT: the file is tab delimited with ID, MEAN-SENTIMENT-RATING, STANDARD DEVIATION, and RAW-SENTIMENT-RATINGS
	DESCRIPTION: Sentiment ratings from a minimum of 20 independent human raters (all pre-screened, trained, and quality checked for optimal inter-rater reliability).
6. nytEditorialSnippets_GroundTruth.txt
	FORMAT: the file is tab delimited with ID, MEAN-SENTIMENT-RATING, and TEXT-SNIPPET
    DESCRIPTION: includes 5,190 sentence-level snippets from 500 New York Times opinion news editorials/articles; we used the NLTK tokenizer to segment the articles into sentence phrases, and added sentiment intensity ratings. The ID and MEAN-SENTIMENT-RATING correspond to the raw sentiment rating data provided in 'nytEditorialSnippets_anonDataRatings.txt' (described below).
7. nytEditorialSnippets_anonDataRatings.txt
	FORMAT: the file is tab delimited with ID, MEAN-SENTIMENT-RATING, STANDARD DEVIATION, and RAW-SENTIMENT-RATINGS
    DESCRIPTION: Sentiment ratings from a minimum of 20 independent human raters (all pre-screened, trained, and quality checked for optimal inter-rater reliability).
8. movieReviewSnippets_GroundTruth.txt
	FORMAT: the file is tab delimited with ID, MEAN-SENTIMENT-RATING, and TEXT-SNIPPET
    DESCRIPTION: includes 10,605 sentence-level snippets from rotten.tomatoes.com. The snippets were derived from an original set of 2000 movie reviews (1000 positive and 1000 negative) in Pang & Lee (2004); we used the NLTK tokenizer to segment the reviews into sentence phrases, and added sentiment intensity ratings. The ID and MEAN-SENTIMENT-RATING correspond to the raw sentiment rating data provided in 'movieReviewSnippets_anonDataRatings.txt' (described below).
9. movieReviewSnippets_anonDataRatings.txt
	FORMAT: the file is tab delimited with ID, MEAN-SENTIMENT-RATING, STANDARD DEVIATION, and RAW-SENTIMENT-RATINGS
    DESCRIPTION: Sentiment ratings from a minimum of 20 independent human raters (all pre-screened, trained, and quality checked for optimal inter-rater reliability).
10. amazonReviewSnippets_GroundTruth.txt
	 FORMAT: the file is tab delimited with ID, MEAN-SENTIMENT-RATING, and TEXT-SNIPPET
     DESCRIPTION: includes 3,708 sentence-level snippets from 309 customer reviews on 5 different products. The reviews were originally used in Hu & Liu (2004); we added sentiment intensity ratings. The ID and MEAN-SENTIMENT-RATING correspond to the raw sentiment rating data provided in 'amazonReviewSnippets_anonDataRatings.txt' (described below).
11. amazonReviewSnippets_anonDataRatings.txt
	 FORMAT: the file is tab delimited with ID, MEAN-SENTIMENT-RATING, STANDARD DEVIATION, and RAW-SENTIMENT-RATINGS
     DESCRIPTION: Sentiment ratings from a minimum of 20 independent human raters (all pre-screened, trained, and quality checked for optimal inter-rater reliability).

ADDITIONAL RESOURCES:
12. _README_.txt
	 This readme file
13. vader_sentiment_comparison_online_weblink
     A short-cut hyperlinked to the online (web-based) sentiment comparison using a "light" version of VADER. http://www.socialai.gatech.edu/apps/sentiment.html .

=======

Python Code EXAMPLE:

	from vaderSentiment import sentiment as vaderSentiment

	# --- example sentences -------
	sentences = [
				"VADER is smart, handsome, and funny.",       # positive sentence example
				"VADER is smart, handsome, and funny!",       # punctuation emphasis handled correctly (sentiment intensity adjusted)
				"VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
				"VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
				"VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
				"VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
				"The book was good.",         # positive sentence
				"The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
				"The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
				"A really bad, horrible book.",       # negative sentence with booster words
				"At least it isn't a horrible book.", # negated negative sentence with contraction
				":) and :D",     # emoticons handled
				"",              # an empty string is correctly handled
				"Today sux",     #  negative slang handled
				"Today sux!",    #  negative slang with punctuation emphasis handled
				"Today SUX!",    #  negative slang with capitalization emphasis
				"Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
				 ].
	for sentence in sentences:
		print sentence,
		vs = vaderSentiment(sentence)
		print "\n\t" + str(vs)
	
# --- output for the above example code ---
VADER is smart, handsome, and funny.
 	{'neg': 0.0, 'neu': 0.254, 'pos': 0.746, 'compound': 0.8316}
VADER is smart, handsome, and funny!
 	{'neg': 0.0, 'neu': 0.248, 'pos': 0.752, 'compound': 0.8439}
VADER is very smart, handsome, and funny.
 	{'neg': 0.0, 'neu': 0.299, 'pos': 0.701, 'compound': 0.8545}
VADER is VERY SMART, handsome, and FUNNY.
 	{'neg': 0.0, 'neu': 0.246, 'pos': 0.754, 'compound': 0.9227}
VADER is VERY SMART, handsome, and FUNNY!!!
 	{'neg': 0.0, 'neu': 0.233, 'pos': 0.767, 'compound': 0.9342}
VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!
 	{'neg': 0.0, 'neu': 0.294, 'pos': 0.706, 'compound': 0.9469}
The book was good.
 	{'neg': 0.0, 'neu': 0.508, 'pos': 0.492, 'compound': 0.4404}
The book was kind of good.
 	{'neg': 0.0, 'neu': 0.657, 'pos': 0.343, 'compound': 0.3832}
The plot was good, but the characters are uncompelling and the dialog is not great.
 	{'neg': 0.327, 'neu': 0.579, 'pos': 0.094, 'compound': -0.7042}
A really bad, horrible book.
 	{'neg': 0.791, 'neu': 0.209, 'pos': 0.0, 'compound': -0.8211}
At least it isn't a horrible book.
 	{'neg': 0.0, 'neu': 0.637, 'pos': 0.363, 'compound': 0.431}
:) and :D
 	{'neg': 0.0, 'neu': 0.124, 'pos': 0.876, 'compound': 0.7925}
 
    {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
Today sux
 	{'neg': 0.714, 'neu': 0.286, 'pos': 0.0, 'compound': -0.3612}
Today sux!
 	{'neg': 0.736, 'neu': 0.264, 'pos': 0.0, 'compound': -0.4199}
Today SUX!
 	{'neg': 0.779, 'neu': 0.221, 'pos': 0.0, 'compound': -0.5461}
Today kinda sux! But I'll get by, lol
 	{'neg': 0.195, 'neu': 0.531, 'pos': 0.274, 'compound': 0.2228}

=======

Online (web-based) Sentiment Comparison using VADER

     http://www.socialai.gatech.edu/apps/sentiment.html .

=======
