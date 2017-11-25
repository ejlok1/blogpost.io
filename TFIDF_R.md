---
title: "Code demo for TF-IDF"
author: "Eu Jin Lok"
date: "12 November 2017"
output: html_document
---



## Using TFIDF to identify context determinant words  

In this notebook we will go into the details of how to run TFIDF step by step in code. For the full background on this topic, please checkout the SandyEdge blog post here: <http://xxxx.com>.

But before we begin, let me first introduce the mathematical equation, which I proudly spent the weekend learning the LaTex code for:

$$ tfidf_{t,d} = tf_{t,d} \cdot \log \frac{N}{df_t} $$
_where_
<br>
$t$ = _term_
<br>
$d$ = _document_
<br>
$tfidf_{t,d}$ = _term $t$ $tfidf$ score for document_ $d$
<br>
$tf_{t,d}$ = _number of occurences of term $t$ in document $d$_
<br>
${df_t}$ = _number of documents containing term $t$_
<br>
$N$ = _total number of documents_


So without further ado, lets begin 

```r
############
# load key libraries
############
#NOTE: R version 3.3.3 (2017-03-06) -- "Another Canoe"
library(tm) #standard text mining processes
library(slam)
library(ggplot2)
```

I'm going to use my own dataset which is based on Student essays. You can easily find other datasets and examples, in the wild but I'm using this as it is one of the better datasets for text mining due to its large signal over noise. 


```r
############
# Read the data
############
data = read.delim ("C:\\Users\\user\\Dropbox\\Pet Project\\Blog\\TFIDF\\training_set.tsv", header = T, quote = "")
data$essay = as.character(data$essay)
data = data[data$essay_set == 1,]

#do some checks
data[data$essay_set == 1,"essay"][1] #
```

```
## [1] "\"Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, it's a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listening.\""
```

If you read through the above, the only issue with highschool essay's is that its filled with spelling mistakes. Now lets get the dataset into the proper object that is a corpus. And then we can start cleaning the text so it looks good


```r
# apply corpus object 
corpus = Corpus(VectorSource(data[,3]))
corpus = tm_map(corpus, tolower)
corpus = tm_map(corpus, removePunctuation)
#corpus = tm_map(corpus, removeWords, stopwords('english'))  # de-activate this for TF-IDF demo 
corpus = tm_map(corpus, stripWhitespace)
inspect(corpus[1])
```

```
## <<SimpleCorpus>>
## Metadata:  corpus specific: 1, document level (indexed): 0
## Content:  documents: 1
## 
## [1] dear local newspaper i think effects computers have on people are great learning skillsaffects because they give us time to chat with friendsnew people helps us learn about the globeastronomy and keeps us out of troble thing about dont you think so how would you feel if your teenager is always on the phone with friends do you ever time to chat with your friends or buisness partner about things well now theres a new way to chat the computer theirs plenty of sites on the internet to do so organization1 organization2 caps1 facebook myspace ect just think now while your setting up meeting with your boss on the computer your teenager is having fun on the phone not rushing to get off cause you want to use it how did you learn about other countrysstates outside of yours well i have by computerinternet its a new way to learn about what going on in our time you might think your child spends a lot of time on the computer but ask them so question about the economy sea floor spreading or even about the date1s youll be surprise at how much heshe knows believe it or not the computer is much interesting then in class all day reading out of books if your child is home on your computer or at a local library its better than being out with friends being fresh or being perpressured to doing something they know isnt right you might not know where your child is caps2 forbidde in a hospital bed because of a driveby rather than your child on the computer learning chatting or just playing games safe and sound in your home or community place now i hope you have reached a point to understand and agree with me because computers can have great effects on you or child because it gives us time to chat with friendsnew people helps us learn about the globe and believe or not keeps us out of troble thank you for listening
```

After applying the cleaning process, see the text is much more readable now? Next we're going to create a DTM object, also known as Document-Term-Matrix. Basically a matrix of documents on rows and terms on columns. Its the native format that is required to also run SVD / LSI (Latent Semantic Indexing) and topic modelling 


```r
#convert to DTM  
#corpus <- tm_map(corpus, PlainTextDocument)
review_dtm <- DocumentTermMatrix(corpus)

# Remove sparse terms 
# review_dtm = removeSparseTerms(review_dtm, 0.999) # de-activate this for TF-IDF demo 

#check the data 
inspect(review_dtm[1:10, 1:10])
```

```
## <<DocumentTermMatrix (documents: 10, terms: 10)>>
## Non-/sparse entries: 50/50
## Sparsity           : 50%
## Maximal term length: 7
## Weighting          : term frequency (tf)
## Sample             :
##     Terms
## Docs about agree all always and are ask because bed being
##   1      8     1   1      1   4   1   1       4   1     3
##   10     3     0   4      0  16   8   0       6   0     0
##   2      0     0   2      0  13   3   0       1   0     0
##   3      0     0   1      0  12   5   0       0   0     1
##   4      0     0   0      1   7   2   0       4   0     0
##   5      2     0   1      0  11   8   0       3   0     0
##   6      0     0   5      0   7   3   0       0   0     1
##   7      1     1   1      0  13   3   0       0   0     0
##   8      8     4   1      0  13   3   0       0   0     0
##   9      1     0   8      0  14   3   0       0   0     0
```

Notice that the matrix is really sparse. This tends to be the case in real life documents, where sparsity is close the 100%. I won't go into details but a few phenomenon follows similar distribution such as movies watched, our social networks etc. In movies for example, there are hundres of thousands movies made in the world, but any one person will watch less than 10% of the whole list of movies.   


```r
#check the vocabulary list 
colnames(review_dtm)[100:110]
```

```
##  [1] "place"    "playing"  "plenty"   "point"    "question" "rather"  
##  [7] "reached"  "reading"  "right"    "rushing"  "safe"
```
Checking through to see the words coming through aren't strange. So now lets apply TF-IDF


```r
#Now apply TFIDF 
term_tfidf = tapply(review_dtm$v/row_sums(review_dtm)[review_dtm$i], review_dtm$j, mean) * log2(nDocs(review_dtm)/col_sums(review_dtm > 0))
```
For more detailed explanation of the formula and how it works, please read through the blog post on SandyEdge. Now lets obtained a list of words and its TFIDF score 


```r
#Lets export the TFIDF scores for each term 
count = slam::row_sums(t(review_dtm))
terms = review_dtm$dimnames$Terms
terms = data.frame(cbind(terms, term_tfidf, count))
terms$term_tfidf = as.numeric(as.character(terms$term_tfidf))
terms = terms[order(terms$term_tfidf),]
terms = terms[order(terms$term_tfidf,decreasing = FALSE),]
```

Lets check the statistics of the TF-IDF score. In practice, we usually remove any words with TFIDF score less than 0.1 (<0.1). However, depending on your data, one may have to adjust this cut-off slightly. Generally if you have really noisy dataset, you want to reduce this down further 


```r
summary(terms[,2]) 
```

```
##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
## 0.0004508 0.0253500 0.0308600 0.0357700 0.0392700 0.9391000
```
From the stat above, I can sufficiently conclude that we have a fairly noisy dataset (median TFIDF = 0.04). But not the end of the world. Lets have a look at what are the top 10 words with lowest TFIDF scores


```r
# check the bottom 10 TFIDF scores 
print(terms[1:10,])
```

```
##         terms   term_tfidf count
## 123       the 0.0004507925 22187
## 5         and 0.0005299006 18585
## 252      that 0.0012089748  9682
## 6         are 0.0013811466  8658
## 59       have 0.0017483948  6765
## 97     people 0.0018898125 10244
## 45        for 0.0019835158  6288
## 142      with 0.0020244145  6659
## 28  computers 0.0020748379 10698
## 32       dear 0.0022822618  1193
```

Notice how it has captured not only stopwords, but also other non-useful words like "computers". Now lets have a look at the top 10 words 


```r
# check the top TFIDF scores 
terms = terms[order(terms$term_tfidf,decreasing = TRUE),]
print(terms[1:10,])
```

```
##           terms term_tfidf count
## 15227      nede  0.9391383     2
## 8204        pol  0.8307762     1
## 10539   difrent  0.5400045     4
## 15036  illeagle  0.4909132     2
## 15224 compurers  0.4695692     1
## 15225      lemt  0.4695692     1
## 15226 lomenteno  0.4695692     1
## 15228  werpsite  0.4695692     1
## 1242  evansmant  0.4408200     2
## 1268        waf  0.4408200     2
```

Looks like there's lots of sparse words appearing with high TFIDF scores. Turns out this spelling mistake is pretty common amongst students and its confusing the model slightly. We could remove sparse terms, or use bigger dataset or apply spelling correction, all of which may help solve this issue

Let's plot the results 

```r
# Plot the histogram of the counts (should be a power-law dist?)
terms$count = as.numeric(as.character(terms$count))
qplot(terms[,2], bins=100)
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-1.png)

Plotting the TFIDF weight shows a right skew, or in other words a large number of words have low TFIDF scores. This makes sense as we intentionally turned off the stopwords and sparewords removal function


```r
qplot(terms[,3], bins=100)
```

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-1.png)

Plotting just the frequencies shows a power law distribution, which again makes sense as we intentionally turned off the stopwords and sparewords removal function. This 2 graphs sort of conforms to the Luhn's Hypothesis where 

