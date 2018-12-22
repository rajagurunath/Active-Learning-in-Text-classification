# Active-Learning-in-Text-classification
To Achieve  Active learning  in Text classification usecases ,Both the Transformers (Representation of the text ) and classifiers built upon the Representations should Learn actively ,i.e able to update itself with new data 

Text classification consists of two stages : 
- Representation learning
- Classifiers
To make the classifiers learn actively we need to introduce active learning in above two stages .
- ### Representation learning :
  - In this stage of the Text classification ,Text are cleaned ,normalized ,and converted into numerical value(Tensors)
  - Commonly used transformers are TFIDF ,word2vec,Doc2vec (paragraph vectors)
  - With this transformers :TFIDF won't support active learning mechanisms since this is Count based feature learning
  - We can learn actively by using word2vec ,doc2vec ,or latest technology introduced in the market ULMFIT ,ELMO,BERT etc
  - We need a strategy to update the Vocabulary lookup as well (i.e one to one mapping between the word and numeric value)
- ### Classifiers:
  - Using continous learners such as Linear models with optimizers,Neural Networks adopts the new data on the fly.
  - While using Tree based models, are difficult to train actively (RandomForest ,Gradient boosting etc)
 
## Active learning in this Repository -Implementation details:
