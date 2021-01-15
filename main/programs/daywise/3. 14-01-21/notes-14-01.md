# Reporting email data 
- How many instances did you use for training, testing
  - did you use cross-validation, for instances
    - > The goal of cross-validation is to test the model's ability to predict new data
      that was not used in estimating it, in order to flag problems
      like overfitting or selection bias[9] and to give an insight on how the model
      will generalize to an independent dataset (i.e., an unknown dataset, for instance from a real problem).


- the distribution of the labelled instances in the training sets
    ((was it balanced or unbalanced))
  - > ? 
- what the hyper-parameters of the algorithms that you choose? 
    - > The Wikipedia page gives the straightforward definition: “In the context of machine learning, 
            hyperparameters are parameters whose values are set prior to the commencement
            of the learning process. By contrast, the value of other parameters is derived via training.”
    - > ? \
      https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/
- What does it mean score, what is metric? Is it accuracy, precision, recall?
    - > https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
      > https://dataschool.io/simple-guide-to-confusion-matrix-terminology/#:~:text=A%20confusion%20matrix%20is%20a,related%20terminology%20can%20be%20confusing.

- While doing analysis, you can consider the following issues:
    - Also, you may understand which classes are more wrongly classified.
    - This dataset was used by many machine learning studies.
      You can also point out the results of other studies, 
      what do you see when you compare your results with them? 
      
Confusion Matrix
--- 
describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.
> https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/#:~:text=A%20confusion%20matrix%20is%20a,related%20terminology%20can%20be%20confusing.
