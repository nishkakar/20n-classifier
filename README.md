# 20n-classifier

I used a Naive-Bayes Classifier to solve this challenge in approximately 3-4 hours of coding and 1 hour of styling. I separated the provided data into two sets: training & testing. To run the classifier on this data, run the following command:

```
python classifier.py training/positive/ training/negative testing/positive testing/negative
```
You can erase the contents of the '.txt' files if you'd like, just make sure the files themselves still exist in the directory.

After you've run the command, you will see results in the Terminal window regarding the accuracy of the classifier and you can see additional details about the correctly classified diseases in the pertinent_data.txt file. 

You can expect to see about 97% accuracy in classification, with specific details on numerous diseases. 

If you'd like to run on smaller training or testing data, I've created directories for those as well. Just alter the data however you'd like and then run the following command: 

```
python classifier.py training-small/positive training-small/negative testing-small/positive testing-small/negative
```