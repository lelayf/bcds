# BlueCargo Case Study

## Datasets

The data we work with is coming from the Parakweet organization.

#### Attributes

This is super simple:

1) label
2) text

# Virtual Environment

I find that `pyenv` is really useful to manage multiple versions of Python on a single OS.
Install it and then use it to install Python 3.9.

`pyenv install 3.9`

Then proceed with the creation of a virtual environment:

`pyenv virtualenv 3.9 bluecargo`

And activate it:
`pyenv activate bluecargo`

# Python packages

Now with the virtual environment we can install the dependencies written in requirements.txt

`pip install -r requirements.txt`

# Exploration

See notebook `notebook/Modelling.ipynb`. 

Since we are dealing with a binary classification problem, we will focus on accuracy, precision, recall, and F1 score as our key performance metrics.

We test a number of approaches:
- Our baseline model is a variant of Naive Bayes applied to TF-IDF vectorization of the data.
- Our improved model is a Support Vector Machine 
- We also try an SGD classifier
- And an ensemble of classifiers with majority voting strategy
- an approach based on prompt engineering and the Mistral Orca 7B model, hosted on Replicate.
- an approach based on sentence embeddings produced by [all-mpnet-base-v2](https://replicate.com/replicate/all-mpnet-base-v2)

Of all those the SVM is the best.

|Model|Accuracy|Precision|Recall|F1|
|-----|--------|---------|------|--|
|**SVM**|83%|80%|82%|80%|
|SGD|77%|80%|82%|80%|
|Naive Bayes|77%|80%|82%|80%|
|Ensemble|77%|80%|82%|80%|
|Zero-shot|55%|80%|82%|80%|


## Zoom on Zero-shot classification

The system prompt we use is the following:

```
<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. You are given the content of an email and your answer should tell the user if a followup action is expected from the author of the email or if the email is informative only. Only use answers 'followup' or 'informative'.
<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

We call the API for every test email with the script `notebook/mistral-orca.py`.
Once this is done we persist the results in file `data/mistral_predictions.pq`.

![mistral test](img/mistral_test.png)

![mistral test2](img/mistral_screenshot_replicate.png)

![mistral test3](img/mistral_screenshot_replicate2.png)


# Productionize the SVM model 

After we have installed all the dependencies we can now run the script in train/svm.py, this script takes the input data and outputs a pipeline that includes the cross-validated search and fitting of: 
- a TF-IDF vectorizer working with both unigrams and bigrams
- an SVM classifier which corrects for training set imbalance and uses radial basis function as a kernel.

`python train/svm.py`

![svm train](img/svm_train.png)

The resulting pipeline is persisted on disk in file `model/e2e_pipeline.dat.gz`, so that we can re-instantiate the trained model whenever the web service is spawn.


# Expose Web end point locally

Finally we can test our web application by running:

`flask run -p 5000`

# Build and publish Docker image

Now that we have our web application running, we can use the Dockerfile to create an image for running our web application inside a container

`docker build . -t svm_email_intent_flask_docker`

And now we can test our application using Docker

`docker run -p 8000:8000 svm_email_intent_flask_docker`

Finally we push our image to the official Docker hub:

`docker push svm_email_intent_flask_docker lelayf/svm_email_intent_flask_docker`

This docker image can now be used by on ECS or EKS, behind a load balancer. Ideally we would want to log the queries and predictions in an adequate system to perform ML observability.

![docker push](img/docker_hub.png)


# Testing

A few HTTP requests have been designed for testing purpose. In a terminal, simply run:

`./test/svm_deployment.sh`

![svm test](img/svm_test.png)
