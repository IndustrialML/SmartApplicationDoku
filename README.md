# How to build a smart application

//anmerkung: schreibe für einen Freund, hilf dem Leser sein problem zu lösen

## Abstract
//todo: motiviere Leser das dokument zu lesen und zeige wie viel arbeit wir investiert haben

This document has been created as a collaborative effort by Sonja Gassner, Veronica Pohl and Matthias Leopold for internal use at the Zuehlke Engineering AG. It will be most interesting for those who are about to start a project, where the objective involves building a business application around a machine learning task.

We will try to get an insight of the possible challenges and problems, when building a *smart application* for production that solves a complex machine learning problem, involving both software engineers and data analysts. Moreover, we will  focus on the collaboration between those two groups in particular and try to point out the communication difficulties.

We define a *smart application* as an application which includes the following tasks (amongst others):
* Fetching data from different sources
* Preprocessing the data
* Training a machine learning model
* Building a stable businnes application around the model and use it for predictions
* Being able to easily replace the prediction model with new model versions

In the following we want to demonstrate a full process cycle, helping you to prepare for your own project.

## Transfer a machine learning model between technologies
As you might have experienced, the technologie stack used by data analysts will most certainly not match with what is used for building big applications in a business environment. The data analyst might use a technology such as R or Python to explore data, extract features and train a machine learning algorithm, but the software engineer wants to build his enterprise application with Java EE, Spring or other technologies. Therefore one of the big challenges can be to enable the enterprise application to use the trained machine learning model for predictions.

For this matter we came up with two different solution approaches:
* **Model as a Service**: The whole machine learning model is supposed to be transfered from Python/R to Java, to directly execute predictions in Java
* **Inference as a Service**: The machine learning model is supposed to be deployed from Python/R and an inference service is supposed to be made available as a RESTful API

Since todays state-of-the-art machine learning is mostly done in either Python or R, we will describe the possibilities for those two technologies.
### Use a model, trained in R
//todo: *insert R introduction here*
#### Inference as a Service
//todo: *docu, (dis)advantages, (links to) code and/or pictures*
#### Model as a Service
//todo: *docu, (dis)advantages, (links to) code and/or pictures*

### Use a model, trained in Python
Python is an interpreted, high-level, multipurpose programming language with a simple and easy to learn syntax, which is one of the contributing factors to Pythons recent and continuous success. Python has a solid claim to being the fastest-growing major programming language. You can see on [Stack Overflow Trends](https://insights.stackoverflow.com/trends?tags=python%2Cjava%2Cr%2Cc%23%2Cjavascript%2Cscala&utm_source=so-owned&utm_medium=blog&utm_campaign=gen-blog&utm_content=blog-link&utm_term=incredible-growth-python) that Python has been growing rapidly in the last few years:

> ![StackOverflow Pythons Trend graphic](https://github.com/IndustrialML/SmartApplicationDoku/images/stackoverflowTrends.png)

When we focus on countries that are classified as high-income by the World Bank, the growth of Python is even larger than it might appear from tools like Stack Overflow Trends, or in other rankings that consider global software development. In June 2017, Python even claimed the title for the most visited tag on Stack Overflow within high-income nations. 

This might correlate with the strong upward trend of Machine Learning and Deep Learning as a bunch of strongly supported and mighty packages are available for Python, such as `SciPy`, `Scikit-learn`, `Theano` and of corse `Tensorflow`. Beyond that, Python has always been a strong programming language for exploring data and doing statistical analysis, as Python is very quick to write and change, and provides strong packages for scientific computing, even with multi dimensional objects, such as `NumPy`.

When querying [indeed.com](https://www.indeed.com/jobtrends/q-python-and-(%22machine-learning%22-or-%22data-science%22)-q-R-and-(%22machine-learning%22-or-%22data-science%22)-q-Java-and-(%22machine-learning%22-or-%22data-science%22)-q-scala-and-(%22machine-learning%22-or-%22data-science%22).html) for jobs in machine learning / data science, we can see that Python is the language with highest demand:

> ![indeed trends graphic](https://github.com/IndustrialML/SmartApplicationDoku/images/mlTrend.png)

So now, that we know that Python is super cool and used all over the place, lets dive in to how we can use our machine learning algorithms, that we trained in Python, in Java.

#### Saving and exporting a model
##### Neural Network
//todo: *docu, (dis)advantages, (links to) code and/or pictures*
##### Random Forest
//todo: *docu, (dis)advantages, (links to) code and/or pictures*

#### Inference as a Service
//todo: *docu, (dis)advantages, (links to) code and/or pictures*
#### Model as a Service
//todo: *docu, (dis)advantages, (links to) code and/or pictures*

### Comparison and benchmark analysis
//todo: *final summary+comparison and benchmark results*

## Glossar
* API
* Decision Tree
* Deep Learning
* Fitting
* Gradient
* Linear Regression
* Machine Learning
* Machine Leagning Process (preprocessing->training->predicting)
* Model
* Neural Network
* Partielle Ableitung
* Random Forest
* REST
* RESTful API
* Training
* Training / Testing / Validation datasets
