# How to build a smart application

## Abstract
//todo: motiviere Leser das dokument zu lesen und zeige wie viel arbeit wir investiert haben

//todo: project scope and overview

//anmerkung: schreibe für einen Freund, hilf dem leser sein problem zu lösen

We will try to get an insight of the possible challenges and problems, when building a live application for production that solves a complex machine learning problem, involving both software engineers and data analysts. The smart application will include the following tasks (amongst others):
* Fetching data from different sources
* Preprocessing the data
* Training a machine learning model
* Using the trained model to make live predictions
* Being able to easily replace the prediction model with new model versions

Furthermore we want to demonstrate a full process cycle, helping you to prepare for your own project.

## Transfer a machine learning model between technologies
As you might have experienced, the technologie stack used by data analysts will most certainly not match with what is used for building big applications in a business environment. For example, the data analyst might use R or Python to explore data, extract features and train a machine learning algorithm, but the software engineer wants to build his enterprise application with Java EE or Spring Boot (or other non-Java technologies). Therefore one of the big challenges can be to enable the enterprise application, to use the trained machine learning model for predictions.

For this matter we came up with two different approaches to solve the problem:
* **Model as a Service**: The whole ML model is supposed to be transfered from Python to Java, to execute predictions directly in Java
* **Inference as a Service**: The ML model is supposed to be deployed from Python and an inference service is supposed to be made available as RESTful API

Since todays state-of-the-art machine learning is mostly done either in Python or R, we will describe the possibilities for both those technologies.
#### Use a model, trained in R
##### Inference as a Service
##### Model as a Service
#### Use a model, trained in Python
##### Inference as a Service
##### Model as a Service
###### Deep Neural Network (Tensorflow etc.)
###### Other machine learning models (e.g. Random Forest) 

### Comparison and benchmark analysis

## Glossar
* Partielle Ableitung
* Gradient
* API
* REST
* RESTful API
* Machine Learning
* Deep Learning
* Neural Network
* Random Forest
* Decision Tree
* Linear Regression
* Model