# How to build a smart application

//anmerkung: schreibe für einen Freund, hilf dem Leser sein problem zu lösen

## Abstract
//todo: motiviere Leser das dokument zu lesen und zeige wie viel arbeit wir investiert haben

This document has been created as a collaborative effort by Sonja Gassner, Veronica Pohl and Matthias Leopold for internal use at the Zuehlke Engineering AG. It will be most interesting for those, who are about to start a project, where the objective involves building a business application around a machine learning task.

We will try to get an insight of the possible challenges and problems, when building a *smart application* for production that solves a complex machine learning problem, involving both software engineers and data analysts. Moreover, we will  focus on the collaboration between those two groups in particular and try to point out the communication difficulties.

We define a *smart application* as an application which includes the following tasks (amongst others):
* Fetching data from different sources
* Preprocessing the data
* Training a machine learning model
* Building a stable businnes application around the model and use it for predictions
* Being able to easily replace the prediction model with new model versions

In the following we want to demonstrate a full process cycle, helping you to prepare for your own project.

## Transfer a machine learning model between technologies
As you might have experienced, the technologie stack used by data analysts will most certainly not match with what is used for building big applications in a business environment. For example, the data analyst might use R or Python to explore data, extract features and train a machine learning algorithm, but the software engineer wants to build his enterprise application with Java EE or Spring Boot (or other non-Java technologies). Therefore one of the big challenges can be to enable the enterprise application to use the trained machine learning model for predictions.

For this matter we came up with two different solution approaches:
* **Model as a Service**: The whole ML model is supposed to be transfered from Python to Java, to execute predictions directly in Java
* **Inference as a Service**: The ML model is supposed to be deployed from Python and an inference service is supposed to be made available as RESTful API

Since todays state-of-the-art machine learning is mostly done either in Python or R, we will describe the possibilities for both those technologies.
### Use a model, trained in R
//todo: *insert R introduction here*
#### Inference as a Service
//todo: *docu, (dis)advantages, (links to) code and/or pictures*
#### Model as a Service
//todo: *docu, (dis)advantages, (links to) code and/or pictures*

### Use a model, trained in Python
//todo: *insert Python introduction here*
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
