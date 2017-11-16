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
* Building a stable business application around the model and use it for predictions
* Being able to easily replace the prediction model with new model versions

In the following we want to demonstrate a full process cycle, helping you to prepare for your own project.

## Transfer a machine learning model between technologies
As you might have experienced, the technologie stack used by data analysts will most certainly not match with what is used for building big applications in a business environment. The data analyst might use a technology such as R or Python to explore data, extract features and train a machine learning algorithm, but the software engineer wants to build his enterprise application with Java EE, Spring or other technologies. Therefore one of the big challenges can be to enable the enterprise application to use the trained machine learning model for predictions.

For this matter we came up with two different solution approaches:
* **Model as a Service**: The whole machine learning model is supposed to be transferred from Python/R to Java, to directly execute predictions in Java
* **Inference as a Service**: The machine learning model is supposed to be deployed from Python/R and an inference service is supposed to be made available as a RESTful API

Since todays state-of-the-art machine learning is mostly done in either Python or R, we will describe the possibilities for those two technologies.
### Use a model, trained in R <img align="right" width="50" height="50" src="images/Rlogo.png">

[R](http://www.r-project.org/) is known as a language and environment for statistical computing and graphics. It is a free open source software and provides linear and nonlinear modeling, classical statistical tests, time-series analysis, classification and clustering. One of R's strengths is the ease with which well-designed publication-quality plots can be produced, including mathematical symbols and formulae. At first, R was primarily used in academics and research, but lately the enterprise world is discovering R as well. This makes R one of the fastest growing statistical languages in the corporate world. Besides, R has an extensive documentation and an active online community support. All in all it is the perfect environment to get started in statistical computing and which is used by many data scientists around the world.

For a detailed introduction the manual ["An Introduction to R"](https://cran.r-project.org/doc/manuals/r-release/R-intro.pdf) is recommended.

We recommend to install the latest free open source [R software](https://cran.r-project.org/) and integrate a development environment (IDE) for R like [RStudio](https://www.rstudio.com/).

#### Inference as a Service
//todo: *docu, (dis)advantages, (links to) code and/or pictures*
#### Model as a Service
We show now how we transferred one whole machine learning (ML) model from R to Java, to execute predictions directly in Java. We choose as ML model the "oldschool" Random Forest, that solve the classification problem for the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) data set. In the MNIST data set one image has 28*28 = 784 pixels with values between 0 and 255. The set is organized row-wise. 

We start training our model in R. It is very important that the trained model should be generated with input data using the formula interface e.g. 
s
```
randomForest( Y ~ . , d.train) 
``` 

instead of 

```{r}
randomForest(x = d.train[, -785], y = d.train[, 785])` ,
```

where ` d.train[, -785]` is our training set and `d.train[, 785]` are the labels for each pixel of this training set. This is one bug which is still in the R package "pmml" (version 1.5.2). Hopefully, it will be solved in the future.

By saving this trained model as a *.pmml file* we have already done the wanted. Okay, it is not so simple but with this following code which convert the model to pmml and save it as *"model.pmml"* most of the work is done:

```{r}
# load xml and pmml library
library(XML)
library(pmml)

# convert model to pmml
model.pmml <- pmml(model, name = "MNIST Random Forest", data = model)

# save to file "model.pmml"
saveXML(model.pmml, file = "/path_where_it_should_be_saved/model.pmml"))
```
Now you can decide in which programming language you want to run your prediction. We choose Java :smile:

Therefore, you have to run the [MNISTClassifier](https://github.com/IndustrialML/R2Java/blob/master/MLJava/src/main/java/RandomForest/MNISTClassifier.java) class. For the full documentation on the `MNISTClassifier` please refer to the [Python2Java](https://github.com/IndustrialML/Python2Java/tree/master/MaschineLearning4J/src/main/java/RandomForest) repository. For now we will try to make you familiar with its usage. We are using the `mvn exec:java` command to run the program. For `Maven` to know which method to run, we need to specify a `mainClass` and pass this argument: `-Dexec.mainClass="RandomForest.MNISTClassifier"`
You can and will need to pass additional program parameters to the class by adding the `-Dexec.args="..."` argument to the `mvn exec:java` command. You can pass in following arguments to the program:
```bash
usage: MNISTClassifier
 -b,--benchmark-test   pass, if you want to run and print the benchmark
                       test
 -c,--compare          pass, if you want the results from Python and Java
                       to be compared for sameness. This action cannot be
                       performed for the R-technology
 -h,--help             print this info message again
 -n,--n-trees <arg>    number of decision trees that were used to create
                       the PMML. Please check for which number of trees,
                       you have a valid PMML file on your system. Default
                       value will be 50
 -p,--picture <arg>    name of the picture, to predict its displayed
                       digit. <Arg> can either be a full path or a file in
                       the Data/Own_dat directory of this project. Default
                       value will be R2Java/MLJava/MNIST-7.png
 -r,--use-R            pass, if the pmml you want to load is from a model,
                       created with R
```
If you are sticking to the following rules, you have made it!

* *Always*! use the `-r` parameter
* Never use the `-c` parameter!
* Make sure to pass in a valid value to the `-n` parameter. Do not pass it in if you want to use the default of 50 trees
* If you use the `-p` parameter, you need to pass in an absolute path to a .png file

For more details we refer to [R2Java](https://github.com/IndustrialML/R2Java).

##### (Dis)Advantages

As you might guess, transferring a ML model needs to save the whole model which may require a lot of memory space. It takes some time to save it - believe us. Furthermore, one can get in trouble to push it to github or send it, e.g. our model is over 786000 KB large. Further, you need a shared folder or any other service to transfer the model, if you don't want to use it yourself locally. It is no REST API and hence not available everywhere. But this means you have control over who get it and use it. Another advantage is that it is very easy to convert the model to pmml and the *.pmml file* can not only be used by Python. Thus, it is easily transferable to other programming languages.

### Use a model, trained in Python
Python is an interpreted, high-level, multipurpose programming language with a simple and easy to learn syntax, which is one of the contributing factors to Pythons recent and continuous success. Python has a solid claim to being the fastest-growing major programming language. You can see on [Stack Overflow Trends](https://insights.stackoverflow.com/trends?tags=python%2Cjava%2Cr%2Cc%23%2Cjavascript%2Cscala&utm_source=so-owned&utm_medium=blog&utm_campaign=gen-blog&utm_content=blog-link&utm_term=incredible-growth-python) that Python has been growing rapidly in the last few years:

> ![StackOverflow Pythons trend graphic](https://github.com/IndustrialML/SmartApplicationDoku/blob/master/images/stackoverflowTrends.png)

When we focus on countries that are classified as high-income by the World Bank, the growth of Python is even larger than it might appear from tools like Stack Overflow Trends, or in other rankings that consider global software development. In June 2017, Python even claimed the title for the most visited tag on Stack Overflow within high-income nations. 
This might correlate with the strong upward trend of Machine Learning and Deep Learning as a bunch of strongly supported and mighty packages are available for Python, such as `SciPy`, `Scikit-learn`, `Theano` and of corse `Tensorflow`. Beyond that, Python has always been a strong programming language for exploring data and doing statistical analysis, as Python is very quick to write and change, and provides strong packages for scientific computing, even with multi dimensional objects, such as `NumPy`.

When querying [indeed.com](https://www.indeed.com/jobtrends/q-python-and-(%22machine-learning%22-or-%22data-science%22)-q-R-and-(%22machine-learning%22-or-%22data-science%22)-q-Java-and-(%22machine-learning%22-or-%22data-science%22)-q-scala-and-(%22machine-learning%22-or-%22data-science%22).html) for jobs in machine learning / data science, we can see that Python is the language with highest demand:

> ![Indeed trends graphic](https://github.com/IndustrialML/SmartApplicationDoku/blob/master/images/mlTrend.png)

So now, that we know that Python is great and used all over the place, lets dive in to how we can use our machine learning algorithms, that we trained with Python, in Java.
I have documented my work in more detail [here](https://github.com/IndustrialML/Python2Java) and if you are interested you are more than welcome to read the whole thing. In the following I will link references whenever appropriate. 

#### Saving and exporting a model
Before we can start using a model, in whatever way, we will have to train one and somehow export it, so that we can use it at a later point in time. As this will be different for each Python framework, I will supply you with a few different options. The main difference between models is going to be, whether they implement a deep learning or a "usual" machine learning algorithm. For the sake of demonstration, I will show you how to export a Neural Network and an "oldschool" Random Forest, that solve the classification problem for the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.
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
