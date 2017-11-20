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
As you might have experienced, the technology stack used by data analysts will most certainly not match with what is used for building big applications in a business environment. The data analyst might use a technology such as R or Python to explore data, extract features and train a machine learning algorithm, but the software engineer wants to build his enterprise application with Java EE, Spring or other technologies. Therefore one of the big challenges can be to enable the enterprise application to use the trained machine learning model for predictions.

For this matter we came up with two different solution approaches:
* **Model as a Service**: The whole machine learning model is supposed to be transferred from Python/R to Java, to directly execute predictions in Java
* **Inference as a Service**: The machine learning model is supposed to be deployed from Python/R and an inference service is supposed to be made available as a RESTful API

Since todays state-of-the-art machine learning is mostly done in either Python or R, we will describe the possibilities for those two technologies.
### Use a model that was trained in R <img align="right" width="50" height="50" src="images/Rlogo.png">

[R](http://www.r-project.org/) is known as a language and environment for statistical computing and graphics. It is a free open source software and provides linear and nonlinear modeling, classical statistical tests, time-series analysis, classification and clustering. One of R's strengths is the ease with which well-designed publication-quality plots can be produced, including mathematical symbols and formulae. At first, R was primarily used in academics and research, but lately the enterprise world is discovering R as well. This makes R one of the fastest growing statistical languages in the corporate world. Besides, R has an extensive documentation and an active online community support. All in all it is the perfect environment to get started in statistical computing and which is used by many data scientists around the world.

For a detailed introduction the manual ["An Introduction to R"](https://cran.r-project.org/doc/manuals/r-release/R-intro.pdf) is recommended.

We recommend to install the latest free open source [R software](https://cran.r-project.org/) and integrate a development environment (IDE) for R like [RStudio](https://www.rstudio.com/).

#### Inference as a Service
//todo: *docu, (dis)advantages, (links to) code and/or pictures*
#### Model as a Service
We show now how we transferred one whole machine learning (ML) model from R to Java, to execute predictions directly in Java. We choose as ML model the "old-school" Random Forest, that solve the classification problem for the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) data set. In the MNIST data set one image has 28*28 = 784 pixels with values between 0 and 255. The set is organized row-wise. 

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

### Use a model that was trained in Python
Python is an interpreted, high-level, multipurpose programming language with a simple and easy to learn syntax, which is one of the contributing factors to Pythons recent and continuous success. Python has a solid claim to being the fastest-growing major programming language. You can see on [Stack Overflow Trends](https://insights.stackoverflow.com/trends?tags=python%2Cjava%2Cr%2Cc%23%2Cjavascript%2Cscala&utm_source=so-owned&utm_medium=blog&utm_campaign=gen-blog&utm_content=blog-link&utm_term=incredible-growth-python) that Python has been growing rapidly in the last few years:

> ![StackOverflow Pythons trend graphic](https://github.com/IndustrialML/SmartApplicationDoku/blob/master/images/stackoverflowTrends.png)

When we focus on countries that are classified as high-income by the World Bank, the growth of Python is even larger than it might appear from tools like Stack Overflow Trends, or in other rankings that consider global software development. In June 2017, Python even claimed the title for the most visited tag on Stack Overflow within high-income nations. 
This might correlate with the strong upward trend of Machine Learning and Deep Learning as a bunch of strongly supported and mighty packages are available for Python, such as `SciPy`, `Scikit-learn`, `Theano` and of corse `Tensorflow`. Beyond that, Python has always been a strong programming language for exploring data and doing statistical analysis, as Python is very quick to write and change, and provides strong packages for scientific computing, even with multi dimensional objects, such as `NumPy`.

When querying [indeed.com](https://www.indeed.com/jobtrends/q-python-and-(%22machine-learning%22-or-%22data-science%22)-q-R-and-(%22machine-learning%22-or-%22data-science%22)-q-Java-and-(%22machine-learning%22-or-%22data-science%22)-q-scala-and-(%22machine-learning%22-or-%22data-science%22).html) for jobs in machine learning / data science, we can see that Python is the language with highest demand:

> ![Indeed trends graphic](https://github.com/IndustrialML/SmartApplicationDoku/blob/master/images/mlTrend.png)

So now, that we know that Python is great and used all over the place, lets dive in to how we can use our machine learning algorithms, that we trained with Python, in Java.
I have documented my work in more detail [here](https://github.com/IndustrialML/Python2Java) and if you are interested you are more than welcome to read the whole thing. In the following I will link references whenever appropriate. 

#### Saving and exporting a model
Before we can start using a model, in whatever way, we will have to train one and somehow export it, so that we can use it at a later point in time. As this will be different for each Python framework, I will supply you with a few different options. The main difference between models is going to be, whether they implement a deep learning or a "usual" machine learning algorithm. For the sake of demonstration, I trained and exported a Neural Network and a Random Forest, that solve the classification problem for the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset, the full examples can be found [here](https://github.com/IndustrialML/Python2Java/tree/master/Maschine%20Learning).
##### Neural Network
As `Tensorflow` emerges as by far the most popular deep learning framework, I will join the majority. The easiest way to save a trained model in `tensorflow` is, to use the `tensorflow.train.Saver()` class, provided directly from `tensorflow`. An example workflow is shown [here](https://github.com/IndustrialML/Python2Java/tree/master/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier#saver), however as this method does not save the model platform independent, but rather only to re-import it into Python later, this is not what we are looking for. Another option would be to make use of the [JPMML-Tensorflow](https://github.com/jpmml/jpmml-tensorflow) project, to parse the trained model to the [PMML](http://dmg.org/pmml/v4-3/GeneralStructure.html) standard format, and use the [JPMMLEvaluator](https://github.com/jpmml/jpmml-evaluator) to read out the PMML file and recreate the model in Java. This option would be somewhat platform independent, but heavily relies on the third party `JPMML` project and the import would be restricted to Java only.

Hence, I am gladly proposing a third method, which limits itself to only use the `Tensorflow` framework for the transfer and offers greater platform independence, even though i will only present you with a Java example. The point of matter here is, that `Tensorflow` officially supports different API’s for different languages. According to the official [documentation](https://www.tensorflow.org/api_docs/), you can definitely use C++ and Go, aswell as some other languages with weaker support, like Scala or Ruby. This method will use the `tensorflow.saved_model` class, for which an official introduction can be found [here](https://www.tensorflow.org/programmers_guide/saved_model#apis_to_build_and_load_a_savedmodel). 

The export of a neural network, built with Tensorflow, can be easily accomplished with:
```python
    export_dir = "./export"
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess, 
            [tf.saved_model.tag_constants.SERVING])
    builder.save()
```
The last line of codes, executes the export and creates files, representing the neural network, inside the "export" directory. It creates the architecture describing "saved_model.pb" file, and another subdirectory, containing all information about the weights and biases.

Now, we could conclude this section without further ado, but often it is advisable to add a `Signature` to the `SavedModel`, describing the set of inputs and outputs of a model. This step is important, because we will need to know the exact names of the input and output structures, in order to being able to use the model for prediction in Java, as we will show later. We can add a signature by modifying the code above:

```python
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input': tf.saved_model.utils.build_tensor_info(x),
                'dropout': tf.saved_model.utils.build_tensor_info(dKeep)},
        outputs={'output': tf.saved_model.utils.build_tensor_info(y)},
    )
    signatureMap = { tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature }

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess, 
        [tf.saved_model.tag_constants.SERVING], 
        signature_def_map=signatureMap)
    builder.save()
```
Note, that `x` is the `placeholder` for the input layer, `dKeep` is the `placeholder` for the dropout rate, and `y` is the output layer of the neural network.


This signature can then be inspected, using the [SavedModel CLI](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel). For our example neural net, the output can look as following: 

> ![Image of SavedModel CLI Output](https://github.com/IndustrialML/Python2Java/blob/master/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier/Feed%20Forward%20NN/SavedModel/SavedModelCLI_example.png)

For more information and a full work through, please refer to the Python2Java repository [here](https://github.com/IndustrialML/Python2Java/tree/master/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier).

As mentioned earlier, performing the export is going to be dependent on which framework you are using to train your model. This introduction covered only plain `Tensorflow`, if you want to, or have already, build a neural network with  `Tensorflow.Estimator` or `Keras` with `Tensorflow` backend, I have covered these in the Python2Java repository aswell. Please follow the link to the documentation for [Estimator](https://github.com/IndustrialML/Python2Java/tree/master/Maschine%20Learning/NeuralNetwork/Estimator/MNISTClassifier) or [Keras](https://github.com/IndustrialML/Python2Java/tree/master/Maschine%20Learning/NeuralNetwork/Keras).

##### Random Forest
Just like `Tensorflow` seems to be the top dog for building neural networks in Python, `scikit-learn` (or short: `sklearn`)  looks to be its counterpart for basically everything else in Python's machine learning. Therefore, I will now show you how to export an `sklearn.ensemble.RandomForestClassifier`. The workflow will be mostly the same for every other `sklearn` classifier.

If you are only looking to save your model, to re-import it into Python at a later point in time, for example to use it to provide a simple RESTfull API, this is a very easy task. Most of the `sklearn` classifiers are "pickleable"(if that is a word), which means that you can simply save your whole classifier to a ".pkl" file, using Pythons build in serializing and de-serializing module `pickle`. In the specific case of an `sklearn` classifier, it may be more interesting to use  `joblib`’s replacement of `pickle`, which is more efficient on objects that carry large `numpy` arrays. This can look like the following:
```python
    clf = RandomForestClassifier(...)
    clf.fit(features, labels)
    joblib.dump(clf, "export.pkl")
```
And that's it, now you have saved your whole `RandomForestClassifier` in the "export.pkl" file inside your working directory. 

Well that was boring, but wait a moment, remember that `pickle` is a Python’s built in tool, which means that it definitely is specific to the language of Python. For the idea of *Model as a Service*, we will need a platform independent format, that we can read into Java. Sadly `sklearn` does not offer convenient API’s in other languages, like `Tensorflow` does. Thus, we will need to fall back to the `JPMML` project, which was mentioned earlier. They characterize their project as a set of "PMML producer and consumer libraries for the Java/JVM platform", which includes the [sklearn2pmml](https://github.com/jpmml/sklearn2pmml) module. This is a Python library for converting Scikit-Learn pipelines to PMML. Note, that we will later be using the [jpmml-evaluator](https://github.com/jpmml/jpmml-evaluator), to load the created PMML file into Java. 

Converting your `sklearn` model into a representational PMML file takes three steps:
1. Build a `PMMLPipeline` around your classifier:
```python
    clf = RandomForestClassifier(...)
    mnist_pipeline = PMMLPipeline([
        ("classifier", clf)
    ])
```
2. Train the model inside the pipeline (therefore you need to call the fit() function on the pipeline, not the original classifier):
```python
    mnist_pipeline.fit(features, labels)
```
3. Parse the Python `PMMLPipeline` object to an actual PMML file:
```python
    sklearn2pmml(mnist_pipeline, "RandomForest.pmml")
```
For more information and a full example, please refer to the [Python2Java](https://github.com/IndustrialML/Python2Java/tree/master/Maschine%20Learning/RandomForest/MaaS) repository.


#### Inference as a Service
Before we begin, let me remind you what we understand under the term of Inference as a Service: *The machine learning model is supposed to be deployed from Python/R and an inference service is supposed to be made available as a RESTful API*.

In order to achieve this, I will be using the convenient [flask](http://flask.pocoo.org/) Python microframework. For now, I will illustrate the general structure of a `flask` application and later on show the different implementations of loading the pretrained model and making a prediction with either `Tensorflow` or `Sklearn`.

1. At the very beginning, you will need to create your `flask` application:
```python
    app = Flask(__name__)
```
2. Reload the pretrained model (model dependent step)
3. Define the route of your REST endpoint, by adding a function decorator:
```python
    @app.route("/predict", methods=['POST'])
    def predict():
        picArray = request.get_json() #get request json Array
        reshaped_Array = reshapePic(picArray) #reshape the Array to your needs
        return predictPicture(reshaped_Array) #model dependent step
```

##### Neural Network
Let's assume, that we have saved a trained `Tensorflow` model, like the one shown [earlier](https://github.com/IndustrialML/SmartApplicationDoku#neural-network). It has one input tensor (input:0), one output tensor (output:0) and a dropout rate tensor (dropoutRate:0). Now the import will look like the following: 
```python
    sess = tf.Session()
    import_dir = "../path/to/savedmodel/directory"
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], import_dir)
    graph = tf.get_default_graph()
```
And with this, the whole model that we saved earlier, will be recreated into the new Tensorflow `Session`. Notice that we are using the `[tf.saved_model.tag_constants.SERVING]` again, to specify which meta graph inside the `SavedModel` to load.

Do you remember how i pointed our earlier, that it is important for us to know the exact names of these tensors and that this is the reason why we will need to use a `Signature`? Well, now we will need to grab these tensors by name from the `Graph`, in order to being able to feed the model with input values and fetch the result:
```python
	x = graph.get_tensor_by_name("input:0")
	dKeep = graph.get_tensor_by_name("dropoutRate:0")
	y = graph.get_tensor_by_name("output:0")
``` 
After we can access these tensors, we can start actually using our model:
```python
    def predictPicture(picArray):
        score = sess.run(y, feed_dict={x: picArray, dKeep: 1.0})
        return np.argmax(score)
```
For a full running example with different `Tensorflow` aswell as `Keras` and `Estimator` models, please refer to [this](https://github.com/IndustrialML/Python2Java/tree/master/Maschine%20Learning/NeuralNetwork/Serving).

**Appendix:** If you are looking to serve a model for production that is going to be retrained frequently, may require serving multiple versions simultaneously and/or is expected to receive an extremely high amount of requests, you might want to get yourself into [Tensorflow Serving](https://www.tensorflow.org/serving/) instead of using Flask. This is a separate and very mighty module of Tensorflow, specifically created to serve Tensorflow models. Making a proper request to the Tensorflow Server from a different technology will require you converting your inputs into a fitting protobuffer format and sending that per RPC though. This can be a bit of work, but you can get a good impression for it, by reading Florian Dahm's [blog](http://fdahms.com/2017/03/05/tensorflow-serving-jvm-client/). For Python, Tensorflow offers a working API, which does all the transformations for you. A small example for that can be found [here](https://github.com/IndustrialML/experiments/tree/master/Tensorflow_Serving).

##### Random Forest
Saving an `sklearn` model by using `pickle` was super easy, and so is re-loading the pickled file into Python. We will again be using `joblib` instead of `pickle` for performance reasons, but they basically share the same syntax:
```python
    clf = joblib.load(file)
```
After reloading the classifier from the .pkl file, you can directly start using the model:
```python
    def predictPic(pngArray):
        prediction = clf.predict(pngArray)
        score = clf.predict_proba(pngArray) #optional
        return prediction
```
And that's it. For a full working example, please refer to [this](https://github.com/IndustrialML/Python2Java/tree/master/Maschine%20Learning/RandomForest/IaaS).

#### Model as a Service
As a different approach on transferring a machine learning model between technologies, we came up with the idea of Model as a Service, with what we mean: *The whole machine learning model is supposed to be transferred from Python/R to Java, to directly execute predictions in Java*. This has several advantages, such as the possibility to work without any network connection and, associated therewith, a significant reduction of prediction time length. You will be able to get a better impression on the prediction time differences between the possible options, at the [comparison and benchmark analysis](https://github.com/IndustrialML/SmartApplicationDoku#comparison-and-benchmark-analysis) section. On the other hand though, the transfer from Python to Java will take a lot longer, which will hinder the replacement of a model with a new version. Furthermore, the question of how to handle the preprocessing for a prediction in Java will become a significant problem, as it becomes more complicated or even impossible to use the same pipeline that we used for training. 

Since the saving of a `Tensorflow` model significantly differs from saving a  `Sklearn` model, the import of these models will have to be handled independently aswell.

##### Neural Network
Recall, that we saved our `Tensorflow` neural network, using `Tensorflow`'s `SavedModel` class. This created a protobuffer file for the meta graph, and a directory in which all the variable where stored. But fortunately we do not have to unwrap these ourselves, but can rather use the convenient `Tensorflow ` [Java API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary). We can install the needed dependency with Maven:
```Maven
    <dependency>
        <groupId>org.tensorflow</groupId>
        <artifactId>tensorflow</artifactId>
        <version>1.3.0</version>
    </dependency>
```
`Tensorflow` makes your lives easy, if you are familiar with the usual Python API, and offers you the same class constructs in Java as in Python. We can create a Java `Session` object from our exported `SavedModel`, by using the `SavedModelBundle` class:
```Java
    SavedModelBundle sb = SavedModelBundle.load(importDir, ModelTag);
    Session session = sb.session();
```
* `importdir` has to be String, containing the path of the directory of where to load the `SavedModel` from. Which will be the directory, containing the "saved_model.pb" file and the "variables" sub-directory.
* `ModelTag` hast to be String, containing the tag of the associated `MetaGraph`, which we want to load from the `SavedModel`. For our example we used the `Tensorflow` constant for serving, which is the String "serve" as only item inside an array.

Now let me remind you, that it is very important to know the exact names of the input and output tensors inside the SavedModel, to use the loaded Session for predictions. These names will be dependent on the way you that you saved your model.
For my example, the important Tensor names look as follows:
```Java
    private static final String input = "input:0";
    private static final String output = "output:0";
    private static final String dropout = "dropoutRate:0";
```

Now you are ready, to load your input parameters and make predictions. For my example, this will be loading the pixel information of an image into a float[]. Afterwards you can load that array into a `Tensor`, feed it to the `Session` and fetch the output `Tensor`:
```Java
    float[] inputArray = TensorflowUtilities.readPic(pathfile);
    Tensor inputTensor = TensorflowUtilities.toTensor(inputArray); #create Tensor from float[]
    Tensor dKeep = Tensor.create(1f);
    
    List<Tensor> resultList = this.session.runner().feed(input, inputTensor)
    												.feed(dropout, dKeep)
                                                    .fetch(output).run(); #actual prediction is executed here
```
Afterwards you can read out the `resultList`, which will contain the corresponding score/probabilities for the prediction of the given image.

I strongly recommend you to take a look at the full implementation and documentation of the example [here](https://github.com/IndustrialML/Python2Java/tree/master/MaschineLearning4J/src/main/java/NeuralNetwork/Tensorflow). There you can find an explanation on how to use a model that was trained with `Keras` or `Tensorflow.Estimator` aswell.

##### Random Forest
Now we are almost ready to conclude this section, with only one more technique to go. Same as for the neural network, it is important for us to remember the way how we saved our model [initially](https://github.com/IndustrialML/SmartApplicationDoku#random-forest). As `Sklearn` does not support multiple languages itself, we fell back to using the [sklearn2pmml](https://github.com/jpmml/sklearn2pmml) Python library and parsed the random forest classifier to a PMML file. As mentioned earlier, we will now be using the [jpmml-evaluator](https://github.com/jpmml/jpmml-evaluator) to read the PMML file into Java and use the created `Evaluator` object to make predictions. To  begin with, we can install the needed dependency with Maven again:
```Maven
    <dependency>
        <groupId>org.jpmml</groupId>
        <artifactId>pmml-evaluator</artifactId>
        <version>1.3.9</version>
    </dependency>
```

In order to create a `jpmml-evaluator` from the saved PMML file, you can follow this workflow:
```java
    InputStream is = new FileInputStream(pmmlFilepath);
    PMML pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
    ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
    Evaluator modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
```
After creating the `Evaluator`, we can now use it to pass input parameters to our previously trained model, and fetch the output value as prediction. Preparing our input parameters for the `Evaluator` is a little bit more complex than what we did for the neural network, due to the fact that the `Evaluator` does not simply accept an array-like object (e.g. a `Tensor`) as input, but rather requires a Map of `FieldName`->`FieldValue`. Each `FieldName` will represent one feature of the previously saved model, for us these will be `x1` to `x784`, each feature representing the grayscale value of one pixel of a 28x28 picture. So, to cut a long story short, this is how it's done:
```Java
    float[] picture = TensorflowUtilities.readPic(picPath);
    
    List<InputField> inputFields = this.evaluator.getInputFields();
    Map<FieldName, FieldValue> inputArguments = new LinkedHashMap<>();
    for (int i = 0; i < inputFields.size(); i++) {
        FieldName inputFieldName = inputFields.get(i).getName();//xi with i in int(1,784)
        int index = Integer.valueOf(inputFieldName.toString().substring(1)); //i in int(1,784)
        Object rawValue = picture[index - 1]; //-1 because indices of DataFields are in int(1,784) and array indices are in int(0,783)
        FieldValue inputFieldValue = inputFields.get(i).prepare(rawValue);
        inputArguments.put(inputFieldName, inputFieldValue);
    }
```
After we have correctly prepared our input parameters, we can start using the `Evaluator` to make a prediction and read out the resulting output:
```Java
    Map<FieldName, ?> results = this.evaluator.evaluate(inputArguments); //actually calling the model for prediction
   
    List<TargetField> targetFields = this.evaluator.getTargetFields(); //get all output fields
    TargetField targetField = targetFields.get(0); //we only have one output value though
    FieldName targetFieldName = targetField.getName(); //get the name of the output field
	
    ProbabilityDistribution targetFieldValue = (ProbabilityDistribution) results.get(targetFieldName); //get the value of the output field as (probability for each class)
    Object predictionObj = targetFieldValue.getResult(); //result of a ProbabilityDistribution is the class with the highest probability
```
Now finally, this will be the last time, i will advice you to read my full documentation. For using a random forest's PMML file to create a `jpmml-evaluator`, you can find my example [here]() or the official, more detailed example, at the jpmml github repository [here](https://github.com/jpmml/jpmml-evaluator). 


### Comparison and benchmark analysis
//todo: *final summary+comparison and benchmark results*

## Glossar
* API
* Decision Tree
* Deep Learning
* Linear Regression
* Machine Learning
* Machine Learning Process (preprocessing->training/fitting->predicting)
* Model
* Neural Network
* Random Forest
* REST
* RESTful API
