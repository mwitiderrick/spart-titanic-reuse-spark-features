# Titanic Survival Example

A classification example with `Spark ML` for predicting the survivals of the Titanic passengers. We will be using the famous [Kaggle Titanic](https://www.kaggle.com/c/titanic/data?select=train.csv) dataset.

## What we are going to learn?

- Feature Store: We are going to use PySpark interface to build the `passenger` features 
- Utilize `SparkSession` from Layer `Context` for Spark SQL queries (i.e. `title` feature)
- Convert Spark DataFrames into Pandas DataFrames as the way Layer stores the features
- Load `passenger` features and use it to train our `survival` model
- Experimentation tracking with
    - logging `BinaryClassificationEvaluator` metric

## Installation & Running

To check out the Layer Titanic Survival example, run:

```bash
layer clone https://github.com/layerml/examples
cd examples/titanic-spark
```

To run the project:

```bash
layer start
```

## File Structure

```yaml
.
├── .layer
├── data
│   ├── passenger_features	        # feature definitions
│   │   ├── ageband.py				# Age Band of the passenger
│   │   ├── embarked.py  			# Embarked or not
│   │   ├── fareband.py			    # Fare Band of the passenger
│   │   ├── is_alone.py			    # Is Passenger travelling alone
│   │   ├── sex.py				    # Sex of the passenger
│   │   ├── survived.py 			# Survived or not
│   │   ├── title.py				# Title of the passenger
│   │   └── requirements.txt		# Environment config file
│   │   └── dataset.yml				# Declares the metadata of the features above
│   └── titanic_data
│       └── dataset.yml				# Declares where our source `titanic` dataset is
├── models
│   └── survival_model
│       ├── model.yml				# Training directives of our model
│       ├── model.py				# Source code of the `Survival` model
│       └── requirements.txt		# Environment config file
└── README.md
```

