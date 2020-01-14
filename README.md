# Training the ML model

This first part can be done in a jupyter notebook.
This is not a tutorial on machine learning. So we’ll train a model on fictional data. That said, it will function like any other sklearn model you could train.
Our model will detect if an animal is a dog, based on the noise the animal makes.
Create fictional data! Within each inner list, the 1st index is the sound of an animal, the 2nd index is a boolean label indicating if the animal is a dog.


```bash
data = [
    ['woof', 1],
    ['bark', 1],
    ['ruff', 1],
    ['bowwow', 1],
    ['roar', 0],
    ['bah', 0],
    ['meow', 0],
    ['ribbit', 0],
    ['moo', 0],
    ['yip', 0],
    ['pika', 0]
]
```

Convert above into lists of features and labels.

```bash
X = []
y = []
for i in data:
    X.append( i[0] )
    y.append( i[1] )
```

Fit a vectorizer and transform the features.

```bash
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
```

Train a linear regression.

```bash
from sklearn.linear_model import LinearRegression
import numpy as np
regressor = LinearRegression()
regressor.fit(X_vectorized, y)

```

## Pickle our models into a byte stream so we can store them in the app.
```bash
import pickle
pickl = {
    'vectorizer': vectorizer,
    'regressor': regressor
}
pickle.dump( pickl, open( 'models' + ".p", "wb" ) )
```

# Building the django application

Open the command line to the directory where you store your django projects. Create a directory for this application and cd into it.

```bash
mkdir DjangoMLAPI && cd DjangoMLAPI
```

Create a virtual environment and install the required packages.

```bash
python3 -m venv env
source env/bin/activate
pip install django djangorestframework sklearn numpy
```

# Testing the API

```bash 
python manage.py runserver
```

And make a couple curl requests to test it out. You could also directly input the URLs into the browser.


[http://127.0.0.1:8000/classify/?sound=meow](http://127.0.0.1:8000/classify/?sound=meow)


[http://127.0.0.1:8000/classify/?sound=woof](http://127.0.0.1:8000/classify/?sound=woof)


Easy peazy! It’s working! A number close to 1 indicates it’s a dog and a number close to 0 indicates it’s not a dog.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

 
