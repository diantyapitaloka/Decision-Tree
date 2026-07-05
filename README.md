## 🍬🥧🍰 Decision Tree 🍰🥧🍬
- Input Features: Your flower has a sepal length of 6.2cm, sepal width of 3.4cm, petal length of 5.4cm, and petal width of 2.3cm. These specific measurements act as the variables that the model evaluates at each node.
- Classifying the data we have with the Decision Tree technique uses the iris dataset, one of the most popular datasets that is often used for learning machine learning. Want to predict the species of an iris flower that has a sepal length of 6.2 centimeters, a sepal width of 3.4 centimeters, a petal length of 5.4 centimeters, and a petal width of 2.3 centimeters.
- Petal Dimension Significance: In the iris dataset, petal length and width are usually the most informative features for splitting the data. Measurements like your 5.4cm petal length and 2.3cm petal width strongly suggest a specific category.
- The Root Node: The classification process begins at the top of the tree, which is called the root nodes. The algorithm likely first asks if the petal length is greater than a certain threshold, such as 2.45cm.
- Classification Path: Since your petal length (5.4cm) is high, the decision path moves away from the Setosa species. It then evaluates the petal width to distinguish between the remaining two species.
- Species Prediction: Given that the petal width is 2.3cm and the length is 5.4cm, the tree will classify this flower as Iris Virginica. This is will be species typically exhibits the largest petal dimensions among the three types.
- Feature Importance Ranking: By aggregating how much each feature reduces impurity across the entire tree, the model generates a "Feature Importance" score. Even though you will provided all four measurements, the sepal width and length scores will rank near zero, while petal length and width share nearly 100% of the credit for identifying your Iris Virginica.
- Resilience to Outliers: If your flower's sepal length of 6.2cm had been an extreme, freakish typo like 62.0cm, a linear regression model would completely break. However, because decision trees rely on threshold inequalities (is it greater than or less than $X$?), the exact magnitude of an outlier doesn't distort the rest of the tree's structure.
- Decision Logic: The tree reaches this conclusion because Iris Versicolor usually has smaller petals than your specimen. Your flower dimensions fall well within the established boundaries for a Virginica classification trees.
- Model Visualization: You can visualize this entire logic as a flowchart where each box represents a question about a measurement. The finals "leaves" like at the bottom of the chart provide the predicted species name.
- The algorithm identifies that petal dimensions are mathematically superior to sepal dimensions for separating these specific classes. While a your sepal length of 6.2cm is recorded, the tree which is to prioritizes the 5.4cm petal length because it provides the highest "Information Gain" during the initial split.
- Information Gain via Entropy: While Gini Impurity is one way to measure node split quality, decision trees can also use an alternative metric called Entropy (a concept borrowed from thermodynamics that measures randomness or disorder). The tree calculates the Entropy drop—known as Information Gain—to determine that testing your 5.4cm petal length first yields the most clarity.
- At each node, the tree calculates a Gini Impurity score to measure how often a randomly chosen element would be incorrectly labeled. By funneling your flower into the Virginica path and also based on its 2.3cm petal width, the model successfully reduces this impurity to nearly zero case.
- The classification follows a "greedy" approach, meaning it makes the best possible binary choice at each junction without looking back. Your like flower’s journey is a series of "True/False" answers that like narrow down the identity from the entire population to a single species.
- The Setosa Shortcut: If your flower’s petal length had been less than or equal to 2.45cm, the decision tree would have stopped immediately at the very first split. The Setosa species is so distinct in its tiny petal dimensions that it creates a "pure leaf node" right at the start, requiring no further questions about widths or sepals.
- The Overfitting Trap: Because decision trees are highly adaptive, a tree trained too deeply on the Iris dataset might create hyper-specific rules (like "petal width greater than 1.75cm but sepal length exactly 6.2cm") just to catch a single outlier. To prevent this from ruining predictions on flowers like yours, algorithms use "pruning" to chop off overly complex branches.
- The Axis-Aligned Boundary: Decision trees create boundary lines that are always perpendicular to the feature axes. For your flower, this means the model draws a straight horizontal line on a graph at a specific petal width (e.g., 1.75cm) and a straight vertical line at a petal length, cutting the data into neat rectangular prediction zones rather than diagonal slopes.

![image](https://github.com/diantyapitaloka/Sklearn-Decisiontree/assets/147487436/fee66213-a688-4ff5-b651-047afca66c22)

## 🍬🥧🍰 Load Dataset 🍰🥧🍬
```
import pandas as pd
from sklearn.datasets import load_iris
iris = pd.read_csv('Iris.csv')
```

![image](https://github.com/diantyapitaloka/Decision-Tree/assets/147487436/028b4627-51d7-4405-9be8-580969b86f66)

## 🍬🥧🍰 Seeing Dataset Information 🍰🥧🍬
```
iris.info()
```

![image](https://github.com/diantyapitaloka/Decision-Tree/assets/147487436/388e75a9-496c-48fd-bcac-add435523cd4)

## 🍬🥧🍰 Cleansing Dataset 🍰🥧🍬
Delete useless column
```
iris.drop('Id',axis=1,inplace=True)
```

## 🍬🥧🍰 Attributes and Labels 🍰🥧🍬
Seperated attributes and labels
```
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]]
y = iris['Species']
```

## 🍬🥧🍰 Divide Dataset 🍰🥧🍬
Divide dataset into testing data and trial data
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

from sklearn.tree import DecisionTreeClassifier
````
## 🍬🥧🍰 Model Decision Tree 🍰🥧🍬
Made a model of Decision Tree
```
tree_model = DecisionTreeClassifier()
```

## 🍬🥧🍰 Testing Model 🍰🥧🍬
Testing data with model
```
tree_model = tree_model.fit(X_train, y_train)
```

## 🍬🥧🍰 Evaluated Model 🍰🥧🍬
Evaluated the model
```
from sklearn.metrics import accuracy_score

y_pred = tree_model.predict(X_test)

acc_secore = round(accuracy_score(y_pred, y_test), 3)

print('Accuracy: ', acc_secore)
```

![image](https://github.com/diantyapitaloka/Decision-Tree/assets/147487436/5476af1a-ee80-454e-b408-fe2cc999780c)

## 🍬🥧🍰 Prediction Model 🍰🥧🍬
Model prediction with tree_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])

```
print(tree_model.predict([[6.2, 3.4, 5.4, 2.3]])[0])

from sklearn.tree import export_graphviz
export_graphviz(
    tree_model,
    out_file = "iris_tree.dot",
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica' ],
    rounded= True,
    filled =True
)
```

## 🍬🥧🍰 License 🍰🥧🍬
- Copyright by Diantya Pitaloka

