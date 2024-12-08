
AI

Prac1
----------------------------------------- BFS -----------------------------------------
from queue import Queue
adj_list={
"M":["N","Q","R"],
"N":["M","O","Q"], 
"O":["3N","P"],
"P":["O","Q"],
"Q":["M","N","P"],
"R":["M"]
}
#initialization
visited= {} #keeping track of all the visited nodes
level={} #keeping track of level of each node
parent={}#keeping track of parent node of each node
bfs_traversal = [] #traversal list
queue = Queue()
for node in adj_list.keys():
    visited[node] = False
    parent[node] = None
    level[ node] = -1
print("Before traversal")
print("visited:",visited)
print("level:",level)
print("parent:",parent)

source = "M"
visited[source] = True
level[source] = 0 #SOURCE NONE IS m SO LEVEL WILL BE ZERO
queue.put(source) #ADD M TO THE QUEUE

while not queue.empty():
    u = queue.get() #GET THE FIRST ELEMENT FROM THE QUEUE
    bfs_traversal.append(u)
    for v in adj_list[u]:
        if not visited [v]:
            visited [v]= True
            parent[v]=u
            level[v]=level[u]+1
            queue.put(v)
print("After traversal")
print("BFS traversal:",bfs_traversal)
##Minimum Distance
print ("Minimum distance")
print("Level N",level["N"])
print("Level O",level["O"])
print("Parent M",parent["M"])
print("Parent P",parent["P"])
node = "O" #destination node
path= []
while node is not None:
    path.append(node)
    node = parent[node]
path.reverse()
print("Shortest path is:", path)



Prac 2 

------------------------------------ DFS ---------------------------------------------

from collections import deque

adj_list = {
    'A': ['C', 'D', 'B'],
    'C': ['A', 'K'],
    'D': ['A', 'K', 'L'],
    'K': ['C', 'D', 'L'],
    'L': ['K', 'D', 'J'],
    'J': ['M'],
    'B': ['A'],
    'M': ['J']
}

# Initialization
visited = {}
level = {}
parent = {}
bfs_traversal = []
stack = deque()

for node in adj_list.keys():
    visited[node] = False
    parent[node] = None
    level[node] = -1

source = input("Enter the exact Source node to reach: ")
visited[source] = True
level[source] = 0
stack.append(source)

while not (len(stack)==0):
    u = stack.pop()
    bfs_traversal.append(u)
    for v in adj_list[u]:
        if not visited[v]:
            visited[v] = True
            parent[v] = u
            level[v] = level[u] + 1
            stack.append(v)

print("BFS traversal:", bfs_traversal)

# Minimum path
target = input("Enter the exact node to reach: ") # Destination node
path = []

if visited[target]:
    node = target
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

print("Minimum path from Source to Destination:", ' -> '.join(path))



Prac 3 

------------------------------------------ DLS ----------------------------------------

adj_list = {
    'A': ['B', 'C', 'D'],
    'B': ['E'],
    'C': ['F', 'G'],
    'D': ['C'],
    'E': ['H', 'F'],
    'F': ['B'],
    'H': [],
    'G': [],
}
visited = {}
parent = {}
dls_traversal = []

# Initialize visited and parent dictionaries
for node in adj_list.keys():
    visited[node] = False
    parent[node] = None

def dls(node, depth):
    if depth < 0:
        return False
    visited[node] = True
    dls_traversal.append(node)
    if node == target:
        return True
    for neighbor in adj_list[node]:
        if not visited[neighbor]:
            parent[neighbor] = node
            if dls(neighbor, depth - 1):
                return True
    return False

source = input("Enter the source node: ")
if source not in adj_list:
    print("Source node not in graph.")
else:
    depth_limit = int(input("Enter the depth limit: "))
    target = input("Enter the destination node: ")

    if target not in adj_list:
        print("Destination node not in graph.")
    else:
        found = dls(source, depth_limit)

        if not found:
            print("No path exists from source to destination within the depth limit.")
        else:
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = parent[node]

            path.reverse()
            print("DLS Traversal: ", dls_traversal)
            print("Path is: ", path)


Prac 4

------------------------------------- IDDFS --------------------------------------------

graph = {
     "M" : ["N","Q","R"],
     "N" : ["O","Q","M"],
     "R" : ["M"],
     "O" : ["P","N"],
     "Q" : ["M","N"],
     "P" : ["O","Q"]
     }
def dls(graph,S,D,L):
        visited = {}
        parent = {}
        level = {}
        dls_list = []
        shortest_path = []
        stack = []
        for i in graph.keys():
            visited[i] = False
            parent[i] = None
            level[i] = -1
        stack.append(S)
        level[S] = 0
        while stack:
            u = stack.pop()
            visited[u] = True
            dls_list.append(u)
            for v in graph[u]:
                if not visited[v] and v not in stack and level[u] + 1<=L:
                    level[v] = level[u] + 1
                    parent[v] = u
                    stack.append(v)
            if D in dls_list:
                temp = D
                while parent[temp] != None:
                    shortest_path.append(temp)
                    temp = parent[temp]
                else:
                    print(f"{D} found in level {L}")
                    shortest_path.append(S)
                    shortest_path.reverse()
                    print(f"Shortest path is: {shortest_path}")
            else:
                print(f"{D} not found in level {L}")
            print(f"DFS till level {L}: {dls_list}\n\n")
S = input("Enter the source node: ")
D = input("Enter the destination node: ")
level = int(input("Enter the limit level: "))
for i in range(level):
    dls(graph,S,D,i)


Prac 5

----------------------------------------- Greedy BFS ------------------------------------

graph = {
    "Oradea": ({"Zerind": 71, "Sibiu": 151}, 380),
    "Zerind": ({"Oradea": 71, "Arad": 75}, 374),
    "Arad": ({"Zerind": 75, "Sibiu": 140 , "Timisoara": 118}, 399),
    "Timisoara": ({"Arad": 118, "Lugoj": 111}, 329),
    "Lugoj": ({"Timisoara": 111, "Mehadia": 70}, 244),
    "Mehadia": ({"Lugoj": 70, "Drobeta": 75}, 241),
    "Sibiu": ({"Oradea": 151, "Fagaras": 99, "Rimnicu Vilcea": 80, "Arad": 140}, 253),
    "Fagaras": ({"Sibiu": 99, "Bucharest": 211}, 176),
    "Rimnicu Vilcea": ({"Sibiu": 80, "Pitesti": 97, "Craiova": 146}, 193),
    "Bucharest": ({"Fagaras": 211, "Pitesti": 101, "Urziceni": 85, "Giurgiu": 90}, 0),
    "Pitesti": ({"Rimnicu Vilcea": 97, "Craiova": 138, "Bucharest": 101}, 100),
    "Craiova": ({"Rimnicu Vilcea": 146, "Pitesti": 138, "Drobeta": 120}, 160),
    "Drobeta": ({"Mehadia": 75, "Craiova": 120}, 242),
    "Urziceni": ({"Bucharest": 85, "Vaslui": 142, "Hirsova": 98}, 80),
    "Giurgiu": ({"Bucharest": 90}, 77),
    "Vaslui": ({"Iasi": 92, "Urziceni": 142}, 199),
    "Hirsova": ({"Urziceni": 98, "Eforie": 86}, 151),
    "Iasi": ({"Vaslui": 92, "Neamt": 87}, 226),
    "Eforie": ({"Hirsova": 86}, 161),
    "Neamt": ({"Iasi": 87}, 234)
    }

def greedy_search_rec(graph, prev, dst, path, q):
    # n: (h(n))
    print("Connected nodes of current node",prev,"with h(n) values:")
    for n in graph[prev][0]:
        if n not in path:
            q[n] = graph[n][1]
            print(n,"->",q[n])
    while q:
        mn = min(q,key =q.get)
        print("Taking minimum h(n) vertex:",mn)
        #print(mn)
        if dst == mn:
            return path + [dst]
        #del q[mn]
        new_path = greedy_search_rec(graph, mn, dst, path + [mn], q)
        if new_path:
            return new_path
        return[]
source=input("Enter source vertex : ")
dest=input("Enter destination vertex : ")
path=greedy_search_rec(graph, source, dest,[source],{})
if path:
    print(path)
else:
    print("Path not found!!")





Prac 6

------------------------------------------ a_star -------------------------------------

graph = {
    "Oradea": ({"Zerind": 71, "Sibiu": 151}, 380),
    "Zerind": ({"Oradea": 71, "Arad": 75}, 374),
    "Arad": ({"Zerind": 75, "Sibiu": 140 , "Timisoara": 118}, 399),
    "Timisoara": ({"Arad": 118, "Lugoj": 111}, 329),
    "Lugoj": ({"Timisoara": 111, "Mehadia": 70}, 244),
    "Mehadia": ({"Lugoj": 70, "Drobeta": 75}, 241),
    "Sibiu": ({"Oradea": 151, "Fagaras": 99, "Rimnicu Vilcea": 80, "Arad": 140}, 253),
    "Fagaras": ({"Sibiu": 99, "Bucharest": 211}, 176),
    "Rimnicu Vilcea": ({"Sibiu": 80, "Pitesti": 97, "Craiova": 146}, 193),
    "Bucharest": ({"Fagaras": 211, "Pitesti": 101, "Urziceni": 85, "Giurgiu": 90}, 0),
    "Pitesti": ({"Rimnicu Vilcea": 97, "Craiova": 138, "Bucharest": 101}, 100),
    "Craiova": ({"Rimnicu Vilcea": 146, "Pitesti": 138, "Drobeta": 120}, 160),
    "Drobeta": ({"Mehadia": 75, "Craiova": 120}, 242),
    "Urziceni": ({"Bucharest": 85, "Vaslui": 142, "Hirsova": 98}, 80),
    "Giurgiu": ({"Bucharest": 90}, 77),
    "Vaslui": ({"Iasi": 92, "Urziceni": 142}, 199),
    "Hirsova": ({"Urziceni": 98, "Eforie": 86}, 151),
    "Iasi": ({"Vaslui": 92, "Neamt": 87}, 226),
    "Eforie": ({"Hirsova": 86}, 161),
    "Neamt": ({"Iasi": 87}, 234)
    }

def get_min(q):
    mn = (0, (0, float("INF")))
    for i in q:
        if sum(q[i]) < sum(mn[1]):
            mn = (i, q[i])
    return mn[0]
    
def a_star(graph, prev, dst, path, pcost, q):
    print("Connected nodes of current node", prev, "with h(n) values: ")    
    for n in graph[prev][0]:    #neighbors list n =Z, S, T
        if n not in path:
            q[n] = (graph[n][1], graph[prev][0][n])
            print(n, "->", q[n])
            add1=sum(q[n])
            path_cost=pcost+add1        
            print("A* value for ",n, "is: ",path_cost)
    while q:
        mn = get_min(q)
        print("Selecting Minimun vertex: ", mn)
        print("__________________________________________________")
        if dst == mn:
            return path + [dst]
        pc = pcost + q[mn][1]
        print("Previous path cost:",pc)
        #del q[mn]
        new_path = a_star(graph, mn, dst, path + [mn], pc, q)
        if new_path:
            return new_path
    return []
source=input("Enter Source vertex: ")
dest=input("Enter destination vertex: ")
heuristic=int(input("Enter given heuristic value for source: "))
path = a_star(graph, source, dest, [], 0, {source: (heuristic, 0)})
if path:
    print(path)
else:
    print("Path not found")



Prac 7

------------------------------------------ NAVIE BAYES LEARNING ALGO-----------------------------------------------------------

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sn
import matplotlib.pyplot as plt

iris = load_iris()
print(iris.feature_names)



X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
GNB=GaussianNB()
GNB.fit(X_train,y_train)
y_pred=GNB.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
confusion_matrix_result=confusion_matrix(y_test,y_pred)
print("Accuracy of the model",accuracy)
print("Confusion matrix of the model:")
print(confusion_matrix_result)
plt.figure(figsize=(7,5))
sn.heatmap(confusion_matrix_result,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


Prac 8
------------------------------------------: Implementing ensemble Learning Algorithm and Support Vector Machine(SVM) Algorithm-----------


1.  Ensemble Learning
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
warnings.simplefilter("ignore")
iris = datasets.load_iris()
x,y = iris.data,iris.target
labels = ["Random Forest","Logistic Regression","GaussianNB","Decision Tree"]
m1 = RandomForestClassifier(random_state=42)
m2 = LogisticRegression(random_state=1)
m3 = GaussianNB()
m4 = DecisionTreeClassifier()
for m,label in zip([m1,m2,m3,m4],labels):
    scores = model_selection.cross_val_score(m,x,y,cv=5,scoring="accuracy")
    print(f"Accuracy: {scores.mean()} {label}")
voting_clf_hard = VotingClassifier(estimators=[(labels[0],m1),(labels[1],m2),(labels[2],m3),(labels[3],m4)],voting='hard')
voting_clf_soft = VotingClassifier(estimators=[(labels[0],m1),(labels[1],m2),(labels[2],m3),(labels[3],m4)],voting='soft')
scores1 = model_selection.cross_val_score(voting_clf_hard,x,y,cv=5,scoring="accuracy")
scores2 = model_selection.cross_val_score(voting_clf_soft,x,y,cv=5,scoring="accuracy")
print(f"Accuracy of the hard voting: {scores1.mean()}")
print(f"Accuracy of the soft voting: {scores2.mean()}")



2. Support Vector Machine
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only the first two features (sepal length and sepal width)
y = iris.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the SVM model with a linear kernel
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length (standardized)')
    plt.ylabel('Sepal width (standardized)')
    plt.title('SVM Decision Boundaries')
    plt.show()
# Plot decision boundaries using the training set
plot_decision_boundaries(X_train, y_train, svm_model)



Prac 9

------------------------------------- : Implement Decision Tree Classifier--------------------------------------------------------


import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn .model_selection import train_test_split 
from sklearn import metrics 
from matplotlib import pyplot as plt
from sklearn import tree

col_names = ['Reservation', 'Raining', 'BadService','Saturday','Result']
hoteldata = pd.read_csv("dtree.csv", header=None, names=col_names)
feature_cols = ['Reservation', 'Raining', 'BadService','Saturday']
X = hoteldata[feature_cols] 
y = hoteldata.Result 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=1)
print(hoteldata)
print("x train data: ", X_train)
print("y train data: ",y_train)
print("x test data: ", X_test)
print("y test data: ",y_test)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("ytest = ", X_test)
print("ypred = ", y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
fig = plt.figure(figsize=(25,20))
t = tree.plot_tree(clf,feature_names=feature_cols,class_names=['Leave','Wait'],filled=True)
fig.savefig("decistion_tree.png")


Prac 10 
----------------------------------------------------: Implement K Nearest Neighbour -----------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
dataset = load_iris()
x , y = dataset.data,dataset.target
knn = KNeighborsClassifier(n_neighbors = 5)
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size = 0.3,random_state = 42)
print(f"XTrain: {x_train}")
print(f"XTest: {x_test}")
print(f"yTrain: {y_train}")
print(f"y test:{y_test}")
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print(f"y_pred:{y_pred}")      
accuracy = metrics.accuracy_score(y_test,y_pred)
print(f"Accuracy :{round(accuracy*100,2)}%")
cm = metrics.confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
print(cm)

