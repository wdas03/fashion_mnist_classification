import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

X_train = np.load('hw2_fashionmnist/train.npy')
X_train = X_train.reshape((len(X_train), -1))

X_test = np.load('hw2_fashionmnist/test.npy')
X_test = X_test.reshape((len(X_test), -1))

y_train = np.load('hw2_fashionmnist/trainlabels.npy')
y_test = np.load('hw2_fashionmnist/testlabels.npy')

def show_img_from_class(images, labels, cls):
    class_imgs = images[labels == cls]
    random.shuffle(class_imgs)

    for img in class_imgs[:5]:
        plt.figure()
        plt.imshow(img.reshape((28, 28)), cmap='gray')
        plt.show()

show_img_from_class(X_train, y_train, 0)
show_img_from_class(X_train, y_train, 1)

max_leaves = []
for i in range(10, 10000, 500):
    dt_clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=i)
    dt_clf.fit(X_train, y_train)

    y_pred_train = dt_clf.predict(X_train)
    y_pred_test = dt_clf.predict(X_test)

    test_loss = np.count_nonzero(y_pred_test != y_test)
    train_loss = np.count_nonzero(y_pred_train != y_train)

    #dec_tree_accuracy = accuracy_score(y_test, y_pred)
    print("0-1 Test Loss: {}, 0-1 Train Loss: {}, max_leaf_nodes: {}, max_leaf_node_used: {}".format(test_loss, train_loss, i, dt_clf.get_n_leaves()))
    max_leaves.append((test_loss, train_loss, i, dt_clf.get_n_leaves()))

max_leaves_np = np.array(max_leaves)
plt.plot(max_leaves_np[:, 2], max_leaves_np[:, 0] / len(y_test), label="Test Loss")
plt.plot(max_leaves_np[:, 2], max_leaves_np[:, 1] / len(y_train), label="Train Loss")

plt.legend()
plt.xlabel("Max Number of Leaves (Log Scale)")
plt.ylabel("0-1 Loss")
plt.xscale("log")

plt.title("Max Leaves vs 0-1 Loss for Decision Tree Classifier")
plt.show()

# Constant estimators
max_leaves_rf_1 = []
for i in range(10, 5500, 500):
    rf_clf = RandomForestClassifier(random_state=0, max_leaf_nodes=i)
    rf_clf.fit(X_train, y_train)

    y_pred_train = rf_clf.predict(X_train)
    y_pred_test = rf_clf.predict(X_test)

    test_loss = np.count_nonzero(y_pred_test != y_test)
    train_loss = np.count_nonzero(y_pred_train != y_train)

    # Calculate the total number of parameters in the random forest classifier
    total_params = i * len(rf_clf.estimators_)

    print("Test Loss: {}, Train Loss: {}, max_leaf_nodes: {}, total_params: {}".format(test_loss, train_loss, i, total_params))
    max_leaves_rf_1.append((test_loss, train_loss, i, total_params))

max_leaves_rf_1 = np.array(max_leaves_rf_1)
plt.plot(max_leaves_rf_1[:, 2], max_leaves_rf_1[:, 0] / len(y_test), label="Test Loss")
plt.plot(max_leaves_rf_1[:, 2], max_leaves_rf_1[:, 1] / len(y_train), label="Train Loss")

plt.legend()
plt.xlabel("Total # of Params (Log Scale)")
plt.ylabel("0-1 Loss")
plt.xscale("log")

plt.title("Total Params vs 0-1 Loss for Random Forest Classifier")
plt.show()

# Constant leaf nodes
max_leaves_rf = []
for i in range(1, 700, 50):
    rf_clf = RandomForestClassifier(random_state=0, n_estimators=i, max_leaf_nodes=200)
    rf_clf.fit(X_train, y_train)

    y_pred_train = rf_clf.predict(X_train)
    y_pred_test = rf_clf.predict(X_test)

    test_loss = np.count_nonzero(y_pred_test != y_test)
    train_loss = np.count_nonzero(y_pred_train != y_train)

    # Calculate the total number of parameters in the random forest classifier
    total_params = 200 * len(rf_clf.estimators_)

    print("Test Loss: {}, Train Loss: {}, n_estimators: {}, total_params: {}".format(test_loss, train_loss, i, total_params))
    max_leaves_rf.append((test_loss, train_loss, i, total_params))

def double_descent():
    double_descent = []

    phase_1 = []
    num_leaves = 10
    while num_leaves <= 5100:
        rf_clf = RandomForestClassifier(random_state=0, n_estimators=1, max_leaf_nodes=num_leaves)
        rf_clf.fit(X_train, y_train)

        y_pred_train = rf_clf.predict(X_train)
        y_pred_test = rf_clf.predict(X_test)

        test_loss = np.count_nonzero(y_pred_test != y_test)
        train_loss = np.count_nonzero(y_pred_train != y_train)

        # Calculate the total number of parameters in the random forest classifier
        total_params = num_leaves * len(rf_clf.estimators_)

        print("Test Loss: {}, Train Loss: {}, max_leaf_nodes: {}, total_params: {}".format(test_loss, train_loss, num_leaves, total_params))
        phase_1.append((test_loss, train_loss, num_leaves, total_params))

        num_leaves += 500

    phase_2 = []
    n_estimators = 2
    while n_estimators <= 128: 
        rf_clf = RandomForestClassifier(random_state=0, n_estimators=n_estimators, max_leaf_nodes=num_leaves)
        rf_clf.fit(X_train, y_train)

        y_pred_train = rf_clf.predict(X_train)
        y_pred_test = rf_clf.predict(X_test)

        test_loss = np.count_nonzero(y_pred_test != y_test)
        train_loss = np.count_nonzero(y_pred_train != y_train)

        # Calculate the total number of parameters in the random forest classifier
        total_params = num_leaves * len(rf_clf.estimators_)

        print("Test Loss: {}, Train Loss: {}, max_leaf_nodes: {}, n_estimators: {}, total_params: {}".format(test_loss, train_loss, num_leaves, n_estimators, total_params))
        phase_2.append((test_loss, train_loss, n_estimators, total_params))

        n_estimators *= 2
    
    return phase_1, phase_2

phase_1, phase_2 = double_descent()

phase_1 = np.array(phase_1)
phase_2 = np.array(phase_2)

plt.plot(phase_1[:, 3], phase_1[:, 0] / len(y_test), label="Test Loss (Phase 1)")
plt.plot(phase_1[:, 3], phase_1[:, 1] / len(y_train), label="Train Loss (Phase 1)")

plt.plot(phase_2[:, 3], phase_2[:, 0] / len(y_test), label="Test Loss (Phase 2)")
plt.plot(phase_2[:, 3], phase_2[:, 1] / len(y_train), label="Train Loss (Phase 2)")

plt.legend()
plt.xlabel("Total # of Params (Log Scale)")
plt.ylabel("0-1 Loss")
plt.xscale("log")

plt.title("Total Params vs 0-1 Loss for Random Forest Classifier (Double Descent)")
plt.show()