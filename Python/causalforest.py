# ---- Logistic regression CV ---- #
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
#----- developmenet causal trees ------#
# this method based on R-learner which minimize 
# 1/N Σ{(Y-m(X)) - (T-pi(X))}
# task : use numpy array.

import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ElasticNet正則化ロジスティック回帰モデルのクロスフィッティングを実行する関数
def cross_fit_logistic_regression(X, y, n_splits=5, l1_ratio=0.5):
    model = LogisticRegressionCV(
        Cs=20,  # 10個の候補パラメータを探索
        cv=4,  # 内部クロスバリデーション
        penalty='elasticnet',
        solver='saga',
        max_iter=1000,
        scoring='accuracy',
        random_state=42,
        l1_ratios=[l1_ratio]  # ElasticNetのl1比率
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', model)
    ])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    probabilities = np.zeros((len(X), 2))  # 各サンプルのクラスごとの予測確率を保存

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        pipeline.fit(X_train, y_train)
        probabilities[test_index] = pipeline.predict_proba(X_test)

    return probabilities

# ElasticNetCVを使用した線形回帰モデルのクロスフィッティングを実行する関数
# see help: https://scikit-learn.org/stable/modules/linear_model.html#elastic-net
def cross_fit_elasticnet_regression(X, y, n_splits=5, l1_ratio=0.5, alphas=np.logspace(-6, 6, 13)):
    model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio, cv=4, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', model)
    ])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = np.zeros(len(X))

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        pipeline.fit(X_train, y_train)
        predictions[test_index] = pipeline.predict(X_test)

    return predictions

# Generate sample datasets
def generate_data(n_samples):
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))
    data = []
    labels = []
    treatments = []
    for i in range(n_samples):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        trt = random.randint(0, 1)
        y = x1 + x2 + trt * x1 + random.uniform(-1, 1)  # Simple linear relationship with noise
        data.append([x1, x2])
        labels.append(y)
        treatments.append(trt)
    return np.array(data), np.array(labels), np.array(treatments), list(range(n_samples))

# Split the data into two halves
def split_data(data, labels, trt, propensity_scores, outcome_regression, indices):
    combined = list(zip(data, [x - y for (x, y) in zip(labels, outcome_regression)], [x - y for (x, y) in zip(trt, propensity_scores)], indices))
    random.shuffle(combined)
    split_index = len(combined) // 2
    data1, labels1, trt1, indices1 = zip(*combined[:split_index])
    data2, labels2, trt2, indices2 = zip(*combined[split_index:])
    return list(data1), list(labels1), list(trt1), list(indices1), list(data2), list(labels2), list(trt2), list(indices2)

# Define the Node class
class Node:
    def __init__(self, mse, num_samples, predicted_value, data_indices):
        self.mse = mse
        self.num_samples = num_samples
        self.predicted_value = predicted_value
        self.data_indices = data_indices
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

    @staticmethod
    def mse(labels):
        if len(labels) == 0:
            return 0
        mean = sum(labels) / len(labels)
        return sum((x - mean) ** 2 for x in labels) / len(labels)

    @staticmethod
    def causal_mse(labels,trts): #this loss is calculated based on centered outcome and treatments.
        if len(labels) == 0:
            return float('inf')
        predicted_value = sum([x/y for (x, y) in zip(labels, trts)]) / len(labels)
        return sum([x-y*predicted_value for (x, y) in zip(labels, trts)])**2 / len(labels)

    @staticmethod
    def best_split(data, labels, trts, pred_data, alpha):
        m, n = len(data), len(data[0])
        if m <= 1:
            return None, None
        
        best_mse = float('inf')
        best_index, best_threshold = None, None
        
        for index in range(n):
            thresholds, sorted_labels, sorted_trts = zip(*sorted(zip([row[index] for row in data], labels, trts)))
            pred_values = [row[index] for row in pred_data]
            
            for i in range(1, m):
                left_pred_indices = [j for j in range(len(pred_values)) if pred_values[j] < thresholds[i]]
                right_pred_indices = [j for j in range(len(pred_values)) if pred_values[j] >= thresholds[i]]
                
                if len(left_pred_indices) < max([1,len(pred_data) * alpha]) or len(right_pred_indices) < max([1,len(pred_data) * alpha]):
                    continue
                
                left_labels = sorted_labels[:i]
                left_trts = sorted_trts[:i]
                right_labels = sorted_labels[i:]
                right_trts = sorted_trts[i:]
                mse_left = Node.causal_mse(left_labels, left_trts)
                mse_right = Node.causal_mse(right_labels, right_trts)
                mse_total = mse_left/i + mse_right/(m-i)   #(i * mse_left + (m - i) * mse_right) / m
                
                if thresholds[i] == thresholds[i - 1]:
                    continue
                
                if mse_total < best_mse:
                    best_mse = mse_total
                    best_index = index
                    best_threshold = thresholds[i]
        return best_index, best_threshold

# Build the tree
def build_tree(data, labels, treatments, propensity_scores, outcome_regressions, indices, k, alpha):
    def build(data_split, labels_split, trts_split, pred_data, pred_labels, pred_trts, data_indices, depth):
        
        predicted_value = sum([x/y for (x, y) in zip(pred_labels, pred_trts)]) / len(pred_labels) # mean of Y-m(x)/T-pi(x)
        node = Node(
            mse=Node.causal_mse(pred_labels, pred_trts),
            num_samples=len(pred_labels),
            predicted_value=predicted_value,
            data_indices=data_indices,
        )

        if len(set(labels_split)) == 1 or len(pred_labels) <= k:
            return node
        
        index, threshold = Node.best_split(data_split, labels_split, trts_split, pred_data, alpha)
        if index is None:
            return node
        
        left_indices = [i for i in range(len(data_split)) if data_split[i][index] < threshold]
        right_indices = [i for i in range(len(data_split)) if data_split[i][index] >= threshold]
        
        # if len(left_indices) == 0 or len(right_indices) == 0:
        #    return node

        left_data_split = [data_split[i] for i in left_indices]
        right_data_split = [data_split[i] for i in right_indices]
        left_labels_split = [labels_split[i] for i in left_indices]
        right_labels_split = [labels_split[i] for i in right_indices]
        left_trts_split = [trts_split[i] for i in left_indices]
        right_trts_split = [trts_split[i] for i in right_indices]
        
        left_pred_indices = [i for i in range(len(pred_data)) if pred_data[i][index] < threshold]
        right_pred_indices = [i for i in range(len(pred_data)) if pred_data[i][index] >= threshold]

        #if len(left_pred_indices) == 0 or len(right_pred_indices) == 0:
        #    return node

        left_pred_data = [pred_data[i] for i in left_pred_indices]
        right_pred_data = [pred_data[i] for i in right_pred_indices]
        left_pred_labels = [pred_labels[i] for i in left_pred_indices]
        right_pred_labels = [pred_labels[i] for i in right_pred_indices]
        left_pred_trts = [pred_trts[i] for i in left_pred_indices]
        right_pred_trts = [pred_trts[i] for i in right_pred_indices]
        
        node.feature_index = index
        node.threshold = threshold
        node.left = build(left_data_split, left_labels_split, left_trts_split, left_pred_data, left_pred_labels, left_pred_trts, [data_indices[i] for i in left_pred_indices], depth + 1)
        node.right = build(right_data_split, right_labels_split, right_trts_split, right_pred_data, right_pred_labels, right_pred_trts, [data_indices[i] for i in right_pred_indices], depth + 1)
        return node
    data_split1, labels_split1, trt_split1, indices_split1, data_split2, labels_split2, trt_split2, indices_split2 = split_data(data, labels, treatments, propensity_scores, outcome_regressions, indices)
    return build(data_split1, labels_split1, trt_split1, data_split2, labels_split2, trt_split2, indices_split2, 0)

# Predict using the tree and return the terminal node's sample indices
def predict(node, row):
    if node.left is None and node.right is None:
        return node.predicted_value, node.data_indices
    
    if row[node.feature_index] < node.threshold:
        return predict(node.left, row)
    else:
        return predict(node.right, row)

#Visualize the tree (Tree-structure)
def plot_tree(node, depth=0, ax=None, pos=None, parent_pos=None, text=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
    if node is None:
        return
    
    if pos is None:
        pos = (0.5, 1)
    
    ax.text(pos[0], pos[1], f'{node.predicted_value:.2f}\nSamples: {node.num_samples}', ha='center', va='top', bbox=dict(facecolor='white', edgecolor='black'))
    
    if parent_pos is not None:
        ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], 'k-')
    
    if node.left is not None or node.right is not None:
        if node.left is not None:
            left_pos = (pos[0] - 0.5/(depth+2), pos[1] - 0.1)
            plot_tree(node.left, depth+1, ax, left_pos, pos, text='Left')
        if node.right is not None:
            right_pos = (pos[0] + 0.5/(depth+2), pos[1] - 0.1)
            plot_tree(node.right, depth+1, ax, right_pos, pos, text='Right')

    if depth == 0:
        plt.show()

# Visualize the tree (Rectangle in 2-dimension)
def plot_tree2(node, depth=0, ax=None, bounds=(0, 10, 0, 10), parent_coords=None, parent_feature=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        
    x_min, x_max, y_min, y_max = bounds

    if node.left is None and node.right is None:
        # Plot terminal node
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=True, edgecolor='black', facecolor='lightgray'))
        ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, f'{node.predicted_value:.2f}', horizontalalignment='center', verticalalignment='center')
    else:
        if node.feature_index == 0:
            # Vertical split
            split_value = node.threshold
            ax.plot([split_value, split_value], [y_min, y_max], 'k-')
            left_bounds = (x_min, split_value, y_min, y_max)
            right_bounds = (split_value, x_max, y_min, y_max)
        else:
            # Horizontal split
            split_value = node.threshold
            ax.plot([x_min, x_max], [split_value, split_value], 'k-')
            left_bounds = (x_min, x_max, y_min, split_value)
            right_bounds = (x_min, x_max, split_value, y_max)
        
        plot_tree2(node.left, depth + 1, ax, left_bounds, (split_value, y_max), node.feature_index)
        plot_tree2(node.right, depth + 1, ax, right_bounds, (split_value, y_min), node.feature_index)
    
    if depth == 0:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Decision Tree Visualization')
        plt.show()

#--- causal forest ---#
class CausalForest:
    def __init__(self, n_trees, max_samples, k, alpha):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.k = k
        self.alpha = alpha
        self.trees = []

    def fit(self, data, labels, treatments, propensity_scores, outcome_regressions, indices):
        for _ in range(self.n_trees):
            sample_indices = random.sample(indices, self.max_samples)
            sample_data = [data[i] for i in sample_indices]
            sample_labels = [labels[i] for i in sample_indices]
            sample_trts = [treatments[i] for i in sample_indices]
            sample_pss = [propensity_scores[i] for i in sample_indices]
            sample_ors = [outcome_regressions[i] for i in sample_indices]
            tree = build_tree(sample_data, sample_labels, sample_trts, sample_pss, sample_ors, sample_indices, self.k, self.alpha)
            self.trees.append(tree)

    def predict(self, row):
        predictions = [predict(tree, row) for tree in self.trees]
        return sum(p[0] for p in predictions) / len(predictions)
    
    def get_usage_matrix(self, row, data):
        usage_matrix = np.zeros((len(data), self.n_trees))
        predictions = [predict(tree, row) for tree in self.trees]
        for tree_index, (_, indices) in enumerate(predictions):
            for index in indices:
                if index <= len(data):
                    usage_matrix[index, tree_index] = 1
        return usage_matrix
    
def normalize_and_sum(data):
    """
    行列の列方向の和を1に正規化し、各行の和を計算する関数

    Parameters:
    data (list of list of float): 入力の行列データ

    Returns:
    tuple: 正規化された配列と各行の和
    """
    # NumPy配列に変換
    array = np.array(data)

    # 列方向の和を1に正規化
    column_sums = array.sum(axis=0)
    normalized_array = array / column_sums

    # 各行の和を計算
    row_sums = normalized_array.sum(axis=1)

    return normalized_array, row_sums

def weighted_least_squares(X, T, Y, w):
    """
    重み付き最小二乗法による回帰係数の推定

    Parameters:
    X (np.ndarray): p次元の変数 (n_samples, p)
    T (np.ndarray): 1次元の処置変数 (n_samples, 1)
    Y (np.ndarray): 結果変数 (n_samples, 1)
    w (np.ndarray): サンプルの重み (n_samples,)

    Returns:
    float: 処置変数Tの回帰係数
    """
    # デザイン行列の構築
    X_design = np.hstack([np.ones((X.shape[0], 1)), X, T.reshape(-1, 1)])  # (n_samples, p + 2)
    
    # 重み行列の作成
    W = np.diag(w)
    
    # 重み付き最小二乗法の計算
    XtW = X_design.T @ W
    beta_hat = np.linalg.inv(XtW @ X_design) @ XtW @ Y
    
    # 処置変数Tの回帰係数を抽出
    treatment_coef = beta_hat[-1]
    
    return treatment_coef

# Generate sample data
data, labels, treatments, indices  = generate_data(1000)
k = 20  # Maximum number of samples in each node for prediction
alpha = 0.2  # Minimum proportion of samples in each split

# cross-fitting logistic regression 
predictions_prob = cross_fit_logistic_regression(data, treatments)
propensity_scores = [p[1] for p in predictions_prob]

# クロスフィッティングの実行
predictions_reg = cross_fit_elasticnet_regression(data, labels)
outcome_regressions = predictions_reg

# Build and test the tree
tree = build_tree(data, labels, treatments, propensity_scores, outcome_regressions, indices, k, alpha)

# Predict on a new sample and get terminal node's sample indices
new_sample = [5, 5]
prediction, terminal_node_indices = predict(tree, new_sample)

# Calculate the mean of the selected labels
selected_labels = [labels[i] for i in terminal_node_indices]
mean_selected_labels = sum(selected_labels) / len(selected_labels)

# tree prediction results
print(prediction, terminal_node_indices)

# plot gradient trees
plot_tree(tree)
plot_tree2(tree)

# Build and test the random forest
rf = CausalForest(n_trees=20, max_samples=200, k=1, alpha=0.1)
rf.fit(data, labels, treatments, propensity_scores, outcome_regressions, indices)

# Test data
test_data = [[2, 5]]

# Get the prediction for the test data
predict_test = rf.predict(test_data[0])
predict_test

# Get the usage_matrix
predict_usage = rf.get_usage_matrix(test_data[0],data)
predict_usage

# get forest-kernel weights
normalized_array, row_sums = normalize_and_sum(predict_usage)

# 結果の表示
print("Normalized Array:")
print(normalized_array)
print("Row Sums:")
print(row_sums)

# 重み付き最小二乗法によるパラメータ推定
treatment_coef = weighted_least_squares(data, treatments, labels, row_sums)
print(treatment_coef)