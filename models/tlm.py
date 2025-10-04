import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_curve
from graphviz import Digraph
from data_utils import  age_based_mixup

class Node:
    def __init__(self, node_id, parent_mae=None, parent=None, is_leaf=False):
        self.regressor = LinearRegression()
        self.classifier = None
        self.threshold = None
        self.is_trained = False
        self.left = None
        self.right = None
        self.is_leaf = is_leaf
        self.node_id = node_id
        self.error = None
        self.train_mae = None
        self.train_rmse = None
        self.depth = 0
        self.test_mae = None
        self.test_rmse = None
        self.eer_train = 0
        self.acc_train = 0
        self.eer_test = 0
        self.acc_test = 0
        self.num_train_samples = 0
        self.num_test_samples = 0
        self.mean = 0
        self.parent = parent 
        self.parent_mae = parent_mae
        self.mae_change = None


class TLM:
    def __init__(self, X, y, max_depth=4, split="deterministic", current_depth=0, 
                 is_leaf=False, node_id=0, parent=None, parent_mae=None):
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.node = Node(node_id, parent_mae=parent_mae, parent=parent, is_leaf=is_leaf)
        self.X = X
        self.y = y
        self.total_error = 0 
        self.split = split

    def train_node(self, X_train, y_train, X_test, y_test, thresholds, use_oracle=False):
        print(f"\nTraining node {self.node.node_id}")

        self.node.num_train_samples = len(y_train)
        self.node.num_test_samples = len(y_test)

        # Train the linear regression model
        self.node.regressor.fit(*age_based_mixup(X_train, y_train))
        self.node.mean = np.mean(y_train)

        # Evaluate model on train data
        train_predictions = self.node.regressor.predict(X_train)
        self.node.train_mae = mean_absolute_error(y_train, train_predictions)
        self.node.train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))

        parent_tse = self.total_squared_error(y_train, train_predictions)

        # Evaluate model on test data
        test_predictions = self.node.regressor.predict(X_test)
        self.node.test_mae = mean_absolute_error(y_test, test_predictions)
        self.node.test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

        self.node.depth = self.current_depth

        # Check stopping criteria
        if (self.current_depth >= self.max_depth or 
            len(np.unique(y_train)) < 2 or 
            len(y_train) < 200):
            self.node.is_leaf = True
            return

        best_threshold = None
        best_reduction = float('-inf')

        for threshold in thresholds:
            cls_y_train = (self.y > threshold).astype(int)
            cls_y_test = (y_test > threshold).astype(int)

            if len(np.unique(cls_y_train)) < 2 or len(np.unique(cls_y_test)) < 2:
                continue
                
            # Train classifier
            if not use_oracle:
                sample_weights = np.ones(len(self.y))
                current_node_mask = np.zeros(len(self.y), dtype=bool)
                for row in X_train:
                    current_node_mask |= np.all(self.X == row, axis=1)

                sample_weights[~current_node_mask] = 0.5

                classifier = LogisticRegression(solver='liblinear', max_iter=1000)
                classifier.fit(self.X, cls_y_train, sample_weight=sample_weights)
            else:
                # Oracle: use threshold directly, no classifier needed
                classifier = None
            
            # Split the data for regression models
            if use_oracle:
                # Oracle: use thresholds directly on true labels
                left_indices_train = y_train <= threshold
                right_indices_train = ~left_indices_train
                left_indices_test = y_test <= threshold
                right_indices_test = ~left_indices_test
            else:
                # Use classifier predictions
                if self.split == "deterministic":
                    y_pred_train = classifier.predict(X_train)
                    y_pred_test = classifier.predict(X_test)

                    left_indices_train = y_pred_train == 0
                    right_indices_train = ~left_indices_train
                    left_indices_test = y_pred_test == 0
                    right_indices_test = ~left_indices_test
                else:
                    # Stochastic routing
                    y_probs_train = classifier.predict_proba(X_train)[:, 1]
                    y_probs_test = classifier.predict_proba(X_test)[:, 1]

                    left_indices_train = np.random.random(len(y_train)) < y_probs_train
                    right_indices_train = ~left_indices_train
                    left_indices_test = y_probs_test < 0.5
                    right_indices_test = ~left_indices_test

            if np.sum(left_indices_train) < 10 or np.sum(right_indices_train) < 10:
                continue

            X_left_train, y_left_train = X_train[left_indices_train], y_train[left_indices_train]
            X_right_train, y_right_train = X_train[right_indices_train], y_train[right_indices_train]

            # Train regression models
            regressor_left = LinearRegression().fit(*age_based_mixup(X_left_train, y_left_train))
            regressor_right = LinearRegression().fit(*age_based_mixup(X_right_train, y_right_train))
            
            # Evaluate performance
            left_error = self.total_squared_error(y_left_train, regressor_left.predict(X_left_train))
            right_error = self.total_squared_error(y_right_train, regressor_right.predict(X_right_train))
            
            error = left_error + right_error
            tse_reduction = parent_tse - error
            
            print(f"TSE reduction {tse_reduction} at threshold = {threshold}")
            if tse_reduction > best_reduction:
                best_threshold = threshold
                best_reduction = tse_reduction
                best_classifier = classifier
            
        if best_threshold is not None and best_reduction > 0:
            self.node.threshold = best_threshold
            self.node.classifier = best_classifier

            cls_y_train = (self.y > best_threshold).astype(int)
            cls_y_test = (y_test > best_threshold).astype(int)
            
            # Compute classifier metrics
            if not use_oracle:
                y_pred = self.node.classifier.predict(self.X)
                y_scores = self.node.classifier.predict_proba(self.X)[:, 1]
                self.node.eer_train, _ = self.calculate_eer(cls_y_train, y_scores)
                self.node.acc_train = accuracy_score(cls_y_train, y_pred)

                y_pred = self.node.classifier.predict(X_test)
                y_scores = self.node.classifier.predict_proba(X_test)[:, 1]
                self.node.eer_test, _ = self.calculate_eer(cls_y_test, y_scores)
                self.node.acc_test = accuracy_score(cls_y_test, y_pred)
            else:
                self.node.eer_train = 0.0
                self.node.acc_train = 1.0
                self.node.eer_test = 0.0
                self.node.acc_test = 1.0

            # Split data for child nodes
            if use_oracle:
                left_indices_train = (y_train <= best_threshold)
                right_indices_train = ~left_indices_train
                left_indices_test = (y_test <= best_threshold)
                right_indices_test = ~left_indices_test
            else:
                # Use classifier predictions
                if self.split == "deterministic":
                    y_pred_test = self.node.classifier.predict(X_test)

                    left_indices_train = (y_train <= best_threshold)
                    right_indices_train = ~left_indices_train
                    left_indices_test = (y_pred_test == 0)
                    right_indices_test = ~left_indices_test
                else:
                    y_probs_test = self.node.classifier.predict_proba(X_test)[:, 1]

                    left_indices_train = (y_train <= best_threshold)
                    right_indices_train = ~left_indices_train
                    left_indices_test = y_probs_test < 0.5
                    right_indices_test = ~left_indices_test

            X_left_train, y_left_train = X_train[left_indices_train], y_train[left_indices_train]
            X_right_train, y_right_train = X_train[right_indices_train], y_train[right_indices_train]
            X_left_test, y_left_test = X_test[left_indices_test], y_test[left_indices_test]
            X_right_test, y_right_test = X_test[right_indices_test], y_test[right_indices_test]

            print(f"Left samples: {len(y_left_train)}, Right samples: {len(y_right_train)}")
            
            # Create child nodes
            if np.any(left_indices_train) and np.any(left_indices_test):
                self.node.left = TLM(self.X, self.y, max_depth=self.max_depth, split=self.split,  
                                   current_depth=self.current_depth + 1, 
                                   node_id=self.node.node_id * 2 + 1, parent=self.node, 
                                   parent_mae=self.node.test_mae)
                self.node.left.train_node(X_left_train, y_left_train, X_left_test, y_left_test, 
                                        [t for t in thresholds if t <= best_threshold], use_oracle)
                                        
            if np.any(right_indices_train) and np.any(right_indices_test):
                self.node.right = TLM(self.X, self.y, max_depth=self.max_depth, split=self.split, 
                                    current_depth=self.current_depth + 1, 
                                    node_id=self.node.node_id * 2 + 2, parent=self.node, 
                                    parent_mae=self.node.test_mae)
                self.node.right.train_node(X_right_train, y_right_train, X_right_test, y_right_test, 
                                         [t for t in thresholds if t > best_threshold], use_oracle)
        else:
            self.node.is_leaf = True
            self.node.threshold = None
            self.node.classifier = None

    def total_squared_error(self, y, y_hat):
        return np.sum((y - y_hat)**2)
    
    def calculate_eer(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return EER, eer_threshold  

    def predict(self, X, inference_method="hard", y_true=None):
        if inference_method == "hard":
            return self._predict_hard(X, y_true)
        elif inference_method == "soft":
            predictions, _ = regression_based_inference(self, X)
            return predictions
        else:
            raise ValueError("inference_method must be 'hard' or 'soft'")

    def _predict_hard(self, X, y_true=None):
        if len(X) == 0:
            return np.array([])

        if self.node.is_leaf:
            return self.node.regressor.predict(X)

        # For oracle mode, we need true labels for routing
        if  y_true is not None:
            # Oracle mode: use true labels with threshold
            left_mask = y_true <= self.node.threshold
            right_mask = ~left_mask
        else:
            # Normal mode: use classifier
            y_pred = self.node.classifier.predict(X)
            left_mask = y_pred == 0
            right_mask = ~left_mask

        predictions = np.zeros(X.shape[0])

        if self.node.left and np.any(left_mask):
            y_true_left = y_true[left_mask] if y_true is not None else None
            predictions[left_mask] = self.node.left._predict_hard(X[left_mask], y_true_left)
     
        if self.node.right and np.any(right_mask):
            y_true_right = y_true[right_mask] if y_true is not None else None
            predictions[right_mask] = self.node.right._predict_hard(X[right_mask], y_true_right)
      
        return predictions
    

    def plot_tree(self, graph=None):
        if graph is None:
            graph = Digraph(comment='TLM Tree', format='png')
            
        label = f'Node {self.node.node_id}\n---\nTrain RMSE: {self.node.train_rmse:.2f}\nTrain MAE: {self.node.train_mae:.2f}\nTrain Samples: {self.node.num_train_samples}'
        label += f'\n---\nTest RMSE: {self.node.test_rmse:.2f}\nTest MAE: {self.node.test_mae:.2f}\nTest Samples: {self.node.num_test_samples}'
       
        if not self.node.is_leaf:
            label += f"\n---\nTest Accuracy: {self.node.acc_test*100:.2f}%\nTest EER: {self.node.eer_test*100:.2f}%"
            label += f"\n---\nTrain Accuracy: {self.node.acc_train*100:.2f}%\nTrain EER: {self.node.eer_train*100:.2f}%"
        
        if self.node.threshold is not None:
            label += f'\nThreshold: {self.node.threshold}'
            
        graph.node(str(self.node.node_id), label=label)
        
        if self.node.left:
            mae_change = self.node.left.node.test_mae - self.node.left.node.parent_mae
            color = 'red' if mae_change > 0 else 'green'
            label = f"left: y <= {self.node.threshold}\nChange: {mae_change:+.2f}"
            graph.edge(str(self.node.node_id), str(self.node.left.node.node_id), label=label, color=color)
            self.node.left.plot_tree(graph)
            
        if self.node.right:
            mae_change = self.node.right.node.test_mae - self.node.right.node.parent_mae
            color = 'red' if mae_change > 0 else 'green'
            label = f"right: y > {self.node.threshold}\nChange: {mae_change:+.2f}"
            graph.edge(str(self.node.node_id), str(self.node.right.node.node_id), label=label, color=color)
            self.node.right.plot_tree(graph)
            
        return graph



def regression_based_inference(root, X):
    """Soft routing inference that considers all paths in the tree"""
    
    def traverse_and_predict(node, x, prob=1.0, depth=0):
        pred = node.node.regressor.predict([x])[0]
        
        if node.node.is_leaf:
            return [(node, pred, prob, depth)]
        
        class_prob = node.node.classifier.predict_proba([x])[0]
        label = node.node.classifier.predict([x])[0]
        
        right_results, left_results = [], []
        
        if node.node.left is not None and label == 0:       
            left_results = traverse_and_predict(node.node.left, x, prob * class_prob[0], depth + 1)
        if node.node.right is not None and label == 1:   
            right_results = traverse_and_predict(node.node.right, x, prob * class_prob[1], depth + 1)
        
        return [(node, pred, prob, depth)] + left_results + right_results

    def predict_single(x):
        node_predictions = traverse_and_predict(root, x)
        nodes, preds, probs, depths = zip(*node_predictions)

        probs = np.array(probs)
        depths = np.array(depths)
        preds = np.array(preds)
        nodes = np.array(nodes)
        
        depth_factor = 1.0
        depth_adjusted_probs = np.array(probs) * (depth_factor ** np.array(depths))
        weights = depth_adjusted_probs / np.sum(depth_adjusted_probs)
        weighted_pred = np.sum(np.array(preds) * weights)
        
        return weighted_pred, list(zip(nodes, weights))

    predictions = []
    node_weights = []
    
    for x in X:
        pred, weights = predict_single(x)
        predictions.append(pred)
        node_weights.append(weights)
    
    return np.array(predictions), node_weights