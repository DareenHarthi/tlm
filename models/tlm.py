import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_curve
from graphviz import Digraph
from data_utils import  age_based_mixup


def _fast_row_membership(A, B):
    """Boolean mask over rows of A: True where the row also appears in B.

    Numerically identical to the original
        mask = np.zeros(len(A), bool)
        for row in B: mask |= np.all(A == row, axis=1)
    but O(len(A) + len(B)) instead of O(len(A) * len(B)), which is what makes
    the classifier's sample-weighting tractable once the training set has been
    expanded ~4x by age-based mixup.
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    dt = np.dtype([('', A.dtype)] * A.shape[1])
    return np.isin(A.view(dt).ravel(), B.view(dt).ravel())


class Node():
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
  
        # Error metrics
        self.parent_mae = parent_mae

        # Direction of MAE change for plotting
        self.mae_change = None
        

class TLM():
    def __init__(self, X, y, max_depth=4, split="deterministic", current_depth=0, is_leaf=False, node_id=0, parent=None, parent_mae=None):
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.node = Node(node_id, parent_mae = parent_mae, parent=parent, is_leaf=is_leaf)
        self.X = X
        self.y = y
        self.total_error = 0 
        self.split = split # deterministic or random 
        

    def train_node(self, X_train, y_train, X_test, y_test, thresholds, use_oracle=False):
        print(f"\nTraining node {self.node.node_id}")

        self.node.num_train_samples = len(y_train)
        self.node.num_test_samples = len(y_test)

        # Train the linear regression model
        self.node.regressor.fit(*age_based_mixup(X_train, y_train))
        self.node.mean = np.mean(y_train)
        # A leaf is underdetermined when it has fewer training samples than
        # features (age-based mixup only adds convex combinations, so it does not
        # raise the rank); its 192-dim linear fit then extrapolates unreliably.
        self.node.underdetermined = len(y_train) < X_train.shape[1]

        # Evaluate model on train data
        train_predictions = self.node.regressor.predict(X_train)
        self.node.train_mae = mean_absolute_error(y_train, train_predictions)
        self.node.train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))

        parent_tse = self.total_squared_error(y_train, train_predictions)
        
#         self.total_error = parent_tse

        # Evaluate model on test data (a split can route zero test samples into a
        # node -- e.g. deep oracle splits -- so guard the empty case).
        if len(y_test) > 0:
            test_predictions = self.node.regressor.predict(X_test)
            self.node.test_mae = mean_absolute_error(y_test, test_predictions)
            self.node.test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        else:
            self.node.test_mae = float('nan')
            self.node.test_rmse = float('nan')

        self.node.depth = self.current_depth

        # Check stopping criteria
        if (self.current_depth >= self.max_depth or 
            len(np.unique(y_train)) < 2 or 
            len(y_train) < 200 ):  # Add a minimum error threshold
            self.node.is_leaf = True
            return

        best_threshold = None
        best_reduction = float('-inf')

        for threshold in thresholds:
            cls_y_train = (self.y > threshold).astype(int)

            if len(np.unique(cls_y_train)) < 2:
                continue
            if not use_oracle:


                ######################################
                # Step 1: Train the classifier 
                # Modifief training strategy 

                # liblinear

                sample_weights = np.ones(len(self.y))

                # Down-weight augmented/mixup samples that do not belong to the
                # data currently routed into this node (the "modified training
                # strategy" from the paper). Vectorized membership test.
                current_node_mask = _fast_row_membership(self.X, X_train)

                sample_weights[~current_node_mask] = 0.5  # Reduce weight for data outside current node

                # Train classifier on all data with sample weights
                # newton-cholesky or 
                classifier = LogisticRegression(solver='liblinear', max_iter=1000)
                classifier.fit(self.X, cls_y_train, sample_weight=sample_weights)


            ########################################

            # Step 2: Split the data for the regression models.
                if self.split == "deterministic":

                    y_pred_train = classifier.predict(X_train)
                    y_pred_test = classifier.predict(X_test)

                    left_indices_train = y_pred_train == 0
                    right_indices_train = ~left_indices_train
                    left_indices_test = y_pred_test == 0
                    right_indices_test = ~left_indices_test
                else:
                    # Use probabilities for stochastic routing during training
                    y_probs_train = classifier.predict_proba(X_train)[:, 1]
                    y_probs_test = classifier.predict_proba(X_test)[:, 1]

                    # Flip coins for training data
                    left_indices_train = np.random.random(len(y_train)) < y_probs_train
                    right_indices_train = ~left_indices_train

                    # For test data, we still use deterministic prediction
                    left_indices_test = y_probs_test < 0.5
                    right_indices_test = ~left_indices_test
                        
            else:
                # Oracle: use threshold directly, no classifier needed
                classifier = None
            
     
                # Oracle: use thresholds directly on true labels
                left_indices_train = y_train <= threshold
                right_indices_train = ~left_indices_train
                left_indices_test = y_test <= threshold
                right_indices_test = ~left_indices_test
            




            if np.sum(left_indices_train) < 10 or np.sum(right_indices_train) < 10:
                continue
            
            

            X_left_train, y_left_train = X_train[left_indices_train], y_train[left_indices_train]
            X_right_train, y_right_train = X_train[right_indices_train], y_train[right_indices_train]


            X_left_test, y_left_test = X_test[left_indices_test], y_test[left_indices_test]
            X_right_test, y_right_test = X_test[right_indices_test], y_test[right_indices_test]
            
            ########################################
            
            # Step 3: Train the regression models
            
            regressor_left = LinearRegression().fit(*age_based_mixup(X_left_train, y_left_train))
            regressor_right = LinearRegression().fit(*age_based_mixup(X_right_train, y_right_train))
            
            ########################################
            
            # Step 4: Evaluate the performance of the current models 

            left_error = self.total_squared_error(y_left_train, regressor_left.predict(X_left_train))
            right_error = self.total_squared_error(y_right_train, regressor_right.predict(X_right_train))
            
            error = left_error + right_error
            tse_reduction = parent_tse  - error
            
            
            #self.net_gain(parent_tse, len(y_train), 
                               #      left_error, len(y_left_test), 
                                 #    right_error, len(y_right_test))
            ########################################
            
            # Step 5: Update the variables when the condition is met  

            print(f"tse_reduction {tse_reduction} at threshold = {threshold}")
            if tse_reduction > best_reduction:
            
                best_threshold = threshold
                best_reduction = tse_reduction
                best_classifier = classifier
                
            ########################################
            
        
            
        if best_threshold is not None and best_reduction > 0:
            self.node.threshold = best_threshold
            self.node.classifier = best_classifier
            
            if not use_oracle:

                cls_y_train = (self.y > best_threshold).astype(int)
                cls_y_test = (y_test > best_threshold).astype(int)

                print(len(np.unique(cls_y_train)))
                print(len(np.unique(cls_y_test)))

                ########################################
                # Step 1: compute EER and accuarcy for the current training data and testing data
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
                ########################################
            
            # Step 2: Split the data

            
            # Split the data for regression models
            if use_oracle:
                # Oracle: use thresholds directly on true labels
                left_indices_train = y_train <= best_threshold
                right_indices_train = ~left_indices_train
                left_indices_test = y_test <= best_threshold
                right_indices_test = ~left_indices_test
            else:
            
                if self.split == "deterministic":

                    y_pred_train = self.node.classifier.predict(X_train)
                    y_pred_test = self.node.classifier.predict(X_test)

                    # Split based on threshold 
                    left_indices_train = (y_train <= best_threshold) #(y_pred_train==0) 
                    right_indices_train = ~left_indices_train

                    left_indices_test = (y_pred_test==0)
                    right_indices_test = ~left_indices_test

                else:
                    # Use probabilities for stochastic routing during training
    #                 y_probs_train = self.node.classifier.predict_proba(X_train)[:, 1]
                    y_probs_test = self.node.classifier.predict_proba(X_test)[:, 1]

                    # Flip coins for training data
    #                 left_indices_train = np.random.random(len(y_train)) < y_probs_train
                    left_indices_train = (y_train <= best_threshold)
                    right_indices_train = ~left_indices_train

                    # For test data, we still use deterministic prediction
                    left_indices_test = y_probs_test < 0.5
                    right_indices_test = ~left_indices_test

            ########################################
            
            # Step 3: Create left and right nodes 

                                                                                 
            X_left_train, y_left_train = X_train[left_indices_train], y_train[left_indices_train]
            X_right_train, y_right_train = X_train[right_indices_train], y_train[right_indices_train]
            X_left_test, y_left_test = X_test[left_indices_test], y_test[left_indices_test]
            X_right_test, y_right_test = X_test[right_indices_test], y_test[right_indices_test]

            
            print(f"right len = {len(y_right_train)}")
            print(f"left len = {len(y_left_train)}")
            # Train left branch
            cond_left = np.any(left_indices_train)

            cond_right = np.any(right_indices_train)
    
            
            if cond_left:
                
            
                self.node.left = TLM(self.X, self.y, max_depth=self.max_depth, split=self.split,  current_depth=self.current_depth + 1, 
                                     node_id=self.node.node_id * 2 + 1, parent_mae=self.node.test_mae)

                self.node.left.train_node(X_left_train, y_left_train, X_left_test, y_left_test,
                                          [t for t in thresholds if t <= best_threshold], use_oracle=use_oracle)
            if cond_right:
                # Train right branch
                self.node.right = TLM(self.X, self.y, max_depth=self.max_depth, split=self.split, current_depth=self.current_depth + 1, 
                                      node_id=self.node.node_id * 2 + 2, parent_mae=self.node.test_mae)

                self.node.right.train_node(X_right_train, y_right_train, X_right_test, y_right_test,
                                           [t for t in thresholds if t > best_threshold], use_oracle=use_oracle)
        else:
            self.node.is_leaf = True
            self.node.threshold = None
            self.node.classifier = None


    def mae_total(self, left, l_count, right, r_count):
        left = left * l_count
        right = right * r_count

        total =  (left + right) / (l_count+ r_count)

        return total
    
    def total_squared_error(self, y, y_hat):
        return np.sum((y - y_hat)**2)
    
    def net_gain(self, mae, samples, left_mae, left_samples, right_mae, right_samples ):
        
        parent = mae * samples 
        child_left = left_mae * left_samples 
        child_right = right_mae * right_samples 
        
        total = samples  + left_samples + right_samples 

        childern = child_left + child_right
        
        net_gain = (parent - childern) / total 
        
        return net_gain
    
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
            # Oracle routing can send test samples to very small leaves whose
            # linear regressor extrapolates wildly; fall back to the leaf mean.
            if y_true is not None and getattr(self.node, 'underdetermined', False):
                return np.full(X.shape[0], self.node.mean)
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

        # Route to each child; if a side has no child (e.g. it was never created),
        # fall back to this node's own regressor so no sample is left unpredicted.
        if np.any(left_mask):
            if self.node.left is not None:
                y_true_left = y_true[left_mask] if y_true is not None else None
                predictions[left_mask] = self.node.left._predict_hard(X[left_mask], y_true_left)
            else:
                predictions[left_mask] = self.node.regressor.predict(X[left_mask])

        if np.any(right_mask):
            if self.node.right is not None:
                y_true_right = y_true[right_mask] if y_true is not None else None
                predictions[right_mask] = self.node.right._predict_hard(X[right_mask], y_true_right)
            else:
                predictions[right_mask] = self.node.regressor.predict(X[right_mask])

        return predictions
    

    def predict_depth(self, X, depth):
        if len(X) == 0 :
            return np.array([])

        if self.node.is_leaf or self.node.depth == depth:
            print(f"depth {self.node.depth}")
            print(f"node id {self.node.node_id}")
            return self.node.regressor.predict(X)


        y_pred = self.node.classifier.predict(X)

        predictions = np.zeros(X.shape[0])
        left_mask = y_pred == 0
        right_mask = ~left_mask

        if self.node.left and np.any(left_mask):
            if self.node.left.node.depth <=depth:
                predictions[left_mask] = self.node.left.predict_depth(X[left_mask], depth)
            else:
                predictions[left_mask] = self.node.regressor.predict(X[left_mask])
                
     
        if self.node.right and np.any(right_mask):
            if self.node.right.node.depth <=depth:
                predictions[right_mask] = self.node.right.predict_depth(X[right_mask], depth)
            else:
                predictions[right_mask] = self.node.regressor.predict(X[right_mask])
                
                
        return predictions
    
    
    def predict_node(self, X, node_id):
        if len(X) == 0:
            return np.array([])

        if self.node.is_leaf or self.node.node_id == node_id:
            print(self.node.node_id)
            return self.node.regressor.predict(X)
        
  
        y_pred = self.node.classifier.predict(X)

        predictions = np.zeros(X.shape[0])
        left_mask = y_pred == 0
        right_mask = ~left_mask

        if self.node.left and np.any(left_mask):
            if self.node.left.node.node_id <= node_id:
                predictions[left_mask] = self.node.left.predict_node(X[left_mask], node_id)
            else:
                predictions[left_mask] = self.node.regressor.predict(X[left_mask])
                
   
        if self.node.right and np.any(right_mask):
            if self.node.right.node.node_id <=node_id:
                predictions[right_mask] = self.node.right.predict_node(X[right_mask], node_id)
            else:
                predictions[right_mask] = self.node.regressor.predict(X[right_mask])
                
        return predictions





    def plot_tree(self, graph=None):
        if graph is None:
            graph = Digraph(comment='Balanced TLM Tree', format='png')
  
            
        label = f'Node {self.node.node_id}\n---\nTrain RMSE: {self.node.train_rmse:.2f}\nTrain MAE: {self.node.train_mae:.2f}\nTrain Samples: {self.node.num_train_samples}'
        
        label += f'\n---\nTest RMSE: {self.node.test_rmse:.2f}\nTest MAE: {self.node.test_mae:.2f}\nTest Samples: {self.node.num_test_samples}'
       
        if not self.node.is_leaf:
            label += f"\n---\nTest Accuracy: {self.node.acc_test*100:.2f}%\nTest EER: {self.node.eer_test*100:.2f}%  "
            label += f"\n---\nTrain Accuracy: {self.node.acc_train*100:.2f}%\nTrain EER: {self.node.eer_train*100:.2f}%  "
        
        if self.node.threshold is not None:
            label += f'\nThreshold: {self.node.threshold}'
            
        if self.node.left and self.node.right:
            parent = self.node.test_mae * self.node.num_test_samples 
            child_left = self.node.left.node.test_mae * self.node.left.node.num_test_samples 
            child_right = self.node.right.node.test_mae * self.node.right.node.num_test_samples 
            total = self.node.num_test_samples  + self.node.left.node.num_test_samples + self.node.right.node.num_test_samples 
            childern = child_left + child_right
            net_gain = (parent - childern) / total 
            
            label += f'\n---\nNet Gain: {net_gain:.4f}'
            
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