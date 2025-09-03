import torch
import torch.nn as nn
from collections import deque

class DecisionTree(nn.Module):
    def __init__(self, max_depth, min_samples_split, n_classes, n_thresholds=32, device='cuda'):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes = n_classes
        self.n_thresholds = n_thresholds
        self.device = device
        
        # Initialize tree attributes (replacing dictionary)
        self.register_buffer('tree_feature', None)
        self.register_buffer('tree_threshold', None)
        self.register_buffer('tree_left', None)
        self.register_buffer('tree_right', None)
        self.register_buffer('tree_value', None)
        self.register_buffer('tree_is_leaf', None)

    def fit(self, X, y, feature_mask):
        print(f"Training DecisionTree on device: {self.device}")
        n_samples, n_features = X.shape
        max_nodes = 2 ** (self.max_depth + 1) - 1
        
        # Initialize tree tensors
        self.tree_feature = torch.full((max_nodes,), -1, dtype=torch.long, device=self.device)
        self.tree_threshold = torch.zeros((max_nodes,), device=self.device)
        self.tree_left = torch.full((max_nodes,), -1, dtype=torch.long, device=self.device)
        self.tree_right = torch.full((max_nodes,), -1, dtype=torch.long, device=self.device)
        self.tree_value = torch.zeros((max_nodes, self.n_classes), device=self.device)
        self.tree_is_leaf = torch.zeros((max_nodes,), dtype=torch.bool, device=self.device)
        
        queue = deque()
        queue.append((0, torch.arange(n_samples, device=self.device), 0))
        next_index = 1
        
        while queue:
            node_idx, idxs, depth = queue.popleft()
            X_node, y_node = X[idxs], y[idxs]
            n_node = X_node.shape[0]
            
            print(f"\nProcessing node {node_idx} (depth={depth}, samples={n_node})")
            
            if depth >= self.max_depth:
                print(f"  => Max depth reached, making leaf node")
                self._make_leaf(node_idx, y_node)
                continue
                
            if n_node < self.min_samples_split:
                print(f"  => Insufficient samples, making leaf node")
                self._make_leaf(node_idx, y_node)
                continue
                
            if (y_node == y_node[0]).all():
                print(f"  => All samples same class, making leaf node")
                self._make_leaf(node_idx, y_node)
                continue
                
            best_gain = -1
            best_feature, best_threshold = -1, 0.0
            best_left_mask = None
            
            candidate_features = torch.nonzero(feature_mask, as_tuple=True)[0]
            print(f"  Evaluating {len(candidate_features)} candidate features...")
            
            for feat in candidate_features:
                gain, threshold, left_mask = self._find_best_split(X_node[:, feat], y_node)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = threshold
                    best_left_mask = left_mask
            
            if best_gain > 0 and best_left_mask is not None:
                print(f"  Best split: feature={best_feature}, threshold={best_threshold:.4f}, gain={best_gain:.4f}")
                self.tree_feature[node_idx] = best_feature
                self.tree_threshold[node_idx] = best_threshold
                self.tree_left[node_idx] = next_index
                self.tree_right[node_idx] = next_index + 1
                
                print(f"  Creating children: left={next_index}, right={next_index+1}")
                queue.append((next_index, idxs[best_left_mask], depth + 1))
                queue.append((next_index + 1, idxs[~best_left_mask], depth + 1))
                next_index += 2
            else:
                print(f"  No valid split found, making leaf node")
                self._make_leaf(node_idx, y_node)
        print(f"Tree construction complete! Total nodes: {next_index}")
    
    def _find_best_split(self, X_feat, y):
        unique_vals = torch.unique(X_feat)
        if unique_vals.numel() <= 1:
            return -1, 0.0, None
        
        n_thresholds = min(self.n_thresholds, unique_vals.numel() - 1)
        thresholds = torch.quantile(unique_vals, torch.linspace(0, 1, n_thresholds + 2, device=self.device)[1:-1])
        
        left_masks = X_feat.unsqueeze(1) <= thresholds.unsqueeze(0)
        total_counts = left_masks.sum(dim=0)
        valid_thresholds = total_counts > 0
        
        if not valid_thresholds.any():
            return -1, 0.0, None
            
        valid_thresholds_idx = torch.where(valid_thresholds)[0]
        left_masks = left_masks[:, valid_thresholds_idx]
        thresholds = thresholds[valid_thresholds_idx]
        
        left_counts = torch.zeros((len(valid_thresholds_idx), self.n_classes), device=self.device)
        right_counts = torch.zeros((len(valid_thresholds_idx), self.n_classes), device=self.device)
        
        for c in range(self.n_classes):
            class_mask = (y == c)
            left_counts[:, c] = (left_masks & class_mask.unsqueeze(1)).sum(dim=0)
            right_counts[:, c] = (~left_masks & class_mask.unsqueeze(1)).sum(dim=0)
        
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(left_counts)
        right_entropy = self._entropy(right_counts)
        
        n_total = len(y)
        info_gain = parent_entropy - (
            left_counts.sum(dim=1) / n_total * left_entropy +
            right_counts.sum(dim=1) / n_total * right_entropy
        )
        
        best_idx = torch.argmax(info_gain)
        return info_gain[best_idx].item(), thresholds[best_idx], left_masks[:, best_idx]
    
    def _entropy(self, counts):
        if counts.dim() == 1:
            counts = counts.unsqueeze(0)
        p = counts / counts.sum(dim=1, keepdim=True).clamp(min=1e-10)
        return -(p * torch.log2(p + 1e-10)).sum(dim=1)
    
    def _make_leaf(self, node_idx, y_node):
        counts = torch.bincount(y_node, minlength=self.n_classes)
        self.tree_value[node_idx] = counts.float()
        self.tree_is_leaf[node_idx] = True
        print(f"  Leaf node {node_idx}: class distribution = {counts.cpu().numpy()}")
    
    def forward(self, X):
        n_samples = X.shape[0]
        current_nodes = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        is_leaf = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        predictions = torch.zeros((n_samples, self.n_classes), device=self.device)
        
        while not is_leaf.all():
            active = ~is_leaf
            if not active.any():
                break
                
            node_indices = current_nodes[active]
            features = self.tree_feature[node_indices]
            thresholds = self.tree_threshold[node_indices]
            
            x_vals = X[active, features]
            decisions = x_vals <= thresholds
            
            next_nodes = torch.where(
                decisions,
                self.tree_left[node_indices],
                self.tree_right[node_indices]
            )
            
            new_leaf_mask = self.tree_is_leaf[next_nodes]
            leaf_nodes = next_nodes[new_leaf_mask]
            
            if new_leaf_mask.any():
                active_indices = torch.where(active)[0]
                leaf_indices = active_indices[new_leaf_mask]
                predictions[leaf_indices] = self.tree_value[leaf_nodes]
            
            current_nodes[active] = next_nodes
            is_leaf[active] = self.tree_is_leaf[next_nodes]
        
        return predictions

class ParallelRandomForest(nn.Module):
    def __init__(self, n_trees, n_classes, max_depth=10, min_samples_split=2, 
                 feature_ratio=0.3, device='cuda'):
        super().__init__()
        self.n_trees = n_trees
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_ratio = feature_ratio
        self.device = device
        self.trees = nn.ModuleList()
        
    def fit(self, X, y, batch_size=None):
        print(f"\n{'='*50}")
        print(f"Training ParallelRandomForest on {self.device.upper()}")
        print(f"Parameters: {self.n_trees} trees, max_depth={self.max_depth}")
        print(f"            min_samples_split={self.min_samples_split}, feature_ratio={self.feature_ratio}")
        print(f"{'='*50}\n")
        
        n_samples, n_features = X.shape
        n_selected_features = max(1, int(self.feature_ratio * n_features))
        
        for i in range(self.n_trees):
            print(f"\nTraining tree {i+1}/{self.n_trees}")
            indices = torch.randint(0, n_samples, (n_samples,), device=self.device)
            X_boot, y_boot = X[indices], y[indices]
            
            feature_mask = torch.zeros(n_features, dtype=torch.bool, device=self.device)
            selected = torch.randperm(n_features, device=self.device)[:n_selected_features]
            feature_mask[selected] = True
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_classes=self.n_classes,
                device=self.device
            )
            tree.fit(X_boot, y_boot, feature_mask)
            self.trees.append(tree)
            print(f"Completed tree {i+1}/{self.n_trees}")
        print("\nRandom Forest training completed!")
        return self  # Return the trained model
    
    def forward(self, X):
        all_preds = []
        for i, tree in enumerate(self.trees):
            print(f"Tree {i+1} predicting...")
            preds = tree(X)
            all_preds.append(preds.unsqueeze(0))
        
        pred_tensor = torch.cat(all_preds, dim=0)
        return pred_tensor.mean(dim=0)
    
    def predict(self, X):
        with torch.no_grad():
            print("Starting forest prediction...")
            probs = self.forward(X)
            print("Prediction complete!")
            return probs.argmax(dim=1)
