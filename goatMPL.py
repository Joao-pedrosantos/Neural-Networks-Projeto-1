"""
🐐 PURE MLP GOAT - MULTI-LAYER PERCEPTRON DEFINITIVO PARA 97%+
Versão que usa APENAS técnicas de MLP puro, sem batch norm ou outras redes

TÉCNICAS MLP PERMITIDAS:
✅ Arquitetura profunda (múltiplas camadas hidden)
✅ Funções de ativação clássicas (ReLU, Leaky ReLU, Sigmoid, Tanh)
✅ Dropout (regularização padrão para MLPs)
✅ Diferentes loss functions (Focal Loss, BCE, Weighted)
✅ Class weights para imbalance
✅ Otimizadores avançados (Adam com momentum)
✅ Learning rate scheduling
✅ Early stopping
✅ Ensemble de múltiplos MLPs
✅ Feature engineering avançado
✅ Preprocessing inteligente

❌ Batch Normalization (não é MLP puro)
❌ Funções de ativação muito complexas (Mish, Swish)
❌ Outros tipos de redes neurais
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class PureMLPActivations:
    """Funções de ativação clássicas para MLP"""
    
    @staticmethod
    def relu(x):
        """ReLU: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(x):
        """Gradiente do ReLU"""
        return (x > 0).astype(np.float32)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU: max(alpha*x, x)"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_grad(x, alpha=0.01):
        """Gradiente do Leaky ReLU"""
        return np.where(x > 0, 1.0, alpha)
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid estável"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-np.clip(x, -500, 500))),
                       np.exp(np.clip(x, -500, 500)) / (1 + np.exp(np.clip(x, -500, 500))))
    
    @staticmethod
    def sigmoid_grad(x):
        """Gradiente do Sigmoid"""
        s = PureMLPActivations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """Tanh"""
        return np.tanh(np.clip(x, -500, 500))
    
    @staticmethod
    def tanh_grad(x):
        """Gradiente do Tanh"""
        t = PureMLPActivations.tanh(x)
        return 1 - t * t

class PureMLPLossFunctions:
    """Loss functions otimizadas para MLPs"""
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred, eps=1e-12):
        """Binary Cross-Entropy clássico"""
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        y_true = np.array(y_true)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def weighted_bce(y_true, y_pred, class_weights, eps=1e-12):
        """BCE com class weights"""
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        y_true = np.array(y_true)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        weights = np.where(y_true == 1, class_weights[1], class_weights[0])
        return -weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, eps=1e-12):
        """Focal Loss para class imbalance"""
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        y_true = np.array(y_true)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Calcular p_t (probabilidade da classe correta)
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        
        # Calcular alpha_t (peso da classe)
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = alpha_t * np.power(1 - p_t, gamma)
        
        # Focal loss
        return -focal_weight * np.log(p_t)

class PureMLPOptimizer:
    """Otimizador Adam para MLP"""
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-5):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, params, grads):
        """Atualização Adam com weight decay"""
        self.t += 1
        
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Weight decay (L2 regularization)
            if 'W' in key:
                grads[key] += self.weight_decay * params[key]
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class PureMLPFeatureEngineering:
    """Feature engineering especializado para MLPs"""
    
    @staticmethod
    def create_domain_features(df):
        """Criar features baseadas no domínio bancário"""
        df_fe = df.copy()
        
        # 1. Job scoring baseado em propensão histórica
        job_success_scores = {
            'management': 0.85, 'admin.': 0.72, 'technician': 0.65, 'entrepreneur': 0.88,
            'blue-collar': 0.42, 'services': 0.55, 'self-employed': 0.78, 'housemaid': 0.35,
            'retired': 0.68, 'unemployed': 0.25, 'student': 0.58, 'unknown': 0.45
        }
        df_fe['job_success_score'] = df_fe['job'].map(job_success_scores).fillna(0.45)
        
        # 2. Features de interação importantes
        df_fe['age_balance_ratio'] = df_fe['age'] / (np.abs(df_fe['balance']) + 1)
        df_fe['duration_per_contact'] = df_fe['duration'] / (df_fe['campaign'] + 1)
        df_fe['contact_efficiency'] = np.log1p(df_fe['duration']) / (df_fe['campaign'] + 1)
        
        # 3. Features temporais
        summer_months = ['may', 'jun', 'jul', 'aug']
        df_fe['is_summer_campaign'] = df_fe['month'].isin(summer_months).astype(int)
        
        # Dias da semana (assumindo day é dia do mês)
        df_fe['is_month_end'] = (df_fe['day'] >= 28).astype(int)
        df_fe['is_month_start'] = (df_fe['day'] <= 3).astype(int)
        
        # 4. Score de saúde financeira
        df_fe['financial_stability'] = (
            (df_fe['balance'] > 0).astype(int) * 3 +
            (df_fe['default'] == 'no').astype(int) * 2 +
            (df_fe['loan'] == 'no').astype(int) * 1 +
            (df_fe['housing'] == 'yes').astype(int) * 1  # Ter casa é estabilidade
        )
        
        # 5. Historical campaign features
        df_fe['was_contacted_before'] = (df_fe['pdays'] != -1).astype(int)
        df_fe['previous_success'] = (df_fe['poutcome'] == 'success').astype(int)
        
        # Para pdays = -1 (nunca contactado), usar valor neutro
        df_fe['pdays_normalized'] = np.where(df_fe['pdays'] == -1, 365, df_fe['pdays'])
        df_fe['recency_score'] = 1 / (df_fe['pdays_normalized'] + 1)
        
        # 6. Features polinomiais seletivas
        df_fe['age_squared'] = df_fe['age'] ** 2
        df_fe['balance_log'] = np.sign(df_fe['balance']) * np.log1p(np.abs(df_fe['balance']))
        df_fe['duration_sqrt'] = np.sqrt(df_fe['duration'])
        
        # 7. Interaction features complexas
        df_fe['age_education_interaction'] = df_fe['age'] * df_fe['education'].map({
            'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 1.5
        }).fillna(1.5)
        
        # 8. Contact type scoring
        contact_scores = {'cellular': 0.8, 'telephone': 0.6, 'unknown': 0.3}
        df_fe['contact_quality_score'] = df_fe['contact'].map(contact_scores).fillna(0.3)
        
        return df_fe
    
    @staticmethod
    def advanced_preprocessing(df_train, df_test=None):
        """Preprocessing otimizado para MLPs"""
        
        # Feature engineering
        df_train_fe = PureMLPFeatureEngineering.create_domain_features(df_train)
        if df_test is not None:
            df_test_fe = PureMLPFeatureEngineering.create_domain_features(df_test)
        
        # Identificar tipos de features
        numeric_features = df_train_fe.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col not in ['id', 'y']]
        
        categorical_features = df_train_fe.select_dtypes(include=['object']).columns.tolist()
        
        # Análise de skewness para escolher scaler apropriado
        skewed_features = []
        normal_features = []
        
        for col in numeric_features:
            skewness = abs(df_train_fe[col].skew())
            if skewness > 1.5:  # Threshold para skewness
                skewed_features.append(col)
            else:
                normal_features.append(col)
        
        print(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
        print(f"  - Normal distribution: {len(normal_features)}")
        print(f"  - Skewed distribution: {len(skewed_features)}")
        
        # Pipeline de preprocessing otimizado para MLPs
        preprocessor = ColumnTransformer([
            ('normal', StandardScaler(), normal_features),
            ('skewed', RobustScaler(), skewed_features),  # Robust para outliers
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             categorical_features)
        ])
        
        # Preparar dados
        X_train = df_train_fe.drop(columns=['id', 'y'] if 'y' in df_train_fe.columns else ['id'])
        y_train = df_train_fe['y'] if 'y' in df_train_fe.columns else None
        
        X_train_processed = preprocessor.fit_transform(X_train)
        
        result = {
            'X_train': X_train_processed,
            'y_train': y_train,
            'preprocessor': preprocessor,
            'n_features': X_train_processed.shape[1]
        }
        
        if df_test is not None:
            X_test = df_test_fe.drop(columns=['id'] + (['y'] if 'y' in df_test_fe.columns else []))
            result['X_test'] = preprocessor.transform(X_test)
            result['test_ids'] = df_test_fe['id']
        
        return result

class PureMLPGoat:
    """
    🐐 Pure Multi-Layer Perceptron GOAT
    MLP puro otimizado para máxima performance
    """
    
    def __init__(self, input_dim, hidden_sizes='auto', activation='leaky_relu', 
                 lr=0.001, dropout_rate=0.3, class_weights=None, loss_type='focal',
                 seed=42):
        
        self.rng = np.random.RandomState(seed)
        
        # Arquitetura adaptativa baseada no input
        if hidden_sizes == 'auto':
            if input_dim < 50:
                hidden_sizes = [512, 256, 128]
            elif input_dim < 100:
                hidden_sizes = [768, 384, 192]
            else:
                hidden_sizes = [1024, 512, 256, 128]
        
        self.sizes = [input_dim] + hidden_sizes + [1]
        self.L = len(self.sizes) - 1
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.class_weights = class_weights
        self.loss_type = loss_type
        
        # Inicializar parâmetros
        self._initialize_parameters()
        
        # Optimizer
        self.optimizer = PureMLPOptimizer(lr=lr, weight_decay=1e-6)
        
        # Learning rate scheduling
        self.initial_lr = lr
        self.epoch = 0
        
        # Early stopping
        self.best_score = -np.inf
        self.patience_counter = 0
        self.best_params = None
        
        print(f"🐐 Pure MLP initialized:")
        print(f"   Architecture: {self.sizes}")
        print(f"   Activation: {self.activation}")
        print(f"   Loss: {self.loss_type}")
        print(f"   Total parameters: ~{self._count_parameters():,}")
    
    def _count_parameters(self):
        """Contar parâmetros totais"""
        total = 0
        for i in range(self.L):
            total += self.sizes[i] * self.sizes[i+1]  # Weights
            total += self.sizes[i+1]  # Biases
        return total
    
    def _initialize_parameters(self):
        """Inicialização He/Xavier otimizada"""
        self.params = {}
        
        for i in range(self.L):
            in_dim, out_dim = self.sizes[i], self.sizes[i+1]
            
            # He initialization para ReLU/Leaky ReLU, Xavier para sigmoid/tanh
            if self.activation in ['relu', 'leaky_relu']:
                std = np.sqrt(2.0 / in_dim)  # He
            else:
                std = np.sqrt(1.0 / in_dim)  # Xavier
            
            self.params[f"W{i+1}"] = self.rng.randn(out_dim, in_dim) * std
            self.params[f"b{i+1}"] = np.zeros((out_dim, 1))
    
    def _get_activation_function(self, x):
        """Aplicar função de ativação escolhida"""
        if self.activation == 'relu':
            return PureMLPActivations.relu(x), PureMLPActivations.relu_grad(x)
        elif self.activation == 'leaky_relu':
            return PureMLPActivations.leaky_relu(x), PureMLPActivations.leaky_relu_grad(x)
        elif self.activation == 'sigmoid':
            return PureMLPActivations.sigmoid(x), PureMLPActivations.sigmoid_grad(x)
        elif self.activation == 'tanh':
            return PureMLPActivations.tanh(x), PureMLPActivations.tanh_grad(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _apply_dropout(self, x, training=True):
        """Aplicar dropout"""
        if not training or self.dropout_rate == 0:
            return x, None
        
        keep_prob = 1 - self.dropout_rate
        mask = self.rng.binomial(1, keep_prob, size=x.shape) / keep_prob
        return x * mask, mask
    
    def forward(self, X, training=True):
        """Forward pass do MLP"""
        # Converter para numpy se necessário
        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X)
        
        cache = {}
        dropout_masks = {}
        
        A = X.T  # (input_dim, batch_size)
        cache["A0"] = A
        
        for i in range(1, self.L + 1):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            
            # Linear transformation
            Z = W.dot(A) + b
            cache[f"Z{i}"] = Z
            
            if i < self.L:  # Hidden layers
                # Activation function
                A, activation_grad = self._get_activation_function(Z)
                cache[f"activation_grad{i}"] = activation_grad
                
                # Dropout
                A, dropout_mask = self._apply_dropout(A, training)
                dropout_masks[f"dropout{i}"] = dropout_mask
            else:  # Output layer
                A = Z  # Linear output
            
            cache[f"A{i}"] = A
        
        # Apply sigmoid to output for probabilities
        logits = cache[f"A{self.L}"]
        probs = PureMLPActivations.sigmoid(logits.ravel())
        
        cache["probs"] = probs
        cache["dropout_masks"] = dropout_masks
        
        return probs, cache
    
    def _compute_loss(self, y_true, y_pred):
        """Computar loss baseado no tipo"""
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        y_true = np.array(y_true)
        
        if self.loss_type == 'bce':
            return PureMLPLossFunctions.binary_crossentropy(y_true, y_pred).mean()
        elif self.loss_type == 'weighted':
            return PureMLPLossFunctions.weighted_bce(y_true, y_pred, self.class_weights).mean()
        elif self.loss_type == 'focal':
            return PureMLPLossFunctions.focal_loss(y_true, y_pred).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def backward(self, cache, y_true):
        """Backward pass do MLP"""
        grads = {}
        
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        y_true = np.array(y_true)
        
        m = y_true.shape[0]
        probs = cache["probs"]
        dropout_masks = cache["dropout_masks"]
        
        # Gradient do output (simplificado para sigmoid + BCE/Focal)
        if self.loss_type == 'focal':
            # Gradiente focal loss (aproximação)
            p_t = np.where(y_true == 1, probs, 1 - probs)
            alpha_t = np.where(y_true == 1, 0.25, 0.75)
            focal_factor = alpha_t * 2.0 * np.power(1 - p_t, 1.0)  # gamma=2, simplified
            dA = focal_factor * (probs - y_true) / m
        else:
            # Standard gradient for BCE (with optional weighting)
            if self.loss_type == 'weighted' and self.class_weights:
                weights = np.where(y_true == 1, self.class_weights[1], self.class_weights[0])
                dA = weights * (probs - y_true) / m
            else:
                dA = (probs - y_true) / m
        
        dA = dA.reshape(1, -1)
        
        # Backpropagation através das camadas
        for i in range(self.L, 0, -1):
            A_prev = cache[f"A{i-1}"]
            W_i = self.params[f"W{i}"]
            
            # Gradientes dos parâmetros
            grads[f"dW{i}"] = dA.dot(A_prev.T)
            grads[f"db{i}"] = dA.sum(axis=1, keepdims=True)
            
            # Propagar para camada anterior
            if i > 1:
                dA_prev = W_i.T.dot(dA)
                
                # Gradient da ativação
                if f"activation_grad{i-1}" in cache:
                    dA_prev = dA_prev * cache[f"activation_grad{i-1}"]
                
                # Gradient do dropout
                if f"dropout{i-1}" in dropout_masks and dropout_masks[f"dropout{i-1}"] is not None:
                    dA_prev = dA_prev * dropout_masks[f"dropout{i-1}"]
                
                dA = dA_prev
        
        return grads
    
    def _update_learning_rate(self):
        """Learning rate scheduling cosine"""
        # Cosine annealing
        T_max = 30  # Ciclo de 30 epochs
        eta_min = self.initial_lr * 0.01
        T_cur = self.epoch % T_max
        lr = eta_min + (self.initial_lr - eta_min) * (1 + np.cos(np.pi * T_cur / T_max)) / 2
        self.optimizer.lr = lr
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=256, 
            patience=25, verbose=True):
        """Treinar o MLP"""
        
        print(f"🚀 Starting Pure MLP training...")
        
        history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            self.epoch = epoch
            self._update_learning_rate()
            
            # Shuffle data
            perm = self.rng.permutation(n_samples)
            X_shuff = X_train[perm]
            y_shuff = y_train[perm]
            
            # Mini-batch training
            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                xb = X_shuff[start:start + batch_size]
                yb = y_shuff[start:start + batch_size]
                
                # Forward pass
                probs, cache = self.forward(xb, training=True)
                loss = self._compute_loss(yb, probs)
                epoch_losses.append(loss)
                
                # Backward pass
                grads = self.backward(cache, yb)
                
                # Convert gradients for optimizer
                opt_grads = {}
                for key, val in grads.items():
                    param_key = key[1:]  # Remove 'd' prefix
                    opt_grads[param_key] = val
                
                # Update parameters
                self.optimizer.update(self.params, opt_grads)
            
            # Validation
            train_loss = np.mean(epoch_losses)
            val_probs, _ = self.forward(X_val, training=False)
            val_loss = self._compute_loss(y_val, val_probs)
            val_auc = roc_auc_score(y_val, val_probs)
            val_acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            history["val_acc"].append(val_acc)
            
            # Early stopping
            if val_auc > self.best_score:
                self.best_score = val_auc
                self.best_params = {k: v.copy() for k, v in self.params.items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}, "
                      f"val_acc={val_acc:.4f}, lr={self.optimizer.lr:.6f}")
            
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best AUC: {self.best_score:.4f}")
                break
        
        # Restore best parameters
        if self.best_params:
            self.params = self.best_params
        
        print(f"✅ Training complete! Best validation AUC: {self.best_score:.4f}")
        return history
    
    def predict_proba(self, X):
        """Predição de probabilidades"""
        probs, _ = self.forward(X, training=False)
        return probs
    
    def predict(self, X, threshold=0.5):
        """Predição de classes"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

class PureMLPEnsemble:
    """Ensemble de múltiplos MLPs puros"""
    
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
        self.weights = None
    
    def fit(self, X_train, y_train, X_val, y_val):
        """Treinar ensemble de MLPs"""
        print(f"🎯 Training Pure MLP Ensemble with {self.n_models} models...")
        
        # Configurações diferentes para diversidade
        configurations = [
            {'hidden_sizes': [768, 384, 192], 'activation': 'leaky_relu', 'loss_type': 'focal'},
            {'hidden_sizes': [1024, 512, 256], 'activation': 'relu', 'loss_type': 'weighted'},
            {'hidden_sizes': [512, 256, 128, 64], 'activation': 'leaky_relu', 'loss_type': 'focal'},
            {'hidden_sizes': [1024, 256, 64], 'activation': 'tanh', 'loss_type': 'weighted'},
            {'hidden_sizes': [896, 448, 224], 'activation': 'leaky_relu', 'loss_type': 'focal'}
        ]
        
        val_scores = []
        
        for i in range(self.n_models):
            config = configurations[i % len(configurations)]
            
            print(f"   Training MLP {i+1}/{self.n_models}: {config['activation']} + {config['loss_type']}")
            
            # Class weights
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = {classes[j]: weights[j] for j in range(len(classes))}
            
            model = PureMLPGoat(
                input_dim=X_train.shape[1],
                class_weights=class_weights,
                seed=42 + i * 17,  # Different seeds for diversity
                **config
            )
            
            model.fit(X_train, y_train, X_val, y_val, epochs=80, verbose=False)
            
            val_pred = model.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_pred)
            val_scores.append(val_auc)
            
            self.models.append(model)
            print(f"      MLP {i+1} AUC: {val_auc:.4f}")
        
        # Calculate weighted ensemble weights based on performance
        val_scores = np.array(val_scores)
        # Softmax weighting with temperature
        temperature = 5.0
        exp_scores = np.exp(val_scores * temperature)
        self.weights = exp_scores / np.sum(exp_scores)
        
        # Test ensemble
        ensemble_pred = self.predict_proba(X_val)
        ensemble_auc = roc_auc_score(y_val, ensemble_pred)
        
        print(f"🏆 Pure MLP Ensemble AUC: {ensemble_auc:.4f}")
        print(f"   Individual AUCs: {val_scores}")
        print(f"   Model weights: {self.weights}")
        
        return ensemble_auc
    
    def predict_proba(self, X):
        """Predição ensemble com weighted average"""
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return np.average(predictions, axis=0, weights=self.weights)
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

def pure_mlp_goat_pipeline(train_df, test_df=None, target_accuracy=0.97):
    """
    🐐 PURE MLP GOAT PIPELINE
    Pipeline completo usando apenas MLPs puros
    """
    print("🐐" * 20)
    print("    PURE MLP GOAT PIPELINE")
    print("    TARGET: 97%+ ACCURACY")
    print("    USING: Multi-Layer Perceptrons ONLY")
    print("🐐" * 20)
    
    # 1. Advanced preprocessing
    print("\n📊 Phase 1: Advanced Feature Engineering...")
    data_result = PureMLPFeatureEngineering.advanced_preprocessing(train_df, test_df)
    
    X = data_result['X_train']
    y = data_result['y_train']
    
    print(f"   Features engineered: {data_result['n_features']}")
    
    # 2. Stratified split
    print("\n🔀 Phase 2: Stratified Split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"   Train: {X_train.shape}, Validation: {X_val.shape}")
    print(f"   Class distribution: {np.bincount(y_train)} / {np.bincount(y_val)}")
    
    # 3. SMOTE balancing (optional)
    try:
        from imblearn.over_sampling import SMOTE
        print("\n⚖️ Phase 3: SMOTE Balancing...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE: {X_train_smote.shape}")
        print(f"   New distribution: {np.bincount(y_train_smote)}")
        X_train, y_train = X_train_smote, y_train_smote
    except ImportError:
        print("\n⚖️ Phase 3: Skipping SMOTE (install imbalanced-learn for better results)")
    
    # 4. Train Pure MLP Ensemble
    print("\n🎯 Phase 4: Training Pure MLP Ensemble...")
    ensemble = PureMLPEnsemble(n_models=5)
    ensemble_auc = ensemble.fit(X_train, y_train, X_val, y_val)
    
    # 5. Threshold optimization
    print("\n🎛️ Phase 5: Threshold Optimization...")
    val_probs = ensemble.predict_proba(X_val)
    
    # Find optimal threshold using F1-score
    from sklearn.metrics import precision_recall_curve, f1_score
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_threshold_idx]
    
    # 6. Final evaluation
    print("\n📈 Phase 6: Final Evaluation...")
    final_pred = (val_probs >= optimal_threshold).astype(int)
    final_accuracy = accuracy_score(y_val, final_pred)
    final_f1 = f1_score(y_val, final_pred)
    
    print(f"\n🏆 PURE MLP GOAT RESULTS:")
    print(f"   🎯 Validation AUC: {ensemble_auc:.4f}")
    print(f"   🎯 Validation Accuracy: {final_accuracy:.4f} ({final_accuracy:.1%})")
    print(f"   🎯 F1-Score: {final_f1:.4f}")
    print(f"   🎯 Optimal Threshold: {optimal_threshold:.4f}")
    
    if final_accuracy >= target_accuracy:
        print(f"   ✅ TARGET ACHIEVED! {final_accuracy:.1%} >= {target_accuracy:.1%}")
        print("   🐐 PURE MLP GOAT STATUS CONFIRMED!")
    else:
        print(f"   ⚠️ Close: {final_accuracy:.1%} < {target_accuracy:.1%}")
        print(f"   💪 Gap to target: {(target_accuracy - final_accuracy)*100:.1f} percentage points")
    
    # 7. Test predictions
    results = {
        'ensemble': ensemble,
        'threshold': optimal_threshold,
        'accuracy': final_accuracy,
        'auc': ensemble_auc,
        'f1': final_f1,
        'preprocessor': data_result['preprocessor']
    }
    
    if test_df is not None and 'X_test' in data_result:
        print("\n🔮 Phase 7: Test Predictions...")
        test_probs = ensemble.predict_proba(data_result['X_test'])
        
        submission = pd.DataFrame({
            'id': data_result['test_ids'],
            'y': test_probs
        })
        
        results['test_predictions'] = test_probs
        results['submission'] = submission
        print(f"   Test predictions ready: {len(test_probs)} samples")
    
    return results

def run_pure_mlp_goat(train_csv_path, test_csv_path=None):
    """
    🐐 ONE-CLICK PURE MLP GOAT
    
    Usage:
    results = run_pure_mlp_goat('data/train.csv', 'data/test.csv')
    """
    
    print("🐐 LOADING PURE MLP GOAT...")
    
    # Load data
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path) if test_csv_path else None
    
    print(f"📁 Loaded: train={train_df.shape}")
    if test_df is not None:
        print(f"          test={test_df.shape}")
    
    # Run Pure MLP GOAT pipeline
    results = pure_mlp_goat_pipeline(train_df, test_df)
    
    # Save submission
    if 'submission' in results:
        results['submission'].to_csv('pure_mlp_goat_submission.csv', index=False)
        print("💾 Submission saved as 'pure_mlp_goat_submission.csv'")
    
    print("\n🎉 PURE MLP GOAT EXECUTION COMPLETE!")
    return results

if __name__ == "__main__":
    # Example usage
    print("🐐 Pure MLP GOAT - Ready for 97%+ accuracy!")
    results = run_pure_mlp_goat('data/train.csv', 'data/test.csv')
    print(f"🎯 Final Accuracy: {results['accuracy']:.1%}")