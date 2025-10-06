import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

# ============================================================================
# BATCH NORMALIZATION
# ============================================================================
class BatchNorm1D:
    """Batch Normalization para camadas densas"""
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((dim, 1), dtype=np.float32)
        self.beta = np.zeros((dim, 1), dtype=np.float32)
        self.running_mean = np.zeros((dim, 1), dtype=np.float32)
        self.running_var = np.ones((dim, 1), dtype=np.float32)
        
    def forward(self, x, training=True):
        """
        x: shape (dim, batch_size)
        """
        if training:
            mean = x.mean(axis=1, keepdims=True)
            var = x.var(axis=1, keepdims=True)
            # Atualiza estatísticas correntes
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        
        # Cache para backward
        cache = {
            'x': x,
            'x_norm': x_norm,
            'mean': mean,
            'var': var,
            'std': np.sqrt(var + self.eps)
        }
        return out, cache
    
    def backward(self, dout, cache):
        """Backward pass para batch norm"""
        x = cache['x']
        x_norm = cache['x_norm']
        mean = cache['mean']
        std = cache['std']
        m = x.shape[1]
        
        # Gradientes dos parâmetros
        dgamma = np.sum(dout * x_norm, axis=1, keepdims=True)
        dbeta = np.sum(dout, axis=1, keepdims=True)
        
        # Gradiente da entrada
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * std**(-3), axis=1, keepdims=True)
        dmean = np.sum(dx_norm * -1/std, axis=1, keepdims=True) + dvar * np.sum(-2*(x - mean), axis=1, keepdims=True) / m
        dx = dx_norm / std + dvar * 2 * (x - mean) / m + dmean / m
        
        return dx, dgamma, dbeta


# ============================================================================
# ADAM OPTIMIZER
# ============================================================================
class AdamOptimizer:
    """Implementação do otimizador Adam"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # primeiro momento
        self.v = {}  # segundo momento
        self.t = 0   # timestep
    
    def step(self, params, grads):
        """Atualiza parâmetros usando Adam"""
        self.t += 1
        
        for key in params:
            grad_key = 'd' + key
            if grad_key not in grads:
                continue
                
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            g = grads[grad_key]
            
            # Atualiza momentos
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
            
            # Correção de viés
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Atualização
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================================
# MLP MELHORADO
# ============================================================================
class ImprovedMLP:
    """MLP com Adam, Dropout, Batch Normalization e Class Weighting"""
    
    def __init__(self, input_dim, hidden_sizes=[512, 256, 128, 64], 
                 lr=0.001, weight_decay=1e-5, dropout_rate=0.3,
                 use_batch_norm=True, pos_weight=7.0, seed=42):
        """
        Args:
            input_dim: dimensão de entrada
            hidden_sizes: lista com tamanho de cada camada oculta
            lr: learning rate para Adam
            weight_decay: L2 regularization
            dropout_rate: taxa de dropout (0 = sem dropout)
            use_batch_norm: usar batch normalization
            pos_weight: peso para classe positiva (class weighting)
            seed: random seed
        """
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)
        
        self.sizes = [input_dim] + hidden_sizes + [1]
        self.L = len(self.sizes) - 1
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.pos_weight = pos_weight
        self.weight_decay = weight_decay
        
        # Inicializa pesos e biases
        self.params = {}
        for i in range(self.L):
            in_dim = self.sizes[i]
            out_dim = self.sizes[i+1]
            
            # He initialization para ReLU
            std = np.sqrt(2.0 / in_dim) if i < self.L - 1 else np.sqrt(1.0 / in_dim)
            self.params[f"W{i+1}"] = self.rng.randn(out_dim, in_dim).astype(np.float32) * std
            self.params[f"b{i+1}"] = np.zeros((out_dim, 1), dtype=np.float32)
        
        # Batch Normalization layers
        self.bn_layers = {}
        if use_batch_norm:
            for i in range(1, self.L):  # não aplica na última camada
                self.bn_layers[f"bn{i}"] = BatchNorm1D(self.sizes[i])
        
        # Inicializa Adam optimizer
        self.optimizer = AdamOptimizer(lr=lr)
        
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(np.float32)
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid estável"""
        pos = x >= 0
        neg = ~pos
        out = np.empty_like(x, dtype=np.float64)
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg])
        out[neg] = ex / (1.0 + ex)
        return out
    
    def weighted_bce_loss(self, y_true, y_pred_prob, eps=1e-12):
        """Binary Cross-Entropy com class weighting"""
        y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
        loss = - (self.pos_weight * y_true * np.log(y_pred_prob) + 
                  (1 - y_true) * np.log(1 - y_pred_prob))
        return loss.mean()
    
    def forward(self, X, training=True):
        """
        Forward pass com dropout e batch normalization
        X: shape (batch, input_dim)
        training: se True, aplica dropout
        """
        cache = {}
        A = X.T  # shape (input_dim, batch)
        cache["A0"] = A
        
        for i in range(1, self.L + 1):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            Z = W.dot(A) + b
            cache[f"Z{i}"] = Z
            
            if i < self.L:  # camadas ocultas
                # Batch Normalization
                if self.use_batch_norm:
                    Z, bn_cache = self.bn_layers[f"bn{i}"].forward(Z, training)
                    cache[f"bn_cache{i}"] = bn_cache
                
                # ReLU
                A = self.relu(Z)
                
                # Dropout
                if training and self.dropout_rate > 0:
                    mask = (self.rng.rand(*A.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                    A = A * mask
                    cache[f"mask{i}"] = mask
            else:
                # Última camada (sigmoid aplicado depois)
                A = Z
            
            cache[f"A{i}"] = A
        
        logits = cache[f"A{self.L}"]
        probs = self.sigmoid(logits.ravel())
        cache["probs"] = probs
        
        return probs, cache
    
    def backward(self, cache, y_true):
        """Backward pass com dropout e batch normalization"""
        grads = {}
        m = y_true.shape[0]
        
        # Gradiente da saída
        probs = cache["probs"]
        dA = ((probs - y_true) * self.pos_weight * y_true + (probs - y_true) * (1 - y_true)) / m
        dA = dA.reshape(1, -1)
        
        for i in range(self.L, 0, -1):
            A_prev = cache[f"A{i-1}"]
            Z_i = cache[f"Z{i}"]
            W_i = self.params[f"W{i}"]
            
            # Gradientes de W e b
            dW = dA.dot(A_prev.T)
            db = dA.sum(axis=1, keepdims=True)
            
            # L2 regularization
            dW += self.weight_decay * W_i
            
            grads[f"dW{i}"] = dW
            grads[f"db{i}"] = db
            
            # Propaga para camada anterior
            if i > 1:
                dA_prev = W_i.T.dot(dA)
                
                # Dropout backward
                if f"mask{i-1}" in cache:
                    dA_prev = dA_prev * cache[f"mask{i-1}"]
                
                # ReLU backward
                dA_prev = dA_prev * self.relu_grad(cache[f"Z{i-1}"])
                
                # Batch Normalization backward
                if self.use_batch_norm:
                    bn_cache = cache[f"bn_cache{i-1}"]
                    dA_prev, dgamma, dbeta = self.bn_layers[f"bn{i-1}"].backward(dA_prev, bn_cache)
                    grads[f"dgamma{i-1}"] = dgamma
                    grads[f"dbeta{i-1}"] = dbeta
                
                dA = dA_prev
        
        return grads
    
    def predict_proba(self, X):
        """Predição de probabilidades (sem dropout)"""
        probs, _ = self.forward(X, training=False)
        return probs
    
    def predict(self, X, threshold=0.5):
        """Predição de classes"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int64)


# ============================================================================
# TREINAMENTO COM EARLY STOPPING E LR SCHEDULING
# ============================================================================
def train_improved_mlp(model, X_train, y_train, X_val, y_val,
                       n_epochs=50, batch_size=512, 
                       early_stopping_patience=10,
                       lr_decay_rate=0.95, lr_decay_epochs=5,
                       verbose=True):
    """
    Treina o MLP melhorado com early stopping e learning rate decay
    """
    n_samples = X_train.shape[0]
    history = {
        "train_loss": [], "val_loss": [], 
        "val_auc": [], "val_acc": [],
        "lr": []
    }
    
    best_auc = -np.inf
    best_params = None
    patience_counter = 0
    initial_lr = model.optimizer.lr
    
    for epoch in range(1, n_epochs + 1):
        # Learning rate decay
        if epoch % lr_decay_epochs == 0:
            model.optimizer.lr *= lr_decay_rate
        
        history["lr"].append(model.optimizer.lr)
        
        # Shuffle
        perm = np.random.permutation(n_samples)
        X_shuff = X_train[perm]
        y_shuff = y_train[perm]
        
        # Mini-batches
        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            xb = X_shuff[start:start+batch_size]
            yb = y_shuff[start:start+batch_size]
            
            # Forward
            probs, cache = model.forward(xb, training=True)
            loss = model.weighted_bce_loss(yb, probs)
            epoch_losses.append(loss)
            
            # Backward
            grads = model.backward(cache, yb)
            
            # Update (Adam)
            model.optimizer.step(model.params, grads)
            
            # Update batch norm parameters
            if model.use_batch_norm:
                for i in range(1, model.L):
                    bn = model.bn_layers[f"bn{i}"]
                    if f"dgamma{i}" in grads:
                        bn.gamma -= model.optimizer.lr * grads[f"dgamma{i}"]
                        bn.beta -= model.optimizer.lr * grads[f"dbeta{i}"]
        
        # Métricas de época
        train_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(train_loss)
        
        # Validação
        val_probs = model.predict_proba(X_val)
        val_loss = float(model.weighted_bce_loss(y_val, val_probs))
        val_preds = (val_probs >= 0.5).astype(np.int64)
        val_auc = roc_auc_score(y_val, val_probs)
        val_acc = accuracy_score(y_val, val_preds)
        
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{n_epochs} — "
                  f"train_loss: {train_loss:.4f} — "
                  f"val_loss: {val_loss:.4f} — "
                  f"val_auc: {val_auc:.4f} — "
                  f"val_acc: {val_acc:.4f} — "
                  f"lr: {model.optimizer.lr:.6f}")
        
        # Early stopping e checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Salva os melhores parâmetros
            best_params = {k: v.copy() for k, v in model.params.items()}
            if verbose:
                print(f"    ✓ Novo melhor AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping na época {epoch}")
                break
    
    # Restaura melhores parâmetros
    if best_params is not None:
        model.params = best_params
    
    print(f"\n{'='*60}")
    print(f"Treinamento completo!")
    print(f"Melhor validação AUC: {best_auc:.4f}")
    print(f"{'='*60}\n")
    
    return history


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    """Adiciona features derivadas"""
    df = df.copy()
    
    # Features de interação
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['campaign_per_previous'] = df['campaign'] / (df['previous'] + 1)
    df['duration_campaign'] = df['duration'] * df['campaign']
    
    # Features binárias
    df['was_contacted_before'] = (df['pdays'] != 999).astype(int)
    df['has_default'] = (df['default'] == 'yes').astype(int)
    df['has_housing'] = (df['housing'] == 'yes').astype(int)
    df['has_loan'] = (df['loan'] == 'yes').astype(int)
    df['success_prev'] = (df['poutcome'] == 'success').astype(int)
    
    # Log transforms para features skewed
    df['log_balance'] = np.log1p(np.maximum(df['balance'], 0))
    df['log_duration'] = np.log1p(df['duration'])
    
    return df


# ============================================================================
# ENSEMBLE TRAINING
# ============================================================================
def train_ensemble(X_train, y_train, X_val, y_val, n_models=5, **model_kwargs):
    """
    Treina múltiplos modelos com diferentes seeds para ensemble
    """
    models = []
    seeds = [42, 123, 456, 789, 1011][:n_models]
    
    print(f"\n{'='*70}")
    print(f"🔥 TREINAMENTO DE ENSEMBLE")
    print(f"{'='*70}")
    print(f"Número de modelos: {n_models}")
    print(f"Seeds: {seeds}")
    print(f"Arquitetura: {model_kwargs.get('hidden_sizes', 'N/A')}")
    print(f"{'='*70}\n")
    
    model_results = []
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'🤖 '*35}")
        print(f"{'🤖 MODELO {}/{} | SEED: {}'.format(i, n_models, seed):^70}")
        print(f"{'🤖 '*35}")
        
        model = ImprovedMLP(seed=seed, **model_kwargs)
        
        import time
        start_time = time.time()
        
        history = train_improved_mlp(
            model, X_train, y_train, X_val, y_val,
            n_epochs=50,
            batch_size=512,
            early_stopping_patience=10,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        
        val_auc = max(history['val_auc'])
        val_acc = max(history['val_acc'])
        
        model_results.append({
            'model': i,
            'seed': seed,
            'auc': val_auc,
            'acc': val_acc,
            'epochs': len(history['val_auc']),
            'time': elapsed
        })
        
        print(f"\n{'─'*70}")
        print(f"✓ Modelo {i} concluído!")
        print(f"  • Melhor AUC: {val_auc:.4f}")
        print(f"  • Melhor Accuracy: {val_acc:.4f}")
        print(f"  • Épocas treinadas: {len(history['val_auc'])}")
        print(f"  • Tempo: {elapsed/60:.1f} minutos")
        print(f"{'─'*70}\n")
        
        models.append(model)
    
    # Avalia ensemble
    print(f"\n{'='*70}")
    print(f"📊 AVALIAÇÃO DO ENSEMBLE")
    print(f"{'='*70}\n")
    
    print("Resultados individuais:")
    print(f"{'─'*70}")
    for result in model_results:
        print(f"Modelo {result['model']} (seed {result['seed']:4d}): "
              f"AUC={result['auc']:.4f} | ACC={result['acc']:.4f} | "
              f"Épocas={result['epochs']:2d} | Tempo={result['time']/60:.1f}min")
    print(f"{'─'*70}\n")
    
    # Estatísticas dos modelos individuais
    individual_aucs = [r['auc'] for r in model_results]
    individual_accs = [r['acc'] for r in model_results]
    
    print("Estatísticas dos modelos individuais:")
    print(f"  AUC  → Média: {np.mean(individual_aucs):.4f} | "
          f"Min: {np.min(individual_aucs):.4f} | "
          f"Max: {np.max(individual_aucs):.4f} | "
          f"Std: {np.std(individual_aucs):.4f}")
    print(f"  ACC  → Média: {np.mean(individual_accs):.4f} | "
          f"Min: {np.min(individual_accs):.4f} | "
          f"Max: {np.max(individual_accs):.4f} | "
          f"Std: {np.std(individual_accs):.4f}")
    
    print(f"\n{'─'*70}")
    print("Calculando predições do ensemble...")
    
    ensemble_probs = np.mean([m.predict_proba(X_val) for m in models], axis=0)
    ensemble_auc = roc_auc_score(y_val, ensemble_probs)
    ensemble_acc = accuracy_score(y_val, (ensemble_probs >= 0.5).astype(int))
    
    print(f"{'─'*70}\n")
    print(f"🏆 RESULTADO FINAL DO ENSEMBLE:")
    print(f"{'─'*70}")
    print(f"  AUC:      {ensemble_auc:.4f}  (ganho: +{ensemble_auc - np.mean(individual_aucs):.4f})")
    print(f"  Accuracy: {ensemble_acc:.4f}  (ganho: +{ensemble_acc - np.mean(individual_accs):.4f})")
    print(f"{'─'*70}")
    
    if ensemble_auc > np.max(individual_aucs):
        print(f"✨ Ensemble SUPEROU o melhor modelo individual!")
        print(f"   Melhor individual: {np.max(individual_aucs):.4f}")
        print(f"   Ensemble: {ensemble_auc:.4f}")
        print(f"   Ganho: +{ensemble_auc - np.max(individual_aucs):.4f}")
    else:
        print(f"ℹ️  Melhor modelo individual teve AUC ligeiramente superior")
    
    print(f"{'='*70}\n")
    
    return models


def predict_ensemble(models, X):
    """Predição usando ensemble (média das probabilidades)"""
    predictions = np.array([model.predict_proba(X) for model in models])
    return np.mean(predictions, axis=0)


# ============================================================================
# GERAÇÃO DE SUBMISSION
# ============================================================================
def create_submission(models, preprocessor, test_path, sample_sub_path, 
                     output_path="submission.csv", use_ensemble=True):
    """
    Cria arquivo de submission para Kaggle
    
    Args:
        models: modelo único ou lista de modelos (para ensemble)
        preprocessor: pipeline de preprocessing fitado
        test_path: caminho do test.csv
        sample_sub_path: caminho do sample_submission.csv
        output_path: onde salvar submission
        use_ensemble: se True e models é lista, faz média das predições
    """
    print(f"\n{'='*70}")
    print(f"📝 GERANDO SUBMISSION PARA KAGGLE")
    print(f"{'='*70}\n")
    
    # Carrega dados
    print("1/6 Carregando dados de teste...")
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)
    
    print(f"    ✓ Test shape: {test.shape}")
    print(f"    ✓ Sample submission shape: {sample_sub.shape}")
    
    # Feature engineering no test
    print("\n2/6 Aplicando feature engineering...")
    test_eng = engineer_features(test)
    print(f"    ✓ Features criadas: {len(test_eng.columns)} colunas totais")
    
    # Remove colunas desnecessárias
    print("\n3/6 Preparando features para preprocessamento...")
    X_test = test_eng.drop(columns=['id'], errors='ignore')
    print(f"    ✓ Features de entrada: {X_test.shape[1]}")
    
    # Preprocessamento
    print("\n4/6 Aplicando preprocessamento (scaling + encoding)...")
    X_test_processed = preprocessor.transform(X_test)
    print(f"    ✓ Features após preprocessamento: {X_test_processed.shape[1]}")
    print(f"    ✓ Tipo: {type(X_test_processed)}, Shape: {X_test_processed.shape}")
    
    # Predição
    print("\n5/6 Gerando predições...")
    if isinstance(models, list) and use_ensemble:
        print(f"    🔮 Usando ENSEMBLE de {len(models)} modelos...")
        
        all_predictions = []
        for i, model in enumerate(models, 1):
            print(f"       • Modelo {i}/{len(models)}...", end=' ', flush=True)
            preds = model.predict_proba(X_test_processed)
            all_predictions.append(preds)
            print(f"✓ (média: {preds.mean():.4f})")
        
        predictions = np.mean(all_predictions, axis=0)
        print(f"\n    ✓ Predições do ensemble calculadas!")
        print(f"       • Média das predições: {predictions.mean():.4f}")
        print(f"       • Desvio padrão entre modelos: {np.std([p.mean() for p in all_predictions]):.4f}")
        
    else:
        if isinstance(models, list):
            model = models[0]
            print(f"    🔮 Usando MODELO ÚNICO (primeiro da lista)...")
        else:
            model = models
            print(f"    🔮 Usando MODELO ÚNICO...")
        
        predictions = model.predict_proba(X_test_processed)
        print(f"    ✓ Predições calculadas!")
    
    # Cria DataFrame de submission
    print("\n6/6 Criando arquivo de submission...")
    
    # Identifica coluna target no sample submission
    target_col = [c for c in sample_sub.columns if c.lower() not in ['id']][0]
    print(f"    • Coluna target identificada: '{target_col}'")
    
    submission = pd.DataFrame()
    submission['id'] = test['id'].values if 'id' in test.columns else test['Id'].values
    submission[target_col] = predictions
    
    # Garante mesma ordem de colunas
    submission = submission[sample_sub.columns]
    
    # Salva
    submission.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ SUBMISSION CRIADA COM SUCESSO!")
    print(f"{'='*70}")
    print(f"📁 Arquivo: {output_path}")
    print(f"📊 Estatísticas:")
    print(f"    • Total de linhas: {len(submission):,}")
    print(f"    • Colunas: {list(submission.columns)}")
    print(f"    • Predição média: {predictions.mean():.6f}")
    print(f"    • Predição min: {predictions.min():.6f}")
    print(f"    • Predição max: {predictions.max():.6f}")
    print(f"    • Predição mediana: {np.median(predictions):.6f}")
    print(f"    • Std: {predictions.std():.6f}")
    
    # Distribuição das predições
    print(f"\n📈 Distribuição das predições:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"    {'Percentil':<12} {'Valor':>10}")
    print(f"    {'-'*12} {'-'*10}")
    for p in percentiles:
        val = np.percentile(predictions, p)
        print(f"    {str(p)+'%':<12} {val:>10.6f}")
    
    print(f"\n🎯 Próximo passo: Faça upload de '{output_path}' no Kaggle!")
    print(f"{'='*70}\n")
    
    # Preview
    print("Preview da submission (primeiras 10 linhas):")
    print(submission.head(10).to_string(index=False))
    print(f"\n... e mais {len(submission)-10:,} linhas")
    
    return submission


# ============================================================================
# EXEMPLO DE USO COMPLETO
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("MLP MELHORADO - PIPELINE COMPLETO")
    print("="*60)
    
    # Configuração
    DATA_DIR = Path("data")
    USE_ENSEMBLE = True  # True para ensemble, False para modelo único
    N_MODELS = 5  # número de modelos no ensemble
    
    # ========================================================================
    # PARTE 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS
    # ========================================================================
    print("\n[PARTE 1/4] Carregando e preparando dados...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    
    # Feature Engineering
    print("Aplicando feature engineering...")
    train_eng = engineer_features(train)
    
    # Definir features
    numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 
                       'pdays', 'previous', 'balance_per_age', 
                       'campaign_per_previous', 'duration_campaign',
                       'was_contacted_before', 'log_balance', 'log_duration']
    
    categorical_features = ['job', 'marital', 'education', 'contact', 
                          'month', 'poutcome']
    
    # Winsorização
    print("Aplicando winsorização...")
    skewed = ['balance', 'duration', 'campaign', 'pdays', 'previous']
    for col in skewed:
        if col in train_eng.columns:
            upper = train_eng[col].quantile(0.99)
            train_eng[col] = np.where(train_eng[col] > upper, upper, train_eng[col])
    
    X = train_eng.drop(columns=['id', 'y'])
    y = train_eng['y'].values
    
    # Preprocessing pipeline
    print("Criando pipeline de preprocessamento...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.1, random_state=42, stratify=y
    )
    
    print(f"\n✓ Dados preparados:")
    print(f"  - Train: {X_train.shape}")
    print(f"  - Validation: {X_val.shape}")
    print(f"  - Features totais: {X_processed.shape[1]}")
    
    # Calcula peso da classe positiva
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"  - Peso classe positiva: {pos_weight:.2f}")
    
    # ========================================================================
    # PARTE 2: TREINAMENTO
    # ========================================================================
    print(f"\n[PARTE 2/4] Treinamento do modelo...")
    
    model_config = {
        'input_dim': X_train.shape[1],
        'hidden_sizes': [512, 256, 128, 64],
        'lr': 0.001,
        'weight_decay': 1e-5,
        'dropout_rate': 0.3,
        'use_batch_norm': True,
        'pos_weight': pos_weight
    }
    
    if USE_ENSEMBLE:
        models = train_ensemble(
            X_train, y_train, X_val, y_val,
            n_models=N_MODELS,
            **model_config
        )
    else:
        print("Treinando modelo único...")
        model = ImprovedMLP(seed=42, **model_config)
        history = train_improved_mlp(
            model, X_train, y_train, X_val, y_val,
            n_epochs=50,
            batch_size=512,
            early_stopping_patience=10,
            verbose=True
        )
        models = model
        
        # Plot curvas de treinamento
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(history['val_auc'], label='AUC')
        axes[1].plot(history['val_acc'], label='Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('LR Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("\n✓ Curvas de treinamento salvas em: training_curves.png")
        plt.show()
    
    # ========================================================================
    # PARTE 3: AVALIAÇÃO FINAL
    # ========================================================================
    print(f"\n[PARTE 3/4] Avaliação final no conjunto de validação...")
    
    if isinstance(models, list):
        val_probs = predict_ensemble(models, X_val)
    else:
        val_probs = models.predict_proba(X_val)
    
    val_preds = (val_probs >= 0.5).astype(int)
    
    print(f"\n✓ Métricas Finais:")
    print(f"  - AUC: {roc_auc_score(y_val, val_probs):.4f}")
    print(f"  - Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    
    from sklearn.metrics import classification_report
    print("\n" + classification_report(y_val, val_preds, 
                                      target_names=['Não assinou', 'Assinou']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não', 'Sim'], 
                yticklabels=['Não', 'Sim'])
    plt.title('Matriz de Confusão - Validação')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n✓ Matriz de confusão salva em: confusion_matrix.png")
    plt.show()
    
    # ========================================================================
    # PARTE 4: GERAÇÃO DA SUBMISSION
    # ========================================================================
    print(f"\n[PARTE 4/4] Gerando submission para Kaggle...")
    
    submission = create_submission(
        models=models,
        preprocessor=preprocessor,
        test_path=DATA_DIR / "test.csv",
        sample_sub_path=DATA_DIR / "sample_submission.csv",
        output_path="submission.csv",
        use_ensemble=USE_ENSEMBLE
    )
    
    print(f"\n{'='*60}")
    print("✓ PIPELINE COMPLETO!")
    print(f"{'='*60}")
    print("\nArquivos gerados:")
    print("  1. submission.csv - arquivo para submeter no Kaggle")
    print("  2. training_curves.png - curvas de treinamento")
    print("  3. confusion_matrix.png - matriz de confusão")
    print("\nPróximos passos:")
    print("  1. Faça upload de submission.csv no Kaggle")
    print("  2. Verifique o score público")
    print("  3. Se necessário, ajuste hiperparâmetros e retreine")
    print(f"{'='*60}\n")