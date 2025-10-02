import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# ============================================
# CONFIGURAÇÕES
# ============================================
def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device

# ============================================
# MODELO
# ============================================
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# ============================================
# TREINAMENTO RÁPIDO
# ============================================
def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print("TREINAMENTO")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        start = time.time()
        
        # TREINO
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Mais rápido
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch.unsqueeze(1)).sum().item()
            total += y_batch.size(0)
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # VALIDAÇÃO
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                
                val_loss += loss.item()
                preds = (outputs >= 0.5).float()
                correct += (preds == y_batch.unsqueeze(1)).sum().item()
                total += y_batch.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        elapsed = time.time() - start
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            marker = "⭐"
        else:
            patience_counter += 1
            marker = "  "
        
        # Log a cada época
        print(f"{marker} Epoch {epoch+1:3d}/{epochs} | "
              f"Time: {elapsed:4.1f}s | "
              f"Train: {train_loss:.4f} ({train_acc*100:5.2f}%) | "
              f"Val: {val_loss:.4f} ({val_acc*100:5.2f}%)")
        
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load('best_model.pth'))
    print(f"\n✅ Best model loaded (val_loss: {best_val_loss:.4f})")
    
    return train_losses, val_losses, train_accs, val_accs

# ============================================
# MAIN
# ============================================
def main():
    print("\n" + "="*60)
    print("  MLP - BANK MARKETING PREDICTION")
    print("="*60)
    
    # Setup
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    device = setup_device()
    
    # CARREGAR DADOS
    print("\n📂 Loading data...", end=' ', flush=True)
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"✓ ({train_df.shape[0]:,} train, {test_df.shape[0]:,} test)")
    
    # PRÉ-PROCESSAMENTO
    print("🔧 Preprocessing...", end=' ', flush=True)
    X_train = train_df.drop(['id', 'y'], axis=1)
    y_train = train_df['y'].values
    X_test = test_df.drop(['id'], axis=1)
    
    # Encode categorical
    for col in X_train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ (16 features)")
    
    # SPLIT
    print("📊 Creating splits...", end=' ', flush=True)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"✓ ({X_train_split.shape[0]:,} train, {X_val.shape[0]:,} val)")
    
    # DATALOADERS
    print("📦 Creating dataloaders...", end=' ', flush=True)
    batch_size = 1024  # Aumentado para melhor uso da GPU
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.FloatTensor(y_train_split)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"✓ (batch_size={batch_size})")
    
    # MODELO
    print("🏗️  Building model...", end=' ', flush=True)
    model = MLP(input_size=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ ({n_params:,} parameters)")
    
    # TREINAR
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device,
        epochs=100,
        lr=0.001
    )
    
    # AVALIAR
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}\n")
    
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        val_outputs = model(X_val_tensor).cpu().numpy().flatten()
        y_val_pred = (val_outputs >= 0.5).astype(int)
    
    acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_val, y_val_pred, target_names=['No', 'Yes']))
    
    # VISUALIZAR
    print(f"\n{'='*60}")
    print("SAVING PLOTS")
    print(f"{'='*60}\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(train_losses, label='Train', linewidth=2, alpha=0.8)
    axes[0].plot(val_losses, label='Validation', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(train_accs, label='Train', linewidth=2, alpha=0.8)
    axes[1].plot(val_accs, label='Validation', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: training_curves.png")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], 
                yticklabels=['No', 'Yes'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: confusion_matrix.png")
    
    # PREDIÇÕES
    print(f"\n{'='*60}")
    print("GENERATING PREDICTIONS")
    print(f"{'='*60}\n")
    
    print("🔮 Predicting on test set...", end=' ', flush=True)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).cpu().numpy().flatten()
    print("✓")
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'y': test_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("✓ Saved: submission.csv")
    
    print(f"\nPrediction stats:")
    print(f"  Min:  {test_predictions.min():.4f}")
    print(f"  Max:  {test_predictions.max():.4f}")
    print(f"  Mean: {test_predictions.mean():.4f}")
    
    print(f"\n{'='*60}")
    print("  ✅ DONE!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()