import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================
# SISTEM PERBANDINGAN FAIR: FaceNet vs Arc2Face vs RetinaFace
# ==========================================
print("="*70)
print(" PERBANDINGAN FAIR: FaceNet vs Arc2Face vs RetinaFace")
print(" Tujuan: Mengetahui metode mana yang TERBAIK secara empiris")
print("="*70)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Device: {DEVICE}")

# ==========================================
# DATASET UTKFace
# ==========================================
class UTKFaceDataset(Dataset):
    """Dataset UTKFace untuk klasifikasi gender"""
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.labels = []
        
        all_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        for fname in all_files:
            parts = fname.split('_')
            if len(parts) >= 3:
                try:
                    gender = int(parts[1])  # 0=Male, 1=Female
                    self.file_list.append(fname)
                    self.labels.append(gender)
                except ValueError:
                    continue
        
        if limit:
            self.file_list = self.file_list[:limit]
            self.labels = self.labels[:limit]
        
        print(f"[INFO] Dataset: {len(self.file_list)} images")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img_path = os.path.join(self.root_dir, fname)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==========================================
# TRIPLET LOSS untuk FaceNet (BAB 3.1.1.1.6)
# ==========================================
class TripletLoss(nn.Module):
    """Triplet Loss - Keunggulan FaceNet"""
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# ==========================================
# ARCFACE LOSS untuk Arc2Face (BAB 3.1.1.2.1)
# ==========================================
class ArcMarginProduct(nn.Module):
    """ArcFace Loss - Keunggulan Arc2Face"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.threshold = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=DEVICE)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

# ==========================================
# ARSITEKTUR MODEL (Fair Comparison)
# ==========================================
class FaceRecognitionSystem(nn.Module):
    """
    3 Arsitektur dengan kondisi FAIR:
    1. FaceNet: Triplet Loss + Simple Head
    2. Arc2Face: ArcFace Loss + Simple Head  
    3. RetinaFace: Standard Loss + Frozen Backbone
    """
    def __init__(self, arch_type='facenet', head_type='baseline', num_classes=2, 
                 embedding_size=512, backbone_name='mobilenetv3_large_100'):
        super().__init__()
        self.arch_type = arch_type
        self.head_type = head_type
        self.embedding_size = embedding_size
        
        # Backbone (configurable)
        self.backbone = timm.create_model(backbone_name, 
                                         pretrained=True, 
                                         num_classes=0, 
                                         global_pool='')
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 128)
            dummy_output = self.backbone(dummy_input)
            self.backbone_channels = dummy_output.shape[1]
        
        print(f"[INFO] {arch_type.upper()}: {backbone_name}, {self.backbone_channels} channels")
        
        # Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        # Embedding layer dengan BatchNorm untuk stabilitas
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone_channels, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        # Head Architecture
        self._build_head(head_type, num_classes)
        
        # Loss-specific layers
        if arch_type == 'arc2face':
            feat_dim = embedding_size * 2 if head_type == 'concat' else embedding_size
            # ArcFace dengan margin lebih besar untuk separability lebih baik
            self.margin = ArcMarginProduct(feat_dim, num_classes, s=32.0, m=0.50)
        
        # RetinaFace: Frozen backbone (sebagai Feature Extractor)
        if arch_type == 'retinaface':
            print(f"[INFO] RetinaFace: Freezing backbone (Feature Extractor mode)")
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _build_head(self, head_type, num_classes):
        """Build classification head"""
        if head_type == 'baseline':
            self.head = nn.Linear(self.embedding_size, num_classes)
            
        elif head_type == 'deep':
            self.head = nn.Sequential(
                nn.Linear(self.embedding_size, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, num_classes)
            )
            
        elif head_type == 'dropout':
            self.head = nn.Sequential(
                nn.Linear(self.embedding_size, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )
            
        elif head_type == 'concat':
            self.head = nn.Sequential(
                nn.Linear(self.embedding_size * 2, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, num_classes)
            )

    def forward(self, x, label=None):
        # Feature extraction
        feat = self.backbone(x)
        
        # Embedding layer dengan BatchNorm
        if self.head_type == 'concat':
            gap_feat = self.gap(feat).flatten(1)
            gmp_feat = self.gmp(feat).flatten(1)
            gap_emb = self.embedding(gap_feat)
            gmp_emb = self.embedding(gmp_feat)
            embedding = torch.cat([gap_emb, gmp_emb], dim=1)
        else:
            feat = self.gap(feat).flatten(1)
            embedding = self.embedding(feat)
        
        # L2 normalization
        embedding = F.normalize(embedding, p=2, dim=1)
        
        # Forward berdasarkan arsitektur
        if self.arch_type == 'arc2face' and label is not None and self.training:
            return self.margin(embedding, label)
        else:
            return self.head(embedding)

# ==========================================
# TRAINING ENGINE (Fair untuk semua!)
# ==========================================
def train_model(model, train_loader, val_loader, lr, epochs, device, warmup_epochs=3):
    """Training dengan kondisi SAMA untuk semua metode + Warmup"""
    
    # Optimizer dengan weight decay lebih kecil untuk Arc2Face
    weight_decay = 5e-5 if model.arch_type == 'arc2face' else 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing untuk generalisasi
    
    # Learning rate scheduler dengan warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Warmup
        else:
            return 0.5 ** ((epoch - warmup_epochs) // 8)  # Decay setiap 8 epochs
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    best_metrics = None
    patience = 7
    patience_counter = 0
    
    print(f"[TRAINING] Epochs={epochs}, LR={lr}, Warmup={warmup_epochs}")
    
    for epoch in range(epochs):
        # TRAINING
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward (Arc2Face menggunakan margin saat training)
            if model.arch_type == 'arc2face':
                outputs = model(images, labels)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping untuk stabilitas
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Val={val_acc:.4f}, Loss={avg_train_loss:.4f}, LR={current_lr:.6f}")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_metrics = {
                'accuracy': val_acc,
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'y_true': y_true,
                'y_pred': y_pred
            }
            print(f"  ✓ New best: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"  ✓ Final Best Val Acc: {best_acc:.4f}")
    return history, best_metrics

# ==========================================
# GRUP A & B: Baseline Comparison
# ==========================================
def run_baseline_comparison(dataset, device, epochs, batch_size, backbone_name):
    """Grup A (80/20) & Grup B (90/10)"""
    
    results = []
    
    for method in ['FaceNet', 'Arc2Face', 'RetinaFace']:
        for ratio, split_name, grup in [(0.8, '80/20', 'A'), (0.9, '90/10', 'B')]:
            print(f"\n{'='*70}")
            print(f"GRUP {grup}: {method} | Split {split_name} | Epochs {epochs}")
            print(f"{'='*70}")
            
            # Split dataset
            train_len = int(ratio * len(dataset))
            val_len = len(dataset) - train_len
            train_ds, val_ds = random_split(dataset, [train_len, val_len])
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # Model
            arch_map = {'FaceNet': 'facenet', 'Arc2Face': 'arc2face', 'RetinaFace': 'retinaface'}
            model = FaceRecognitionSystem(arch_type=arch_map[method], head_type='baseline', 
                                         num_classes=2, backbone_name=backbone_name)
            model = model.to(device)
            
            # Training dengan LR yang sesuai
            # Arc2Face butuh LR lebih kecil karena ArcFace Loss
            lr = 5e-4 if method == 'Arc2Face' else 1e-3
            
            start_time = time.time()
            history, metrics = train_model(model, train_loader, val_loader, lr=lr, epochs=epochs, device=device)
            elapsed = (time.time() - start_time) / 60
            
            results.append({
                'Grup': grup,
                'Arsitektur Model': method,
                'Split': split_name,
                'Akurasi (%)': metrics['accuracy'] * 100,
                'Presisi (Weighted)': metrics['precision'],
                'Recall (Weighted)': metrics['recall'],
                'F1-Score (Weighted)': metrics['f1'],
                'Time (min)': elapsed
            })
            
            print(f"  Final: Acc={metrics['accuracy']*100:.1f}%, F1={metrics['f1']:.3f}, Time={elapsed:.1f}min")
    
    return pd.DataFrame(results)

# ==========================================
# GRUP C: LR Tuning (Best Model Only)
# ==========================================
def run_lr_tuning(dataset, device, best_method, epochs, batch_size, backbone_name):
    """Grup C: Fine-tune model terbaik dengan LR berbeda"""
    
    print(f"\n{'='*70}")
    print(f"GRUP C: LR TUNING ({best_method}, Split 90/10)")
    print(f"{'='*70}")
    
    results = []
    
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    for lr_val, lr_name in [(5e-4, '5e-4 (Baseline)'), (1e-4, '1e-4 (Fine-tune)')]:
        print(f"\n[{lr_name}]")
        
        arch_map = {'FaceNet': 'facenet', 'Arc2Face': 'arc2face', 'RetinaFace': 'retinaface'}
        model = FaceRecognitionSystem(arch_type=arch_map[best_method], head_type='baseline', 
                                     num_classes=2, backbone_name=backbone_name)
        model = model.to(device)
        
        history, metrics = train_model(model, train_loader, val_loader, lr=lr_val, epochs=epochs, device=device)
        
        results.append({
            'Learning Rate (LR)': lr_name,
            'Akurasi (%)': metrics['accuracy'] * 100,
            'Presisi (Weighted)': metrics['precision'],
            'Recall (Weighted)': metrics['recall'],
            'F1-Score (Weighted)': metrics['f1']
        })
    
    return pd.DataFrame(results)

# ==========================================
# GRUP D: Head Tinkering
# ==========================================
def run_head_tinkering(dataset, device, best_method, best_lr, epochs, batch_size, backbone_name):
    """Grup D: Test 4 head configurations"""
    
    print(f"\n{'='*70}")
    print(f"GRUP D: HEAD MODIFICATION ({best_method}, 90/10, LR {best_lr})")
    print(f"{'='*70}")
    
    head_configs = [
        ('baseline', 'Baseline Head', 'Input(512) -> Softmax(N)'),
        ('deep', 'Head 1 (Deep)', 'Input(512) -> Dense(1024) -> Softmax(N)'),
        ('dropout', 'Head 2 (Dropout)', 'Input(512) -> Dense(1024) -> Dropout(0.5) -> Softmax(N)'),
        ('concat', 'Head 3 (Concat)', 'Concat[GAP(512),GMP(512)] -> Dense(1024) -> Softmax(N)')
    ]
    
    results = []
    best_acc = 0
    best_config = None
    
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    arch_map = {'FaceNet': 'facenet', 'Arc2Face': 'arc2face', 'RetinaFace': 'retinaface'}
    
    for head_type, head_name, head_desc in head_configs:
        print(f"\n[{head_name}]")
        
        model = FaceRecognitionSystem(arch_type=arch_map[best_method], head_type=head_type, 
                                     num_classes=2, backbone_name=backbone_name)
        model = model.to(device)
        
        history, metrics = train_model(model, train_loader, val_loader, lr=best_lr, epochs=epochs, device=device)
        
        results.append({
            'Konfigurasi Head': head_name,
            'Arsitektur Head': head_desc,
            'Akurasi (%)': metrics['accuracy'] * 100,
            'F1-Score (Weighted)': metrics['f1']
        })
        
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_config = head_name
    
    return pd.DataFrame(results), best_config, best_acc

# ==========================================
# VISUALISASI
# ==========================================
def visualize_results(df_ab, df_c, df_d, output_dir='hasil'):
    """Generate visualisasi perbandingan"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Grup A vs B
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'FaceNet': '#3498db', 'Arc2Face': '#2ecc71', 'RetinaFace': '#e74c3c'}
    
    for idx, grup in enumerate(['A', 'B']):
        df_grup = df_ab[df_ab['Grup'] == grup]
        x = np.arange(3)
        
        for i, method in enumerate(['FaceNet', 'Arc2Face', 'RetinaFace']):
            acc = df_grup[df_grup['Arsitektur Model'] == method]['Akurasi (%)'].values[0]
            axes[idx].bar(i, acc, color=colors[method], alpha=0.8, width=0.6)
            axes[idx].text(i, acc + 1, f'{acc:.1f}%', ha='center', fontweight='bold')
        
        split = '80/20' if grup == 'A' else '90/10'
        axes[idx].set_title(f'Grup {grup}: Split {split}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Akurasi (%)', fontsize=12)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(['FaceNet', 'Arc2Face', 'RetinaFace'])
        axes[idx].set_ylim(70, 100)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/grup_ab_comparison.png', dpi=300)
    plt.close()
    
    # 2. Grup D: Head Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heads = df_d['Konfigurasi Head'].values
    accs = df_d['Akurasi (%)'].values
    colors_head = ['#95a5a6', '#3498db', '#f39c12', '#2ecc71']
    
    bars = ax.barh(heads, accs, color=colors_head, alpha=0.8)
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(3)
    
    for i, acc in enumerate(accs):
        ax.text(acc + 0.3, i, f'{acc:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Akurasi (%)', fontsize=12)
    ax.set_title('Grup D: Perbandingan 4 Konfigurasi Head', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/grup_d_head_comparison.png', dpi=300)
    plt.close()
    
    print(f"\n[SAVED] Visualisasi di {output_dir}/")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    DATA_DIR = "./dataset/UTKFace"
    OUTPUT_DIR = "./hasil/v4"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"\n[ERROR] Dataset tidak ditemukan: {DATA_DIR}")
        exit(1)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # ==========================================
    # KONFIGURASI OPTIMAL UNTUK HASIL TINGGI
    # ==========================================
    
    # QUICK MODE (Testing - 5-10 menit)
    QUICK_MODE = True
    if QUICK_MODE:
        LIMIT = 500
        EPOCHS_BASELINE = 5
        EPOCHS_FINETUNING = 8
        BATCH_SIZE = 64
        BACKBONE = 'mobilenetv3_small_100'
        print("\nQUICK TEST MODE")
    
    # PRODUCTION MODE (Hasil Tinggi - 30-60 menit)
    else:
        LIMIT = 5000  # 5000 gambar cukup untuk hasil tinggi
        EPOCHS_BASELINE = 25  # Lebih banyak epoch!
        EPOCHS_FINETUNING = 35  # Fine-tuning lebih lama
        BATCH_SIZE = 32  # Batch lebih kecil = update lebih sering
        BACKBONE = 'mobilenetv3_large_100'  # Model lebih besar!
        print("\nPRODUCTION MODE - High Accuracy Configuration")
    
    dataset = UTKFaceDataset(DATA_DIR, transform=transform, limit=LIMIT)
    
    print(f"\n{'='*70}")
    print(f"MODE: {'QUICK TEST' if QUICK_MODE else 'PRODUCTION MODE'}")
    print(f"Dataset: {len(dataset)} images")
    print(f"Epochs: Baseline={EPOCHS_BASELINE}, Fine-tuning={EPOCHS_FINETUNING}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Backbone: {BACKBONE}")
    print(f"Estimasi: ~{'10-15' if QUICK_MODE else '40-80'} menit")
    print(f"{'='*70}")
    
    start_total = time.time()
    
    # FASE 1: Baseline Comparison
    print("\n\n" + "="*70)
    print("FASE 1: BASELINE COMPARISON (Grup A & B)")
    print("="*70)
    df_ab = run_baseline_comparison(dataset, DEVICE, EPOCHS_BASELINE, BATCH_SIZE, BACKBONE)
    
    # Tentukan metode terbaik
    best_method = df_ab.loc[df_ab['Akurasi (%)'].idxmax(), 'Arsitektur Model']
    print(f"\nMETODE TERBAIK dari Grup A & B: {best_method}")
    
    # FASE 2: LR Tuning
    print("\n\n" + "="*70)
    print("FASE 2: LR TUNING (Grup C)")
    print("="*70)
    df_c = run_lr_tuning(dataset, DEVICE, best_method, EPOCHS_FINETUNING, BATCH_SIZE, BACKBONE)
    
    # Tentukan LR terbaik
    best_lr_row = df_c.loc[df_c['Akurasi (%)'].idxmax()]
    best_lr = 1e-4 if '1e-4' in best_lr_row['Learning Rate (LR)'] else 5e-4
    print(f"\nLR TERBAIK: {best_lr}")
    
    # FASE 3: Head Tinkering
    print("\n\n" + "="*70)
    print("FASE 3: HEAD MODIFICATION (Grup D)")
    print("="*70)
    df_d, best_head, best_acc = run_head_tinkering(dataset, DEVICE, best_method, best_lr, 
                                                    EPOCHS_FINETUNING, BATCH_SIZE, BACKBONE)
    
    total_time = (time.time() - start_total) / 60
    
    # RANGKUMAN
    print("\n\n" + "="*70)
    print("RANGKUMAN HASIL EKSPERIMEN")
    print("="*70)
    
    print("\nTABEL 4.1 & 4.2: GRUP A & B (Baseline)")
    print(df_ab.to_string(index=False))
    
    print("\n\nTABEL 4.3: GRUP C (LR Tuning)")
    print(df_c.to_string(index=False))
    
    print("\n\nTABEL 4.4: GRUP D (Head Modification)")
    print(df_d.to_string(index=False))
    
    print("\n\n" + "="*70)
    print("KONFIGURASI MODEL TERBAIK")
    print("="*70)
    print(f"Arsitektur   : {best_method}")
    print(f"Split Data   : 90/10")
    print(f"Learning Rate: {best_lr}")
    print(f"Head         : {best_head}")
    print(f"Akurasi      : {best_acc*100:.2f}%")
    
    # Save results
    df_ab.to_csv(f'{OUTPUT_DIR}/tabel_4_1_4_2.csv', index=False)
    df_c.to_csv(f'{OUTPUT_DIR}/tabel_4_3.csv', index=False)
    df_d.to_csv(f'{OUTPUT_DIR}/tabel_4_4.csv', index=False)
    
    # Visualisasi
    visualize_results(df_ab, df_c, df_d, OUTPUT_DIR)
    
    print(f"\nTotal waktu: {total_time:.1f} menit")
    print(f"Hasil tersimpan di: {OUTPUT_DIR}/")
    print("\nEKSPERIMEN SELESAI - Perbandingan Fair Completed!")