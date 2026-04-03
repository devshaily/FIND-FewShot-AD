"""
FIND: Few-Shot Anomaly Inspection with Normal-Only Multi-Modal Data
Complete implementation for MVTec 3D-AD dataset

Requirements:
    pip install torch torchvision timm tifffile open3d tqdm scikit-learn opencv-python

MVTec 3D-AD Dataset Structure Expected:
    dataset/mvtec3d/
        bagel/
            train/
                good/
                    rgb/        ← .png files
                    xyz/        ← .tiff files (point cloud maps)
            test/
                good/
                    rgb/
                    xyz/
                crack/          ← anomaly categories
                    rgb/
                    xyz/
                    gt/         ← ground truth masks
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import cv2
import numpy as np
import tifffile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = 224
PATCH_SIZE = 8                          # ViT-B/8 → 28×28 spatial grid
PATCH_H   = IMG_SIZE // PATCH_SIZE      # 28
EMBED_DIM = 768
K_SHOT = None                                        # set to None for full-shot, or set 5/10/50
ALPHA     = 0.5 if K_SHOT is None else 0.8                         # few-shot weight: more cross-modal
EPOCHS    = 300
LR        = 1e-4
BATCH     = 2
CATEGORY  = "dowel"
DATA_ROOT = r"D:\Deep_Learning\FIND\mvtec_3d_anomaly_detection"

# ──────────────────────────────────────────────
# SURFACE NORMAL COMPUTATION
# ──────────────────────────────────────────────

def xyz_to_surface_normal(xyz: np.ndarray) -> np.ndarray:
    H, W, _ = xyz.shape
    xyz_clean = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
    pts = xyz_clean.reshape(-1, 3)
    valid_mask = np.abs(pts).sum(axis=1) > 0

    normal_map = np.zeros((H * W, 3), dtype=np.float32)

    if valid_mask.sum() > 9:   # ← need at least knn=9 points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[valid_mask])
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=9)
        )
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0., 0., 1000.])
        )
        normal_map[valid_mask] = np.asarray(pcd.normals, dtype=np.float32)

    normal_map = normal_map.reshape(H, W, 3)
    normal_map = (normal_map + 1.0) / 2.0
    return normal_map.astype(np.float32)


def remove_background(rgb: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Mask out background pixels (where xyz is all zeros) in the RGB image.
    Fills background with black (0) following the paper's protocol.
    """
    mask = (np.abs(xyz).sum(axis=-1) == 0)      # True where background
    rgb_masked = rgb.copy()
    rgb_masked[mask] = 0
    return rgb_masked


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────

class MVTec3DDataset(Dataset):
    """
    Loads RGB + surface normals from MVTec 3D-AD.
    For training: loads only normal (good) samples.
    For testing:  loads all samples + ground truth masks.
    """

    def __init__(self, root: str, category: str, split: str = "train", k_shot: int = None):
        """
        Args:
            root:     path to mvtec3d folder
            category: e.g. 'bagel'
            split:    'train' or 'test'
            k_shot:   if set, randomly pick k_shot samples for few-shot training
        """
        self.split = split
        self.samples = []   # list of (rgb_path, xyz_path, mask_path, label)

        if split == "train":
            rgb_dir = os.path.join(root, category, "train", "good", "rgb")
            xyz_dir = os.path.join(root, category, "train", "good", "xyz")
            paths = self._pair_paths(rgb_dir, xyz_dir)

            if k_shot is not None:
                np.random.shuffle(paths)
                paths = paths[:k_shot]

            for rgb_p, xyz_p in paths:
                self.samples.append((rgb_p, xyz_p, None, 0))

        else:  # test
            test_root = os.path.join(root, category, "test")
            for defect_type in sorted(os.listdir(test_root)):
                rgb_dir  = os.path.join(test_root, defect_type, "rgb")
                xyz_dir  = os.path.join(test_root, defect_type, "xyz")
                gt_dir   = os.path.join(test_root, defect_type, "gt")
                label    = 0 if defect_type == "good" else 1

                if not os.path.exists(rgb_dir):
                    continue

                for rgb_p, xyz_p in self._pair_paths(rgb_dir, xyz_dir):
                    # Find corresponding GT mask
                    mask_p = None
                    if label == 1 and os.path.exists(gt_dir):
                        fname = os.path.splitext(os.path.basename(rgb_p))[0] + ".png"
                        candidate = os.path.join(gt_dir, fname)
                        if os.path.exists(candidate):
                            mask_p = candidate
                    self.samples.append((rgb_p, xyz_p, mask_p, label))

        # Image transform (no normalisation — keep [0,1] for reconstruction loss)
        self.img_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),   # → [0, 1]
        ])

        # Normalisation for ViT-DINO (ImageNet stats)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    # ── helpers ──────────────────────────────

    @staticmethod
    def _pair_paths(rgb_dir: str, xyz_dir: str):
        """Match RGB .png files with XYZ .tiff files by stem name."""
        pairs = []
        if not os.path.exists(rgb_dir) or not os.path.exists(xyz_dir):
            return pairs
        rgb_files = {os.path.splitext(f)[0]: f for f in os.listdir(rgb_dir) if f.endswith(".png")}
        xyz_files = {os.path.splitext(f)[0]: f for f in os.listdir(xyz_dir) if f.endswith(".tiff")}
        for stem in sorted(rgb_files.keys()):
            if stem in xyz_files:
                pairs.append((
                    os.path.join(rgb_dir, rgb_files[stem]),
                    os.path.join(xyz_dir, xyz_files[stem]),
                ))
        return pairs

    # ── __getitem__ ───────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, xyz_path, mask_path, label = self.samples[idx]

        # ── Load RGB ─────────────────────────
        rgb_raw = cv2.imread(rgb_path)
        rgb_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)

        # ── Load XYZ → surface normal ─────────
        xyz = tifffile.imread(xyz_path).astype(np.float32)  # H×W×3
        normal_raw = xyz_to_surface_normal(xyz)              # H×W×3, [0,1]

        # ── Remove background from RGB ────────
        rgb_raw = remove_background(rgb_raw, xyz)

        # ── Apply transforms ──────────────────
        rgb_t    = self.img_transform(rgb_raw)              # [3, 224, 224]
        normal_t = self.img_transform((normal_raw * 255).astype(np.uint8))

        # Normalised versions for ViT input
        rgb_norm    = self.normalize(rgb_t)
        normal_norm = self.normalize(normal_t)

        # ── Ground truth mask ─────────────────
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            mask = torch.tensor(mask > 0, dtype=torch.float32)
        else:
            mask = torch.zeros(IMG_SIZE, IMG_SIZE)

        return {
            "rgb":         rgb_t,           # [0,1] — for reconstruction loss
            "normal":      normal_t,         # [0,1]
            "rgb_norm":    rgb_norm,         # normalised — for ViT input
            "normal_norm": normal_norm,
            "mask":        mask,
            "label":       label,
        }


# ──────────────────────────────────────────────
# TEACHER ViT  (DINO ViT-B/8, frozen)
# ──────────────────────────────────────────────

class TeacherViT(nn.Module):
    """
    Pretrained DINO ViT-B/8.
    Returns intermediate patch-token feature maps at layers 4, 6, 8, 10.
    Layer indices follow the paper (1-indexed → 0-indexed: 3,5,7,9).
    """

    INTRA_LAYERS = [3, 5, 7]      # layers for intra-modal loss (k=1..K-1)
    CROSS_LAYER  = 9              # layer for cross-modal student input

    def __init__(self):
        super().__init__()
        # Load DINO ViT-B/8
        self.vit = timm.create_model("vit_base_patch8_224.dino", pretrained=True)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

    @torch.no_grad()
    def forward(self, x):
        """
        Returns:
            intra_feats: list of 3 tensors, each (B, C, H, W) — layers 4,6,8
            cross_feat:  tensor (B, C, H, W)                  — layer 10
        """
        B = x.shape[0]

        # Patch embedding + CLS token + positional encoding
        tokens = self.vit.patch_embed(x)
        cls    = self.vit.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.vit.pos_embed
        tokens = self.vit.pos_drop(tokens)

        intra_feats = []
        cross_feat  = None

        for i, blk in enumerate(self.vit.blocks):
            tokens = blk(tokens)
            if i in self.INTRA_LAYERS:
                # Exclude CLS, reshape to spatial
                intra_feats.append(self._to_spatial(self.vit.norm(tokens), B))
            if i == self.CROSS_LAYER:
                cross_feat = self._to_spatial(self.vit.norm(tokens), B)

        return intra_feats, cross_feat   # ([B,C,H,W]×3,  B,C,H,W)

    @staticmethod
    def _to_spatial(tokens: torch.Tensor, B: int) -> torch.Tensor:
        """Remove CLS token and reshape (B, N+1, C) → (B, C, H, W)."""
        patch_tokens = tokens[:, 1:]          # (B, N, C)
        N, C = patch_tokens.shape[1], patch_tokens.shape[2]
        H = W = int(N ** 0.5)
        return patch_tokens.permute(0, 2, 1).reshape(B, C, H, W).contiguous()


# ──────────────────────────────────────────────
# CROSS-MODAL STUDENT
# ──────────────────────────────────────────────

class CrossModalStudent(nn.Module):
    """
    Encoder-decoder that maps features from one modality to another.
    Input/output: (B, C, H, W) at the cross-layer spatial resolution (28×28).
    """

    def __init__(self, in_channels: int = EMBED_DIM, out_channels: int = EMBED_DIM):
        super().__init__()

        # Encoder: compress to hidden representation
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Decoder: reconstruct cross-modal features
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────
# INTRA-MODAL STUDENT  (Reverse Distillation)
# ──────────────────────────────────────────────

class IntraModalStudent(nn.Module):
    """
    Decoder-only reverse distillation student.
    Takes deepest teacher feature (28×28) and progressively reconstructs
    back to the original image resolution (224×224).

    Multi-layer outputs match teacher's intermediate layers for supervision.
    """

    def __init__(self, in_channels: int = EMBED_DIM):
        super().__init__()

        # Stage 1: 28 → 56
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, EMBED_DIM, 3, padding=1),  # match teacher channel dim
        )

        # Stage 2: 56 → 112
        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(EMBED_DIM, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, EMBED_DIM, 3, padding=1),
        )

        # Stage 3: 112 → 224
        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(EMBED_DIM, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1),           # final RGB/normal output
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
            s1: (B, C, 56, 56)   — matches teacher layer feature resolution
            s2: (B, C, 112, 112)
            s0: (B, 3, 224, 224) — pixel-level reconstruction
        """
        s1 = self.stage1(x)       # 28 → 56
        s2 = self.stage2(s1)      # 56 → 112
        s0 = self.stage3(s2)      # 112 → 224
        return [s1, s2], s0       # intermediate features, final recon


# ──────────────────────────────────────────────
# FIND MODEL
# ──────────────────────────────────────────────

class FIND(nn.Module):

    def __init__(self):
        super().__init__()

        # Two shared-architecture but separate teachers (one per modality)
        self.teacher_2d = TeacherViT()
        self.teacher_3d = TeacherViT()

        # Cross-modal students
        self.cross_3d2d = CrossModalStudent()   # 3D features → predict 2D
        self.cross_2d3d = CrossModalStudent()   # 2D features → predict 3D

        # Intra-modal reverse distillation students
        self.intra_2d = IntraModalStudent()     # reconstruct RGB
        self.intra_3d = IntraModalStudent()     # reconstruct surface normal

    def forward(self, rgb_norm: torch.Tensor, normal_norm: torch.Tensor):
        """
        Args:
            rgb_norm:    (B, 3, 224, 224) normalised RGB
            normal_norm: (B, 3, 224, 224) normalised surface normal

        Returns dict of all intermediate and final outputs.
        """
        # ── Teacher forward passes (no grad) ─────────────────
        intra_f2d, cross_f2d = self.teacher_2d(rgb_norm)    # frozen
        intra_f3d, cross_f3d = self.teacher_3d(normal_norm) # frozen

        # ── Cross-modal students ──────────────────────────────
        # s_2d = predicted 2D features from 3D cross-layer feature
        s_2d = self.cross_3d2d(cross_f3d)
        # s_3d = predicted 3D features from 2D cross-layer feature
        s_3d = self.cross_2d3d(cross_f2d)

        # ── Intra-modal students (start from deepest teacher feat) ──
        # Use the last intra-layer feature as start point (paper: F_T^K)
        intra_s2d_feats, recon_2d = self.intra_2d(intra_f2d[-1])
        intra_s3d_feats, recon_3d = self.intra_3d(intra_f3d[-1])

        return {
            # Teacher features
            "cross_f2d":       cross_f2d,         # (B,C,28,28)
            "cross_f3d":       cross_f3d,
            "intra_f2d":       intra_f2d,          # list of 3 (B,C,H,W)
            "intra_f3d":       intra_f3d,

            # Cross-modal student outputs
            "s_2d":            s_2d,               # (B,C,28,28)
            "s_3d":            s_3d,

            # Intra-modal student outputs
            "intra_s2d_feats": intra_s2d_feats,    # list of 2 (B,C,H,W)
            "intra_s3d_feats": intra_s3d_feats,
            "recon_2d":        recon_2d,            # (B,3,224,224)
            "recon_3d":        recon_3d,
        }


# ──────────────────────────────────────────────
# LOSSES
# ──────────────────────────────────────────────

def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, averaged over batch and spatial positions."""
    # Flatten spatial: (B, C, H, W) → (B*H*W, C)
    B, C, H, W = a.shape
    a = a.permute(0,2,3,1).reshape(-1, C)
    b = b.permute(0,2,3,1).reshape(-1, C)
    return (1 - F.cosine_similarity(a, b, dim=-1)).mean()


def cross_modal_loss(out: dict) -> torch.Tensor:
    """
    Eq. (2) from paper: bidirectional cosine loss between cross-modal
    student predictions and corresponding teacher features.
    """
    loss_2d = cosine_loss(out["cross_f2d"], out["s_2d"])   # teacher 2D vs predicted 2D
    loss_3d = cosine_loss(out["cross_f3d"], out["s_3d"])   # teacher 3D vs predicted 3D
    return loss_2d + loss_3d


def intra_modal_loss(out: dict, rgb: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    loss = 0.0

    # FIXED: reversed(intra_f[:-1]) → [layer6, layer4], matching [s1, s2]
    teacher_2d_for_intra = list(reversed(out["intra_f2d"][:-1]))  # [layer6, layer4]
    teacher_3d_for_intra = list(reversed(out["intra_f3d"][:-1]))  # [layer6, layer4]

    for t_feat, s_feat in zip(teacher_2d_for_intra, out["intra_s2d_feats"]):
        if t_feat.shape[-1] != s_feat.shape[-1]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=False)
        loss += cosine_loss(t_feat, s_feat)

    for t_feat, s_feat in zip(teacher_3d_for_intra, out["intra_s3d_feats"]):
        if t_feat.shape[-1] != s_feat.shape[-1]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=False)
        loss += cosine_loss(t_feat, s_feat)

    loss += F.mse_loss(out["recon_2d"], rgb)
    loss += F.mse_loss(out["recon_3d"], normal)
    return loss


def total_loss(out: dict, rgb: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Eq. (4): L_total = L_cross + L_intra"""
    l_cross = cross_modal_loss(out)
    l_intra = intra_modal_loss(out, rgb, normal)
    return l_cross + l_intra


# ──────────────────────────────────────────────
# ANOMALY SCORING
# ──────────────────────────────────────────────

def compute_anomaly_map(model: FIND, batch: dict) -> np.ndarray:
    """
    Produces the final anomaly map A_final = α·A_cross + (1-α)·A_intra
    for a single batch. Returns (B, H, W) numpy array.
    """
    model.eval()
    with torch.no_grad():
        rgb_norm    = batch["rgb_norm"].to(DEVICE)
        normal_norm = batch["normal_norm"].to(DEVICE)
        rgb         = batch["rgb"].to(DEVICE)
        normal      = batch["normal"].to(DEVICE)

        out = model(rgb_norm, normal_norm)

    B = rgb_norm.shape[0]

    # ── Cross-modal anomaly maps ─────────────────────────
    # A_{2D→3D}: discrepancy between teacher 3D and student prediction of 3D
    a_2d3d = _feature_diff_map(out["cross_f3d"], out["s_3d"], size=IMG_SIZE)
    # A_{3D→2D}: discrepancy between teacher 2D and student prediction of 2D
    a_3d2d = _feature_diff_map(out["cross_f2d"], out["s_2d"], size=IMG_SIZE)
    a_cross = (a_2d3d + a_3d2d) / 2.0

    # ── Intra-modal anomaly maps ─────────────────────────
    # FIXED: same reversal as in loss
    intra_maps_2d = []
    for t_feat, s_feat in zip(list(reversed(out["intra_f2d"][:-1])), out["intra_s2d_feats"]):
        if t_feat.shape[-1] != s_feat.shape[-1]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=False)
        intra_maps_2d.append(_feature_diff_map(t_feat, s_feat, size=IMG_SIZE))
    # Add pixel reconstruction error
    recon_err_2d = (out["recon_2d"] - rgb).pow(2).mean(dim=1, keepdim=True)
    recon_err_2d = F.interpolate(recon_err_2d, IMG_SIZE, mode="bilinear").squeeze(1)
    intra_maps_2d.append(recon_err_2d.cpu().numpy())
    a_2d2d = np.mean(intra_maps_2d, axis=0)

    intra_maps_3d = []
    for t_feat, s_feat in zip(list(reversed(out["intra_f3d"][:-1])), out["intra_s3d_feats"]):
        if t_feat.shape[-1] != s_feat.shape[-1]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=False)
        intra_maps_3d.append(_feature_diff_map(t_feat, s_feat, size=IMG_SIZE))
    recon_err_3d = (out["recon_3d"] - normal).pow(2).mean(dim=1, keepdim=True)
    recon_err_3d = F.interpolate(recon_err_3d, IMG_SIZE, mode="bilinear").squeeze(1)
    intra_maps_3d.append(recon_err_3d.cpu().numpy())
    a_3d3d = np.mean(intra_maps_3d, axis=0)

    a_intra = (a_2d2d + a_3d3d) / 2.0

    # ── Final fusion  Eq.(5) ──────────────────────────────
    a_final = ALPHA * a_cross + (1 - ALPHA) * a_intra
    return a_final   # (B, H, W)


def _feature_diff_map(
    teacher_feat: torch.Tensor,
    student_feat: torch.Tensor,
    size: int = IMG_SIZE
) -> np.ndarray:
    """
    Compute per-pixel Euclidean distance between L2-normalised features,
    upsample to `size`, and apply Gaussian smoothing.
    Returns numpy (B, size, size).
    """
    # L2 normalise along channel dim
    t = F.normalize(teacher_feat, p=2, dim=1)
    s = F.normalize(student_feat, p=2, dim=1)

    diff = (t - s).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)

    # Upsample
    diff = F.interpolate(diff, size=(size, size), mode="bilinear", align_corners=False)
    diff = diff.squeeze(1).cpu().numpy()   # (B, size, size)

    # Gaussian smoothing per sample
    smoothed = np.stack([
        #cv2.GaussianBlur(d, (0, 0), sigmaX=4, sigmaY=4) for d in diff
        cv2.GaussianBlur(d, (0, 0), sigmaX=2, sigmaY=2) for d in diff
    ])
    return smoothed


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train(model: FIND, loader: DataLoader, epochs: int = EPOCHS):
    model.to(DEVICE)

    SAVE_AT = {50, 100, 150, 200, 250, 300}
    
    # Only train the 4 student networks — teachers are frozen
    trainable = (
        list(model.cross_3d2d.parameters()) +
        list(model.cross_2d3d.parameters()) +
        list(model.intra_2d.parameters()) +
        list(model.intra_3d.parameters())
    )
    optimizer = torch.optim.Adam(trainable, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        model.teacher_2d.eval()   # keep teachers frozen
        model.teacher_3d.eval()

        epoch_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            rgb_norm    = batch["rgb_norm"].to(DEVICE)
            normal_norm = batch["normal_norm"].to(DEVICE)
            rgb         = batch["rgb"].to(DEVICE)       # [0,1] for MSE
            normal      = batch["normal"].to(DEVICE)

            out  = model(rgb_norm, normal_norm)
            loss = total_loss(out, rgb, normal)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        scheduler.step()
        avg = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1:3d} | Loss: {avg:.4f}")

        # ── Save checkpoint at milestone epochs ──
        if (epoch + 1) in SAVE_AT:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/find_{CATEGORY}_k{K_SHOT}_ep{epoch+1}.pt"
            torch.save({
                "cross_3d2d": model.cross_3d2d.state_dict(),
                "cross_2d3d": model.cross_2d3d.state_dict(),
                "intra_2d":   model.intra_2d.state_dict(),
                "intra_3d":   model.intra_3d.state_dict(),
                "epoch":      epoch + 1,
                "loss":       avg,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved → {ckpt_path}")

    print("Training complete.")


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def evaluate(model: FIND, loader: DataLoader):
    """
    Computes image-level AUROC and pixel-level AUROC on the test set.
    """
    model.eval()
    all_scores  = []    # image-level anomaly scores
    all_labels  = []    # image-level GT labels
    all_maps    = []    # pixel-level anomaly maps (flattened)
    all_masks   = []    # pixel-level GT masks    (flattened)

    for batch in tqdm(loader, desc="Evaluating"):
        a_maps = compute_anomaly_map(model, batch)   # (B, H, W)

        # Image-level score = max of anomaly map
        scores = a_maps.reshape(a_maps.shape[0], -1).max(axis=1)
        all_scores.extend(scores.tolist())
        all_labels.extend(batch["label"].tolist())

        # Pixel-level
        masks = batch["mask"].numpy()   # (B, H, W)
        all_maps.extend(a_maps.reshape(-1).tolist())
        all_masks.extend(masks.reshape(-1).tolist())

    # Image-level AUROC
    img_auroc = roc_auc_score(all_labels, all_scores)

    # Pixel-level AUROC (only if any anomalies exist)
    if len(set(all_masks)) > 1:
        pix_auroc = roc_auc_score(
            (np.array(all_masks) > 0).astype(int), all_maps
        )
    else:
        pix_auroc = float("nan")

    print(f"\nResults:")
    print(f"  Image-level AUROC : {img_auroc:.4f}")
    print(f"  Pixel-level AUROC : {pix_auroc:.4f}")
    return img_auroc, pix_auroc

def save_anomaly_maps_for_official_eval(model, loader, save_dir, category):
    model.eval()
    sample_idx = 0
    dataset = loader.dataset

    for batch in tqdm(loader, desc="Saving anomaly maps"):
        a_maps = compute_anomaly_map(model, batch)  # (B, 224, 224)

        for i in range(len(a_maps)):
            rgb_path = dataset.samples[sample_idx][0]
            parts = rgb_path.replace("\\", "/").split("/")
            defect_type = parts[-3]
            image_id    = os.path.splitext(parts[-1])[0]

            # ── Get original image size ───────────────
            original_rgb = cv2.imread(rgb_path)
            orig_h, orig_w = original_rgb.shape[:2]  # e.g. 400×400

            # ── Resize anomaly map to original size ───
            a_map_resized = cv2.resize(
                a_maps[i],
                (orig_w, orig_h),                    # cv2 uses (W, H)
                interpolation=cv2.INTER_LINEAR
            )

            out_dir = os.path.join(save_dir, category, "test", defect_type)
            os.makedirs(out_dir, exist_ok=True)
            tifffile.imwrite(
                os.path.join(out_dir, f"{image_id}.tiff"),
                a_map_resized.astype(np.float32)     # now 400×400
            )
            sample_idx += 1

    print(f"Anomaly maps saved → {save_dir}/")
    
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_results(model: FIND, loader: DataLoader, num_samples: int = 8, save_dir: str = "visualizations"):
    """
    Saves side-by-side visualizations:
    RGB | Surface Normal | Anomaly Map | Overlay
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    sample_count = 0
    fig_rows = []

    for batch in tqdm(loader, desc="Visualizing"):
        a_maps = compute_anomaly_map(model, batch)   # (B, H, W)

        for i in range(len(a_maps)):
            if sample_count >= num_samples:
                break

            # ── Get images ───────────────────────────
            rgb    = batch["rgb"][i].permute(1,2,0).numpy()        # (H,W,3) [0,1]
            normal = batch["normal"][i].permute(1,2,0).numpy()     # (H,W,3) [0,1]
            mask   = batch["mask"][i].numpy()                       # (H,W)
            label  = batch["label"][i].item()
            a_map  = a_maps[i]                                      # (H,W)

            # ── Normalise anomaly map to [0,1] ───────
            a_min, a_max = a_map.min(), a_map.max()
            a_norm = (a_map - a_min) / (a_max - a_min + 1e-8)

            # ── Heatmap overlay on RGB ───────────────
            heatmap = cm.jet(a_norm)[:, :, :3]           # (H,W,3) RGB heatmap
            overlay = 0.5 * rgb + 0.5 * heatmap          # blend
            overlay = np.clip(overlay, 0, 1)

            # ── Plot ─────────────────────────────────
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            fig.suptitle(
                f"Sample {sample_count+1} | Label: {'ANOMALY' if label==1 else 'NORMAL'} | "
                f"Max Score: {a_map.max():.4f}",
                fontsize=12, fontweight='bold',
                color='red' if label == 1 else 'green'
            )

            axes[0].imshow(rgb)
            axes[0].set_title("RGB Input")
            axes[0].axis("off")

            axes[1].imshow(normal)
            axes[1].set_title("Surface Normal")
            axes[1].axis("off")

            axes[2].imshow(a_norm, cmap="jet")
            axes[2].set_title("Anomaly Map")
            axes[2].axis("off")

            axes[3].imshow(overlay)
            axes[3].set_title("Overlay")
            axes[3].axis("off")

            # GT mask (or blank if normal)
            if mask.max() > 0:
                axes[4].imshow(mask, cmap="Reds")
                axes[4].set_title("GT Mask")
            else:
                axes[4].imshow(np.zeros_like(mask), cmap="gray")
                axes[4].set_title("GT Mask (None)")
            axes[4].axis("off")

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"sample_{sample_count+1}_{'anomaly' if label==1 else 'normal'}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            sample_count += 1

        if sample_count >= num_samples:
            break

    print(f"\nSaved {sample_count} visualizations → '{save_dir}/'")
    
#  ──────────────────────────────────────────────
#      MAIN
#  ──────────────────────────────────────────────

if __name__ == "__main__":
    test_dataset = MVTec3DDataset(
        root=DATA_ROOT, category=CATEGORY, split="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH, 
                             shuffle=False, num_workers=0)

    model = FIND().to(DEVICE)
    ckpt = torch.load(
        f"checkpoints/find_{CATEGORY}_kNone.pt", 
        map_location=DEVICE
    )
    model.cross_3d2d.load_state_dict(ckpt["cross_3d2d"])
    model.cross_2d3d.load_state_dict(ckpt["cross_2d3d"])
    model.intra_2d.load_state_dict(ckpt["intra_2d"])
    model.intra_3d.load_state_dict(ckpt["intra_3d"])
    print("Checkpoint loaded!")

    # Re-save at correct resolution
    save_anomaly_maps_for_official_eval(
        model, test_loader, "anomaly_maps", CATEGORY
    )