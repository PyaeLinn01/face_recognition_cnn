# Where to Find Everything - Visual Guide

## 📊 MLflow UI - Where to Find Models

### Step 1: Start MLflow UI
```bash
mlflow ui --port 5000
```
Open: http://localhost:5000

### Step 2: Navigation Path

```
MLflow UI Home
  └─> Experiments
       └─> "face_recognition" experiment
            └─> Click any run (e.g., "attendance_20250114_123456")
                 │
                 ├─> Parameters Section (TOP)
                 │   ├─> mtcnn_min_confidence ← MTCNN setting
                 │   ├─> mtcnn_margin_ratio ← MTCNN setting
                 │   ├─> mtcnn_use_alignment ← MTCNN setting
                 │   ├─> mtcnn_version ← MTCNN version
                 │   ├─> facenet_use_prewhitening ← FaceNet setting
                 │   └─> facenet_use_tta_flip ← FaceNet setting
                 │
                 ├─> Metrics Section (MIDDLE)
                 │   ├─> verification_distance ← Distance metric
                 │   ├─> verification_threshold ← Threshold used
                 │   └─> verification_match ← 1.0 or 0.0
                 │
                 └─> Artifacts Section (BOTTOM) ⭐
                     ├─> facenet/ ← YOUR FACENET MODEL IS HERE!
                     │   ├─> MLmodel (model metadata)
                     │   ├─> model.keras (downloadable model file)
                     │   └─> requirements.txt
                     │
                     ├─> weights/ ← Model weights directory
                     │   └─> (all weight CSV files)
                     │
                     └─> sample_images/ ← Sample face images
                         └─> (example images from dataset)
```

### Step 3: View FaceNet Model

**In Artifacts section:**
1. Click on **`facenet/`** folder
2. You'll see the model files
3. Click **`model.keras`** to download
4. Or use code:
   ```python
   import mlflow.tensorflow
   model = mlflow.tensorflow.load_model("runs:/<run-id>/facenet")
   ```

### Step 4: View MTCNN Configuration

**In Parameters section:**
- Search/filter for `mtcnn`
- See all MTCNN settings:
  - Detection confidence
  - Margin ratio
  - Alignment settings
  - Version info

**Note:** MTCNN is a pre-trained library, so we track its **configuration**, not the model file.

---

## 📁 DVC - Where to Find Dataset

### Step 1: Check Status
```bash
python check_dvc_status.py
```

**Output shows:**
```
============================================================
DVC Dataset Tracking Status
============================================================
✓ DVC initialized

📊 DVC Status:
  ✓ All datasets up to date

📁 Tracked Datasets:
  • data/mongodb_export
    Size: 2.45 MB

💾 Remote Storage:
  local (./dvc_storage)

🔗 Git Integration:
  ✓ All .dvc files committed
============================================================
```

### Step 2: View Dataset Structure

```bash
# Pull dataset
dvc pull data/mongodb_export

# View structure
tree data/mongodb_export/
```

**Structure:**
```
data/mongodb_export/
├── images/              ← Face images organized by identity
│   ├── John/
│   │   ├── John_1.jpg
│   │   ├── John_2.jpg
│   │   ├── John_3.jpg
│   │   └── John_4.jpg
│   └── Jane/
│       └── ...
│
├── attendance.json      ← Attendance records
│   [
│     {
│       "timestamp": "2025-01-14T10:30:00",
│       "entered_name": "John",
│       "matched_identity": "John",
│       "distance": 0.5234
│     },
│     ...
│   ]
│
└── metadata.json        ← Export information
    {
      "export_timestamp": "2025-01-14T10:30:00",
      "mongodb_connection": "mongodb://localhost:27017/",
      "database": "face_attendance"
    }
```

### Step 3: View Dataset History

```bash
# See all versions
git log --oneline --all -- data/mongodb_export.dvc

# Output:
# abc1234 Add MongoDB dataset export - 2025-01-14
# def5678 Update dataset - 2025-01-13
# ...
```

### Step 4: Compare Versions

```bash
# Compare current vs previous
dvc diff HEAD~1 data/mongodb_export

# Compare specific commits
dvc diff abc1234 def5678 data/mongodb_export
```

---

## 🎯 Quick Reference Table

| What You Want | Where to Find It |
|--------------|------------------|
| **FaceNet Model** | MLflow UI → Run → Artifacts → `facenet/` |
| **MTCNN Config** | MLflow UI → Run → Parameters → `mtcnn_*` |
| **Verification Metrics** | MLflow UI → Run → Metrics → `verification_*` |
| **Dataset Images** | `data/mongodb_export/images/` |
| **Attendance Records** | `data/mongodb_export/attendance.json` |
| **Dataset Status** | Run: `python check_dvc_status.py` |
| **MLflow Runs** | Run: `python check_mlflow_runs.py` |

---

## 🚀 One-Command Viewing

### View Everything in MLflow
```bash
mlflow ui --port 5000 && open http://localhost:5000
```

### View Everything in DVC
```bash
python check_dvc_status.py && dvc pull data/mongodb_export && tree data/mongodb_export/
```

---

## 📸 Screenshot Guide

### MLflow UI - What You'll See

**Main Page:**
- List of experiments
- "face_recognition" experiment
- Number of runs

**Run Page:**
- **Top**: Parameters (MTCNN & FaceNet settings)
- **Middle**: Metrics (verification results)
- **Bottom**: Artifacts (models, weights, images)

**Artifacts:**
- `facenet/` folder (click to see model)
- `weights/` folder (click to see weights)
- `sample_images/` folder (click to see images)

### DVC Status - What You'll See

**After running `python check_dvc_status.py`:**
- ✓ DVC initialized
- 📊 Status (up to date or needs update)
- 📁 Tracked datasets list
- 💾 Remote storage info
- 🔗 Git integration status

---

## 💡 Pro Tips

1. **Bookmark MLflow runs**: Click star icon to bookmark important runs
2. **Download models**: Click on `model.keras` in Artifacts to download
3. **Compare runs**: Select multiple runs, click "Compare" button
4. **Export metrics**: Click "Download CSV" to export metrics
5. **DVC history**: Use `git log` to see dataset version history

---

For step-by-step instructions, see:
- `VIEW_RESULTS_GUIDE.md` - Detailed guide
- `QUICK_VIEW_GUIDE.md` - Quick reference
