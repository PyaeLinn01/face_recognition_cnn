# How to View Your Models & Dataset - Complete Guide

## 🎯 Quick Answer

### See FaceNet & MTCNN in MLflow:
```bash
mlflow ui --port 5000
# Open http://localhost:5000
# Go to: Run → Artifacts → facenet/ (for FaceNet model)
# Go to: Run → Parameters → mtcnn_* (for MTCNN config)
```

### See Dataset in DVC:
```bash
python check_dvc_status.py
# Or: dvc status
# Dataset location: data/mongodb_export/
```

---

## 📊 MLflow UI - Step by Step

### 1. Start MLflow UI
```bash
cd /Users/pyaelinn/face_recon/face_recognition_cnn
mlflow ui --port 5000
```

### 2. Open Browser
```
http://localhost:5000
```

### 3. Find FaceNet Model

**Navigation:**
1. Click **"face_recognition"** experiment
2. Click any run (e.g., `attendance_20250114_123456`)
3. Scroll down to **"Artifacts"** section
4. Click **`facenet/`** folder ← **YOUR MODEL IS HERE!**

**What you'll see:**
```
facenet/
├── MLmodel          ← Model metadata (click to view)
├── model.keras      ← Your FaceNet model (click to download)
├── requirements.txt ← Dependencies
└── ...
```

**To use the model:**
```python
import mlflow.tensorflow
model = mlflow.tensorflow.load_model("runs:/<run-id>/facenet")
```

### 4. Find MTCNN Configuration

**In the same run page:**
1. Look at **"Parameters"** section (top of page)
2. Search/filter for `mtcnn`
3. You'll see:
   - `mtcnn_min_confidence: 0.90`
   - `mtcnn_margin_ratio: 0.20`
   - `mtcnn_use_alignment: True`
   - `mtcnn_version: 0.1.1`
   - `mtcnn_model_type: MTCNN Face Detector`

**Note:** MTCNN is a pre-trained library, so we track its **configuration**, not the model file itself.

### 5. View Metrics

**In "Metrics" section:**
- `verification_distance` - Distance between embeddings
- `verification_threshold` - Threshold used
- `verification_match` - 1.0 (match) or 0.0 (no match)

### 6. Compare Runs

1. Select multiple runs (checkboxes)
2. Click **"Compare"** button
3. See side-by-side:
   - Parameters comparison
   - Metrics comparison
   - Model artifacts

---

## 📁 DVC Dataset - Step by Step

### 1. Check Status
```bash
python check_dvc_status.py
```

**Output shows:**
- ✓ Which datasets are tracked
- ✓ Data size
- ✓ If data needs to be pulled
- ✓ Git integration status

### 2. View Dataset
```bash
# Pull dataset if needed
dvc pull data/mongodb_export

# View structure
ls -R data/mongodb_export/

# View contents
cat data/mongodb_export/metadata.json
```

### 3. See Dataset History
```bash
# See all versions
git log --oneline --all -- data/mongodb_export.dvc

# Compare versions
dvc diff HEAD~1 data/mongodb_export
```

### 4. Dataset Structure

After pulling, you'll see:
```
data/mongodb_export/
├── images/              ← Face images by identity
│   ├── John/
│   │   ├── John_1.jpg
│   │   ├── John_2.jpg
│   │   ├── John_3.jpg
│   │   └── John_4.jpg
│   └── Jane/
│       └── ...
│
├── attendance.json       ← All attendance records
│   [
│     {
│       "timestamp": "2025-01-14T10:30:00",
│       "entered_name": "John",
│       "matched_identity": "John",
│       "distance": 0.5234
│     }
│   ]
│
└── metadata.json         ← Export info
    {
      "export_timestamp": "2025-01-14T10:30:00",
      "mongodb_connection": "mongodb://localhost:27017/",
      "database": "face_attendance"
    }
```

---

## 🔍 Visual Navigation Guide

### MLflow UI Layout

```
┌─────────────────────────────────────┐
│  MLflow UI (http://localhost:5000) │
└─────────────────────────────────────┘
           │
           ├─> Experiments
           │   └─> "face_recognition"
           │       └─> [List of Runs]
           │           └─> Click a run
           │               │
           │               ├─> Parameters (TOP)
           │               │   ├─> mtcnn_* ← MTCNN HERE
           │               │   └─> facenet_* ← FaceNet params
           │               │
           │               ├─> Metrics (MIDDLE)
           │               │   └─> verification_* ← Results
           │               │
           │               └─> Artifacts (BOTTOM)
           │                   ├─> facenet/ ← FACENET MODEL HERE
           │                   ├─> weights/
           │                   └─> sample_images/
```

### DVC Dataset Location

```
Project Root
├── data/
│   └── mongodb_export/  ← DATASET HERE
│       ├── images/
│       ├── attendance.json
│       └── metadata.json
│
├── data/mongodb_export.dvc  ← DVC tracking file
└── .dvc/  ← DVC configuration
```

---

## ✅ Verification Checklist

### MLflow Models:
- [ ] Started MLflow UI: `mlflow ui --port 5000`
- [ ] Opened http://localhost:5000
- [ ] Found "face_recognition" experiment
- [ ] Clicked on a run
- [ ] Saw `facenet/` in Artifacts (FaceNet model)
- [ ] Saw `mtcnn_*` in Parameters (MTCNN config)
- [ ] Saw metrics in Metrics section

### DVC Dataset:
- [ ] Ran: `python check_dvc_status.py`
- [ ] Saw tracked datasets listed
- [ ] Ran: `dvc pull data/mongodb_export` (if needed)
- [ ] Viewed: `ls -R data/mongodb_export/`
- [ ] Checked history: `git log --oneline -- data/mongodb_export.dvc`

---

## 🚀 One-Liner Commands

### View MLflow:
```bash
mlflow ui --port 5000 && open http://localhost:5000
```

### View DVC:
```bash
python check_dvc_status.py && dvc pull data/mongodb_export && ls -R data/mongodb_export/
```

### Check Everything:
```bash
python check_mlflow_runs.py && echo "---" && python check_dvc_status.py
```

---

## 📝 What Gets Tracked

### MLflow Tracks:
- ✅ **FaceNet Model** (saved as `facenet/model.keras`)
- ✅ **MTCNN Configuration** (parameters: `mtcnn_*`)
- ✅ **Model Weights** (in `weights/` folder)
- ✅ **Sample Images** (in `sample_images/` folder)
- ✅ **Verification Metrics** (distance, threshold, match)
- ✅ **Dataset Info** (num_images, num_identities)

### DVC Tracks:
- ✅ **Face Images** (in `images/` organized by identity)
- ✅ **Attendance Records** (in `attendance.json`)
- ✅ **Export Metadata** (in `metadata.json`)

---

## 🎓 Learning Path

1. **Start Simple**: Run `mlflow ui` and explore the UI
2. **Find Models**: Look in Artifacts section
3. **Check DVC**: Run `python check_dvc_status.py`
4. **Read Guides**: See `VIEW_RESULTS_GUIDE.md` for details

---

## 📚 Related Guides

- `QUICK_VIEW_GUIDE.md` - 2-minute quick start
- `VIEW_RESULTS_GUIDE.md` - Detailed step-by-step
- `WHERE_TO_FIND.md` - Visual navigation guide
- `MLFLOW_DVC_GUIDE.md` - Complete documentation

---

## 💡 Remember

- **FaceNet Model** = MLflow UI → Run → Artifacts → `facenet/`
- **MTCNN Config** = MLflow UI → Run → Parameters → `mtcnn_*`
- **Dataset** = `data/mongodb_export/` (after DVC pull)
- **Status** = Run `python check_dvc_status.py` or `python check_mlflow_runs.py`
