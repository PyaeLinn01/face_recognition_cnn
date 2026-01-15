# Quick Guide: View Models & Dataset

## 🚀 Quick Start (2 minutes)

### View MLflow Models

```bash
# 1. Start MLflow UI
mlflow ui --port 5000

# 2. Open browser
open http://localhost:5000  # macOS
```

**In MLflow UI:**
1. Click **"face_recognition"** experiment
2. Click any run (e.g., `attendance_20250114_123456`)
3. Scroll to **"Artifacts"** section
4. You'll see:
   - **`facenet/`** ← Your FaceNet model is here!
   - **`weights/`** ← Model weights
   - **`sample_images/`** ← Sample face images

**To see MTCNN:**
- Go to **"Parameters"** section
- Look for `mtcnn_*` parameters
- MTCNN config is logged as parameters (it's a library, not a saved model)

### View DVC Dataset

```bash
# 1. Check what's tracked
python check_dvc_status.py

# Or manually:
dvc status

# 2. View dataset
dvc pull data/mongodb_export
ls -R data/mongodb_export/
```

---

## 📊 Detailed Viewing

### MLflow: See FaceNet Model

**Step-by-step:**
1. `mlflow ui --port 5000`
2. Open http://localhost:5000
3. Click experiment: **"face_recognition"**
4. Click a run name
5. Scroll to **"Artifacts"**
6. Click **`facenet/`** folder
7. You'll see:
   ```
   facenet/
   ├── MLmodel          ← Model metadata
   ├── model.keras      ← Your FaceNet model file
   ├── requirements.txt ← Dependencies
   └── ...
   ```

**Download model:**
- Click on `model.keras` to download
- Or use Python: `mlflow.tensorflow.load_model("runs:/<run-id>/facenet")`

### MLflow: See MTCNN Configuration

**In the same run page:**
1. Look at **"Parameters"** section (top of page)
2. Filter/search for `mtcnn`
3. You'll see:
   - `mtcnn_min_confidence: 0.90`
   - `mtcnn_margin_ratio: 0.20`
   - `mtcnn_use_alignment: True`
   - `mtcnn_version: 0.1.1`
   - `mtcnn_model_type: MTCNN Face Detector`

**Note:** MTCNN is a pre-trained library, so we track its **configuration** rather than the model file.

### DVC: See Dataset Tracking

**Check status:**
```bash
python check_dvc_status.py
```

**Output shows:**
- ✓ Which datasets are tracked
- ✓ Data size
- ✓ If data needs to be pulled
- ✓ Git integration status

**View dataset:**
```bash
# Pull if needed
dvc pull data/mongodb_export

# View structure
tree data/mongodb_export/

# View contents
cat data/mongodb_export/metadata.json
cat data/mongodb_export/attendance.json | head -20
```

**See history:**
```bash
# See when dataset was updated
git log --oneline --all -- data/mongodb_export.dvc

# Compare versions
dvc diff HEAD~1 data/mongodb_export
```

---

## 🎯 What You'll See

### In MLflow UI:

**FaceNet Model:**
- Location: Run → Artifacts → `facenet/`
- Contains: Saved Keras model, metadata, requirements
- Can download or load programmatically

**MTCNN Info:**
- Location: Run → Parameters → `mtcnn_*`
- Contains: Detection settings, version, configuration
- Note: MTCNN is a library, config is tracked

**Metrics:**
- Location: Run → Metrics
- Shows: Verification distances, thresholds, match results

### In DVC:

**Tracked Datasets:**
- Location: `data/mongodb_export/`
- Contains: Face images, attendance records, metadata
- Versioned: Each export creates a new version

**Status:**
- Shows: What's tracked, what needs updating
- History: Git commits show dataset versions

---

## 💡 Pro Tips

1. **Bookmark MLflow runs**: Tag important runs for easy finding
2. **Regular DVC exports**: Export MongoDB data daily/weekly
3. **Compare runs**: Select multiple runs in MLflow to compare
4. **Version tags**: Use Git tags for important dataset versions

---

## 🐛 Troubleshooting

**Can't see models in MLflow:**
- Make sure you've run verification with MLflow enabled
- Check `./mlruns/` directory exists
- Verify MLflow tracking checkbox is enabled

**Can't see dataset in DVC:**
- Run: `python export_mongodb_for_dvc.py --export-all`
- Then: `dvc add data/mongodb_export/`
- Check: `dvc status`

For more details, see `VIEW_RESULTS_GUIDE.md`
