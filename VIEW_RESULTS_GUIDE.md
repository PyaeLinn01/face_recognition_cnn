# How to View MLflow Models & DVC Dataset Tracking

This guide shows you exactly how to view your FaceNet and MTCNN models in MLflow UI and track your dataset with DVC.

---

## 📊 Viewing Models in MLflow UI

### Step 1: Start MLflow UI

```bash
# Navigate to your project directory
cd /Users/pyaelinn/face_recon/face_recognition_cnn

# Start MLflow UI
mlflow ui --port 5000
```

You should see:
```
[INFO] Starting MLflow UI at http://127.0.0.1:5000
```

### Step 2: Open MLflow UI in Browser

Open your browser and go to:
```
http://localhost:5000
```

### Step 3: View Your Models

#### A. View Experiments

1. **Main Page**: You'll see the "face_recognition" experiment
2. **Click on the experiment** to see all runs
3. Each run represents a verification session

#### B. View FaceNet Model

1. **Click on any run** (e.g., `attendance_20250114_123456`)
2. Scroll down to **"Artifacts"** section
3. You'll see:
   - `facenet/` - **This is your FaceNet model!**
     - Click to expand
     - Contains: `MLmodel`, `model.keras`, `requirements.txt`, etc.
   - `weights/` - Model weights directory
   - `sample_images/` - Sample face images

**To load the model:**
```python
import mlflow
import mlflow.tensorflow

# Load the logged FaceNet model
model = mlflow.tensorflow.load_model("runs:/<run-id>/facenet")
```

#### C. View MTCNN Information

1. In the same run page, look at **"Parameters"** section
2. You'll see MTCNN parameters:
   - `mtcnn_min_confidence` - Detection confidence threshold
   - `mtcnn_margin_ratio` - Face crop margin
   - `mtcnn_use_alignment` - Alignment setting
   - `mtcnn_version` - MTCNN version
   - `mtcnn_model_type` - "MTCNN Face Detector"
   - `detector_type` - "MTCNN"

**Note**: MTCNN is a pre-trained library, so we log its configuration rather than the model file itself.

#### D. View Metrics

In the **"Metrics"** section, you'll see:
- `verification_distance` - Distance between embeddings
- `verification_threshold` - Threshold used
- `verification_match` - 1.0 (match) or 0.0 (no match)

#### E. Compare Runs

1. **Select multiple runs** (checkboxes on the left)
2. Click **"Compare"** button
3. See side-by-side comparison of:
   - Parameters (MTCNN & FaceNet settings)
   - Metrics (verification results)
   - Model artifacts

### Step 4: View Model Registry (Optional)

1. Click **"Models"** tab in MLflow UI
2. You'll see registered models:
   - `facenet` - Your FaceNet model
3. Click to see all versions

---

## 📁 Viewing Dataset Tracking in DVC

### Step 1: Check DVC Status

```bash
# See what's tracked
dvc status

# See detailed status
dvc status --verbose
```

**Output shows:**
- Which files are tracked
- If data is up to date
- If data needs to be pulled/pushed

### Step 2: View Tracked Datasets

```bash
# List all tracked files
dvc list .

# See dataset info
dvc diff
```

### Step 3: View Dataset History

```bash
# See commit history for dataset
git log --oneline --all -- data/mongodb_export.dvc

# See when dataset was last updated
dvc status data/mongodb_export
```

### Step 4: View Dataset in Git

```bash
# See .dvc file (contains hash and metadata)
cat data/mongodb_export.dvc

# Output shows:
# - md5 hash of dataset
# - Size
# - Path to data in DVC storage
```

### Step 5: View Dataset Contents

```bash
# Pull dataset if needed
dvc pull data/mongodb_export

# View exported data
ls -la data/mongodb_export/
# You'll see:
# - images/ (face images organized by identity)
# - attendance.json (attendance records)
# - metadata.json (export information)
```

### Step 6: Compare Dataset Versions

```bash
# See differences between versions
dvc diff HEAD~1 data/mongodb_export

# Or compare specific commits
dvc diff <commit1> <commit2> data/mongodb_export
```

---

## 🎯 Quick Reference

### MLflow UI Navigation

| What to View | Where in MLflow UI |
|-------------|-------------------|
| **FaceNet Model** | Run → Artifacts → `facenet/` |
| **MTCNN Config** | Run → Parameters → `mtcnn_*` |
| **Verification Metrics** | Run → Metrics → `verification_*` |
| **Dataset Info** | Run → Parameters → `dataset_*` |
| **Compare Runs** | Select runs → Click "Compare" |
| **Model Registry** | "Models" tab → `facenet` |

### DVC Commands

| What to Check | Command |
|--------------|---------|
| **Dataset Status** | `dvc status` |
| **View Tracked Files** | `dvc list .` |
| **Pull Dataset** | `dvc pull data/mongodb_export` |
| **See History** | `git log --oneline data/mongodb_export.dvc` |
| **Compare Versions** | `dvc diff HEAD~1 data/mongodb_export` |

---

## 🔍 Detailed Viewing Guide

### MLflow: View FaceNet Model Details

1. **In MLflow UI**, go to a run
2. Click **"Artifacts"** → `facenet/`
3. You'll see:
   ```
   facenet/
   ├── MLmodel          # Model metadata
   ├── model.keras      # Saved Keras model
   ├── requirements.txt  # Dependencies
   └── ...
   ```
4. **Download model**: Click on `model.keras` to download
5. **View model info**: Click on `MLmodel` to see model structure

### MLflow: View MTCNN Configuration

1. In run page, **"Parameters"** section
2. Filter by typing `mtcnn` in search
3. See all MTCNN settings:
   - Detection parameters
   - Processing options
   - Version information

### DVC: View Dataset Structure

```bash
# After pulling
tree data/mongodb_export/

# Output:
data/mongodb_export/
├── images/
│   ├── John/
│   │   ├── John_1.jpg
│   │   ├── John_2.jpg
│   │   ├── John_3.jpg
│   │   └── John_4.jpg
│   └── Jane/
│       └── ...
├── attendance.json
└── metadata.json
```

### DVC: View Dataset Metadata

```bash
# View export metadata
cat data/mongodb_export/metadata.json

# Shows:
# - Export timestamp
# - MongoDB connection
# - Database name
```

---

## 📈 Visualizing Results

### MLflow: Create Charts

1. In MLflow UI, select multiple runs
2. Click **"Compare"**
3. View charts:
   - **Metrics over time** (if you have multiple runs)
   - **Parameter comparison** (side-by-side)
   - **Metric distributions**

### DVC: Visualize Dataset Changes

```bash
# See dataset size changes
dvc metrics show

# See when dataset was modified
git log --format="%h %ad %s" --date=short -- data/mongodb_export.dvc
```

---

## 🚀 Quick Start Commands

### View Everything in MLflow

```bash
# 1. Start MLflow UI
mlflow ui --port 5000

# 2. Open browser
open http://localhost:5000  # macOS
# or
xdg-open http://localhost:5000  # Linux
```

### View Everything in DVC

```bash
# 1. Check status
dvc status

# 2. Pull latest dataset
dvc pull data/mongodb_export

# 3. View dataset
ls -R data/mongodb_export/

# 4. View history
git log --oneline --graph --all -- data/mongodb_export.dvc
```

---

## 💡 Tips

### MLflow:
- **Bookmark runs**: Tag important runs for easy finding
- **Export data**: Click "Download CSV" to export metrics
- **Share runs**: Copy run ID to share with team

### DVC:
- **Regular exports**: Export MongoDB data regularly
- **Tag versions**: Use Git tags for important dataset versions
- **Backup**: Push to remote storage regularly

---

## 🐛 Troubleshooting

### MLflow UI Not Showing Models

**Problem**: Models not visible in Artifacts
- **Solution**: Ensure you've run verification with MLflow enabled
- **Solution**: Check `./mlruns/` directory exists
- **Solution**: Verify MLflow tracking is enabled in Streamlit

### DVC Status Shows "Not in cache"

**Problem**: Dataset not found locally
- **Solution**: Run `dvc pull data/mongodb_export`
- **Solution**: Check remote storage is configured
- **Solution**: Verify `.dvc` file exists

### Can't See Dataset History

**Problem**: No Git history
- **Solution**: Ensure you've committed `.dvc` files to Git
- **Solution**: Check `git log --all -- data/mongodb_export.dvc`

---

## 📝 Example Workflow

### Daily Workflow:

1. **Morning**: 
   ```bash
   # Export MongoDB data
   python export_mongodb_for_dvc.py --export-all
   
   # Version with DVC
   dvc add data/mongodb_export/
   git add data/mongodb_export.dvc
   git commit -m "Daily dataset export"
   dvc push
   ```

2. **During Day**:
   - Run attendance system with MLflow tracking enabled
   - Verify faces (automatically tracked)

3. **Evening**:
   ```bash
   # View results in MLflow
   mlflow ui --port 5000
   
   # Check dataset status
   dvc status
   ```

---

For more details, see:
- `MLFLOW_DVC_GUIDE.md` - Complete guide
- `DVC_MONGODB_GUIDE.md` - MongoDB DVC guide
