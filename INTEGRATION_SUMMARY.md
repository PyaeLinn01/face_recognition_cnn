# MLflow & DVC Integration Summary

## ✅ What Was Added

### 1. MLflow Tracking in `attend_app.py`

**MTCNN Parameters Tracked:**
- `min_confidence` - Minimum face detection confidence
- `margin_ratio` - Face crop margin ratio
- `use_alignment` - Whether face alignment is used

**FaceNet Parameters Tracked:**
- `use_prewhitening` - Whether prewhitening normalization is used
- `use_tta_flip` - Whether test-time augmentation (flip) is used

**Metrics Tracked:**
- `verification_distance` - Distance between query and reference embeddings
- `verification_threshold` - Matching threshold used
- `verification_match` - Whether verification was successful (1.0 or 0.0)
- `verified_identity` - Identity that was matched

**Dataset Info Tracked:**
- `dataset_num_images` - Total number of images
- `dataset_num_identities` - Number of unique identities

### 2. DVC for MongoDB Dataset

**Export Script:** `export_mongodb_for_dvc.py`
- Exports face images from MongoDB to directory structure
- Exports attendance records to JSON
- Creates metadata file with export information

**Usage:**
```bash
# Export all MongoDB data
python export_mongodb_for_dvc.py --export-all

# Add to DVC
dvc add data/mongodb_export/

# Commit
git add data/mongodb_export.dvc
git commit -m "Add MongoDB dataset export"
```

## 🚀 How to Use

### MLflow Tracking

1. **Enable in Streamlit UI:**
   - Go to Attendance tab
   - Check "Enable MLflow tracking" in sidebar
   - All verifications will be tracked automatically

2. **View Results:**
   ```bash
   # Start MLflow UI
   mlflow ui --port 5000
   
   # Open in browser
   http://localhost:5000
   ```

3. **What Gets Tracked:**
   - Every face verification attempt
   - MTCNN and FaceNet parameters
   - Verification results (distance, match status)
   - Dataset information

### DVC for MongoDB

1. **Export MongoDB Data:**
   ```bash
   python export_mongodb_for_dvc.py \
     --connection-string "mongodb://localhost:27017/" \
     --database "face_attendance" \
     --export-all
   ```

2. **Version with DVC:**
   ```bash
   dvc add data/mongodb_export/
   git add data/mongodb_export.dvc
   git commit -m "Version MongoDB dataset"
   dvc push
   ```

3. **Restore Dataset:**
   ```bash
   dvc pull data/mongodb_export
   ```

## 📁 Files Created/Modified

### New Files:
- `mlflow_utils.py` - MLflow tracking utilities
- `export_mongodb_for_dvc.py` - MongoDB export script
- `MLFLOW_DVC_GUIDE.md` - Comprehensive guide
- `DVC_MONGODB_GUIDE.md` - MongoDB DVC guide
- `QUICK_START_MLFLOW_DVC.md` - Quick start guide
- `example_mlflow_integration.py` - Code examples
- `.dvcignore` - DVC ignore patterns
- `dvc.yaml` - DVC pipeline configuration
- `.dvc/config` - DVC remote configuration

### Modified Files:
- `attend_app.py` - Added MLflow tracking integration
- `requirements.txt` - Added mlflow and dvc
- `Dockerfile` - Added MLflow environment variables

## 🎯 Features

### MLflow Integration:
- ✅ Automatic parameter logging (MTCNN & FaceNet)
- ✅ Verification metrics tracking
- ✅ Dataset information logging
- ✅ Session-based run tracking
- ✅ Error handling (graceful fallback if MLflow unavailable)

### DVC Integration:
- ✅ MongoDB data export
- ✅ Directory structure preservation
- ✅ Metadata tracking
- ✅ Attendance records export
- ✅ Easy versioning workflow

## 📊 MLflow UI

After enabling tracking and running verifications:

1. Start MLflow UI: `mlflow ui --port 5000`
2. View experiments: http://localhost:5000
3. See all tracked parameters and metrics
4. Compare different verification runs
5. Track model performance over time

## 🔄 Workflow

### Daily Workflow:
1. **Run attendance system** with MLflow tracking enabled
2. **Export MongoDB data** periodically: `python export_mongodb_for_dvc.py --export-all`
3. **Version dataset**: `dvc add data/mongodb_export/ && git commit`
4. **Push to storage**: `dvc push`

### Experiment Tracking:
1. Enable MLflow tracking in Streamlit
2. Adjust parameters (threshold, confidence, etc.)
3. Run verifications
4. View results in MLflow UI
5. Compare different parameter combinations

## 📝 Notes

- MLflow tracking is **optional** - app works without it
- DVC export is **manual** - run when you want to version dataset
- MLflow runs are stored in `./mlruns/` directory
- DVC data is stored in `./dvc_storage/` (configurable)
- All tracking is **non-blocking** - errors won't break the app

## 🐛 Troubleshooting

**MLflow not tracking:**
- Check if "Enable MLflow tracking" is checked
- Verify MLflow is installed: `pip install mlflow`
- Check `./mlruns/` directory exists and is writable

**DVC export fails:**
- Verify MongoDB connection string
- Check database name is correct
- Ensure collections exist

For detailed troubleshooting, see:
- `MLFLOW_DVC_GUIDE.md`
- `DVC_MONGODB_GUIDE.md`
