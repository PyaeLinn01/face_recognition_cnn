# DVC for MongoDB Dataset - Guide

This guide explains how to use DVC to version control your MongoDB dataset.

## Overview

Since MongoDB is a database, we can't directly track it with DVC. Instead, we export the data to a directory structure that DVC can track.

## Quick Start

### 1. Export MongoDB Data

```bash
# Export all data (images + attendance)
python export_mongodb_for_dvc.py --export-all

# Or export separately
python export_mongodb_for_dvc.py --export-images
python export_mongodb_for_dvc.py --export-attendance

# Custom connection and output
python export_mongodb_for_dvc.py \
  --connection-string "mongodb://localhost:27017/" \
  --database "face_attendance" \
  --output-dir "data/mongodb_export" \
  --export-all
```

### 2. Add to DVC

```bash
# Initialize DVC if not already done
dvc init

# Add exported data to DVC
dvc add data/mongodb_export/

# Commit to Git
git add data/mongodb_export.dvc .gitignore
git commit -m "Add MongoDB dataset export"
```

### 3. Push to Remote Storage

```bash
# Configure remote (if not already done)
dvc remote add -d local ./dvc_storage

# Push data
dvc push
```

## Workflow

### Regular Export and Versioning

```bash
# 1. Export current MongoDB state
python export_mongodb_for_dvc.py --export-all

# 2. Update DVC tracking
dvc add data/mongodb_export/

# 3. Commit changes
git add data/mongodb_export.dvc
git commit -m "Update MongoDB dataset - $(date +%Y%m%d)"

# 4. Push to storage
dvc push
```

### Restore Dataset

```bash
# Pull from DVC storage
dvc pull data/mongodb_export

# Import back to MongoDB (if needed)
# Use your own import script or MongoDB tools
```

## Automated Export Script

Create a script `scripts/export_for_dvc.sh`:

```bash
#!/bin/bash
# Export MongoDB and add to DVC

python export_mongodb_for_dvc.py --export-all

dvc add data/mongodb_export/
git add data/mongodb_export.dvc
git commit -m "Auto-export MongoDB dataset - $(date +%Y%m%d_%H%M%S)"
dvc push
```

Make it executable:
```bash
chmod +x scripts/export_for_dvc.sh
```

## Integration with CI/CD

Add to your CI/CD pipeline:

```yaml
# .github/workflows/export-dataset.yml
name: Export MongoDB Dataset

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:

jobs:
  export:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Export MongoDB
        env:
          MONGODB_CONNECTION_STRING: ${{ secrets.MONGODB_CONNECTION_STRING }}
        run: |
          python export_mongodb_for_dvc.py --export-all
      - name: Add to DVC
        run: |
          dvc add data/mongodb_export/
          git config user.name "CI"
          git config user.email "ci@example.com"
          git add data/mongodb_export.dvc
          git commit -m "Auto-export dataset"
          git push
      - name: Push to DVC storage
        run: dvc push
```

## Best Practices

1. **Regular Exports**: Export MongoDB data regularly (daily/weekly)
2. **Meaningful Commits**: Use descriptive commit messages with dates
3. **Backup Storage**: Use cloud storage (S3, GCS) for DVC remote
4. **Metadata**: The export script creates metadata.json with export info
5. **Version Tags**: Tag important dataset versions in Git

## File Structure

After export:
```
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

## Troubleshooting

### Export Fails
- Check MongoDB connection string
- Verify database name
- Ensure collections exist

### DVC Add Fails
- Check disk space
- Verify output directory exists
- Check file permissions

### Push Fails
- Verify remote storage configuration
- Check credentials for cloud storage
- Ensure network connectivity

## Next Steps

1. Set up automated exports (cron job or CI/CD)
2. Configure cloud storage for DVC remote
3. Create import script to restore from DVC export
4. Integrate with MLflow to track dataset versions
