"""
Export MongoDB dataset for DVC versioning.

This script exports face images and attendance records from MongoDB
to a directory structure that can be tracked with DVC.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("Warning: pymongo not installed. Install with: pip install pymongo")


def export_face_images(
    connection_string: str,
    database_name: str,
    output_dir: Path,
    collection_name: str = "face_images",
) -> Dict[str, int]:
    """Export face images from MongoDB to directory structure."""
    if not MONGODB_AVAILABLE:
        raise ImportError("pymongo not installed")
    
    client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
    db = client[database_name]
    collection = db[collection_name]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total_images": 0, "identities": set()}
    
    # Group by identity name
    for doc in collection.find():
        name = doc.get("name", "unknown")
        image_index = doc.get("image_index", 0)
        image_bytes = doc.get("image_bytes")
        
        if image_bytes:
            # Create identity directory
            identity_dir = output_dir / name
            identity_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image
            image_path = identity_dir / f"{name}_{image_index}.jpg"
            with open(image_path, "wb") as f:
                f.write(bytes(image_bytes))
            
            stats["total_images"] += 1
            stats["identities"].add(name)
    
    stats["num_identities"] = len(stats["identities"])
    stats["identities"] = list(stats["identities"])
    return stats


def export_attendance_records(
    connection_string: str,
    database_name: str,
    output_file: Path,
    collection_name: str = "attendance",
) -> int:
    """Export attendance records from MongoDB to JSON file."""
    if not MONGODB_AVAILABLE:
        raise ImportError("pymongo not installed")
    
    client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
    db = client[database_name]
    collection = db[collection_name]
    
    records = []
    for doc in collection.find():
        # Convert ObjectId and datetime to strings
        record = {
            "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None,
            "entered_name": doc.get("entered_name", ""),
            "matched_identity": doc.get("matched_identity", ""),
            "distance": float(doc.get("distance", 0.0)),
        }
        records.append(record)
    
    # Sort by timestamp
    records.sort(key=lambda x: x["timestamp"] or "")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)
    
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Export MongoDB data for DVC versioning")
    parser.add_argument(
        "--connection-string",
        default="mongodb://localhost:27017/",
        help="MongoDB connection string",
    )
    parser.add_argument(
        "--database",
        default="face_attendance",
        help="MongoDB database name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mongodb_export"),
        help="Output directory for exported data",
    )
    parser.add_argument(
        "--export-images",
        action="store_true",
        help="Export face images",
    )
    parser.add_argument(
        "--export-attendance",
        action="store_true",
        help="Export attendance records",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export both images and attendance",
    )
    
    args = parser.parse_args()
    
    if not MONGODB_AVAILABLE:
        print("Error: pymongo not installed. Install with: pip install pymongo")
        return
    
    try:
        # Test connection
        client = MongoClient(args.connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        print(f"✓ Connected to MongoDB: {args.database}")
    except ConnectionFailure as e:
        print(f"Error: Failed to connect to MongoDB: {e}")
        return
    
    export_all = args.export_all or (not args.export_images and not args.export_attendance)
    
    if export_all or args.export_images:
        print("\nExporting face images...")
        try:
            stats = export_face_images(
                args.connection_string,
                args.database,
                args.output_dir / "images",
            )
            print(f"✓ Exported {stats['total_images']} images for {stats['num_identities']} identities")
            print(f"  Identities: {', '.join(stats['identities'])}")
        except Exception as e:
            print(f"Error exporting images: {e}")
    
    if export_all or args.export_attendance:
        print("\nExporting attendance records...")
        try:
            count = export_attendance_records(
                args.connection_string,
                args.database,
                args.output_dir / "attendance.json",
            )
            print(f"✓ Exported {count} attendance records")
        except Exception as e:
            print(f"Error exporting attendance: {e}")
    
    # Create metadata file
    metadata = {
        "export_timestamp": datetime.now().isoformat(),
        "mongodb_connection": args.connection_string,
        "database": args.database,
        "exported_by": "export_mongodb_for_dvc.py",
    }
    metadata_file = args.output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Created metadata file: {metadata_file}")
    
    print(f"\n✓ Export complete! Data saved to: {args.output_dir}")
    print("\nNext steps:")
    print(f"  1. Review exported data in: {args.output_dir}")
    print(f"  2. Add to DVC: dvc add {args.output_dir}")
    print(f"  3. Commit: git add {args.output_dir}.dvc && git commit -m 'Add MongoDB export'")


if __name__ == "__main__":
    main()
