"""
Quick script to check DVC dataset tracking status.
Run this to see what datasets are tracked and their status.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list) -> tuple[str, int]:
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip(), 0
    except subprocess.CalledProcessError as e:
        return e.stderr.strip(), e.returncode
    except FileNotFoundError:
        return "Command not found", 1


def check_dvc_status():
    """Check DVC tracking status."""
    print("=" * 60)
    print("DVC Dataset Tracking Status")
    print("=" * 60)
    
    # Check if DVC is initialized
    dvc_dir = Path(".dvc")
    if not dvc_dir.exists():
        print("❌ DVC not initialized. Run: dvc init")
        return
    
    print("✓ DVC initialized")
    print()
    
    # Check DVC status
    print("📊 DVC Status:")
    output, code = run_command(["dvc", "status"])
    if code == 0:
        if "nothing to commit" in output.lower() or not output:
            print("  ✓ All datasets up to date")
        else:
            print(f"  {output}")
    else:
        print(f"  ⚠️  {output}")
    print()
    
    # Find all .dvc files
    print("📁 Tracked Datasets:")
    dvc_files = list(Path(".").rglob("*.dvc"))
    if not dvc_files:
        print("  No datasets tracked yet.")
        print("  To track: dvc add <directory>")
    else:
        for dvc_file in sorted(dvc_files):
            rel_path = dvc_file.relative_to(Path("."))
            data_path = rel_path.with_suffix("")
            print(f"  • {data_path}")
            
            # Check if data exists
            if data_path.exists():
                size = sum(f.stat().st_size for f in data_path.rglob("*") if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"    Size: {size_mb:.2f} MB")
            else:
                print(f"    ⚠️  Data not found locally (run: dvc pull {data_path})")
    print()
    
    # Check remote storage
    print("💾 Remote Storage:")
    output, code = run_command(["dvc", "remote", "list"])
    if code == 0 and output:
        print(f"  {output}")
    else:
        print("  No remote configured")
        print("  To configure: dvc remote add -d local ./dvc_storage")
    print()
    
    # Check Git integration
    print("🔗 Git Integration:")
    output, code = run_command(["git", "status", "--porcelain"])
    if code == 0:
        dvc_changes = [line for line in output.split("\n") if ".dvc" in line]
        if dvc_changes:
            print("  ⚠️  Uncommitted .dvc files:")
            for change in dvc_changes:
                print(f"    {change}")
        else:
            print("  ✓ All .dvc files committed")
    print()
    
    print("=" * 60)
    print("Quick Commands:")
    print("  View status:     dvc status")
    print("  Pull dataset:    dvc pull <path>")
    print("  Push dataset:    dvc push")
    print("  View history:    git log --oneline --all -- *.dvc")
    print("=" * 60)


if __name__ == "__main__":
    check_dvc_status()
