"""
Fix checkpoint JSON files by replacing NaN with null
"""

import json
from pathlib import Path

def fix_checkpoint_file(file_path):
    """Fix a checkpoint JSON file by replacing NaN with null"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace NaN with null
        content = content.replace(': NaN', ': null')
        content = content.replace(':NaN', ': null')
        
        # Parse to validate
        data = json.loads(content)
        
        # Write back
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, allow_nan=False)
        
        print(f"Fixed: {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

if __name__ == "__main__":
    dataset_dir = Path('Dataset')
    
    # Fix all checkpoint files
    checkpoint_files = list(dataset_dir.glob('checkpoint_*.json'))
    
    if not checkpoint_files:
        print("No checkpoint files found")
    else:
        print(f"Found {len(checkpoint_files)} checkpoint file(s)")
        for file_path in checkpoint_files:
            fix_checkpoint_file(file_path)
        
        # Also fix main output file if it exists
        output_file = dataset_dir / 'nba_game_data.json'
        if output_file.exists():
            fix_checkpoint_file(output_file)
        
        print("\nAll files fixed!")

