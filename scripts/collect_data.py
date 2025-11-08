import json
from pathlib import Path

def collect_training_data():
    """Collect conversation data for potential retraining"""
    data_dir = Path(__file__).parent.parent / "data"
    backup_file = data_dir / "backup.json"

    if not backup_file.exists():
        return []

    with open(backup_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    # Convert to training format
    training_data = []
    for conv in conversations:
        if "prompt" in conv and "response" in conv:
            training_data.append({
                "prompt": conv["prompt"],
                "completion": conv["response"]
            })

    return training_data

if __name__ == "__main__":
    data = collect_training_data()
    print(f"Collected {len(data)} conversation pairs for potential training")