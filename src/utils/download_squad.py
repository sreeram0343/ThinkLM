import os
import json
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThinkLM.DownloadSQuAD")

def download_and_cache_squad(output_path: str = "data/anchor_set.json", num_samples: int = 50):
    logger.info("Starting download of SQuAD validation split...")
    try:
        # Load validation split of SQuAD v1.1
        dataset = load_dataset("squad", split="validation")
        logger.info(f"Successfully loaded SQuAD validation split. Total samples: {len(dataset)}")
        
        # Take num_samples
        subset = []
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            subset.append({
                "id": item["id"],
                "title": item["title"],
                "context": item["context"],
                "question": item["question"],
                "answers": {
                    "text": item["answers"]["text"],
                    "answer_start": item["answers"]["answer_start"]
                }
            })
            
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully saved {len(subset)} samples to {output_path}")
    except Exception as e:
        logger.error(f"Failed to download/save SQuAD: {e}")
        raise e

if __name__ == "__main__":
    # Base path should be relative to workspace or absolute
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_file = os.path.join(base_dir, "data", "anchor_set.json")
    download_and_cache_squad(output_file)
