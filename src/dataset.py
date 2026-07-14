import json
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

LABEL_MAP = {
    "false": 0,
    "TrAP": 1,
    "TrWP": 2,
    "TrCP": 3,
    "TrIP": 4,
    "TrNAP": 5,
    "TeRP": 6,
    "TeCP": 7,
    "PIP": 8
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

class N2C2Dataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_seq_length: int = 128, is_train: bool = True, language: str = None):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_train = is_train
        
        # Auto-detect language from file path if not explicitly provided
        if language is None:
            if "_vi" in os.path.basename(file_path):
                self.language = "vi"
            else:
                self.language = "en"
        else:
            self.language = language
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        self._load_and_preprocess(file_path)

    def _load_and_preprocess(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                sentence = data.get(f"{self.language}_sentence_str", "")
                entities = data.get("entities", [])
                relations = data.get("relations", [])
                
                # Create a map from entity ID to entity dict for quick access
                ent_map = {ent["id"]: ent for ent in entities}
                
                # Relational maps: (ent1_id, ent2_id) -> relation_label
                rel_map = {}
                for rel in relations:
                    sub_id = rel["subject_id"]
                    obj_id = rel["object_id"]
                    rel_label = rel["label"]
                    rel_map[(sub_id, obj_id)] = rel_label
                    rel_map[(obj_id, sub_id)] = rel_label # undirected matching for candidate generation

                # Valid entity labels for candidate pairs:
                # - Treatment & Problem
                # - Test & Problem
                # - Problem & Problem
                valid_types = {"Treatment", "Problem", "Test"}
                
                # Generate candidate pairs
                num_entities = len(entities)
                for i in range(num_entities):
                    for j in range(i + 1, num_entities):
                        ent1 = entities[i]
                        ent2 = entities[j]
                        
                        t1 = ent1["label"]
                        t2 = ent2["label"]
                        
                        if t1 not in valid_types or t2 not in valid_types:
                            continue
                            
                        # Filter candidate types
                        is_valid_pair = False
                        if (t1 == "Treatment" and t2 == "Problem") or (t1 == "Problem" and t2 == "Treatment"):
                            is_valid_pair = True
                        elif (t1 == "Test" and t2 == "Problem") or (t1 == "Problem" and t2 == "Test"):
                            is_valid_pair = True
                        elif t1 == "Problem" and t2 == "Problem":
                            is_valid_pair = True
                            
                        if not is_valid_pair:
                            continue
                            
                        # Get relation label if exists
                        label = "false"
                        pair_key = (ent1["id"], ent2["id"])
                        if pair_key in rel_map:
                            label = rel_map[pair_key]
                            
                        # Perform entity marking
                        marked_sentence = self._mark_entities(sentence, ent1, ent2)
                        
                        self.examples.append({
                            "text": marked_sentence,
                            "label": LABEL_MAP.get(label, 0),
                            "metadata": {
                                "sentence_id": line_idx,
                                "ent1_id": ent1["id"],
                                "ent2_id": ent2["id"],
                                "ent1_term": ent1[f"{self.language}_term"],
                                "ent2_term": ent2[f"{self.language}_term"],
                                "original_relation": label
                            }
                        })

    def _mark_entities(self, sentence: str, ent1: dict, ent2: dict) -> str:
        words = sentence.split()
        
        s1, e1 = ent1[f"{self.language}_start_token_idx"], ent1[f"{self.language}_end_token_idx"]
        s2, e2 = ent2[f"{self.language}_start_token_idx"], ent2[f"{self.language}_end_token_idx"]
        
        # Determine order of entities in sentence to insert tags correctly without index shifting problems
        if s1 < s2:
            first_ent = (s1, e1, "[unused1]", "[unused2]")
            second_ent = (s2, e2, "[unused3]", "[unused4]")
        else:
            first_ent = (s2, e2, "[unused1]", "[unused2]")
            second_ent = (s1, e1, "[unused3]", "[unused4]")
            
        # Reconstruct sentence with tags
        new_words = []
        
        # Segment 1: before first entity
        new_words.extend(words[:first_ent[0]])
        # First entity with tags
        new_words.append(first_ent[2])
        new_words.extend(words[first_ent[0]:first_ent[1] + 1])
        new_words.append(first_ent[3])
        # Segment 2: between first and second entity
        new_words.extend(words[first_ent[1] + 1:second_ent[0]])
        # Second entity with tags
        new_words.append(second_ent[2])
        new_words.extend(words[second_ent[0]:second_ent[1] + 1])
        new_words.append(second_ent[3])
        # Segment 3: after second entity
        new_words.extend(words[second_ent[1] + 1:])
        
        return " ".join(new_words)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Squeeze to remove batch dimension added by return_tensors="pt"
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example["label"], dtype=torch.long)
        }
        
        # Token type ids might not be present for all models (e.g., RoBERTa)
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
            
        return item
