"""
LogBERT-based Anomaly Detection for ICS/OT Logs
Supports multiple protocols: Modbus, DNP3, IEC 61850, BACnet, OPC UA
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import re
from collections import defaultdict
import json
from typing import List, Dict, Tuple
import pickle

class LogTokenizer:
    """Custom tokenizer for ICS/OT logs"""
    def __init__(self):
        self.vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4}
        self.token_to_id = self.vocab.copy()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.next_id = 5
        
        # ICS protocol keywords (omitted for brevity)
        self.protocol_keywords = {
            'modbus': ['READ_COILS', 'WRITE_COILS', 'READ_HOLDING', 'WRITE_REGISTER',
                       'FC01', 'FC02', 'FC03', 'FC04', 'FC05', 'FC06', 'FC15', 'FC16'],
            # ... rest of keywords
        }

    def parse_log(self, log_line: str) -> List[str]:
        """Parse log line into tokens"""
        # Extract timestamp
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'
        log_line = re.sub(timestamp_pattern, '[TIME]', log_line)

        # Extract IP addresses
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        log_line = re.sub(ip_pattern, '[IP]', log_line)

        # Extract hex values
        hex_pattern = r'0x[0-9a-fA-F]+'
        log_line = re.sub(hex_pattern, '[HEX]', log_line)

        # Extract numbers
        num_pattern = r'\b\d+\b'
        log_line = re.sub(num_pattern, '[NUM]', log_line)

        # Tokenize
        tokens = re.findall(r'\[?\w+\]?|[^\w\s]', log_line.lower())
        return tokens

    def fit(self, logs: List[str]):
        """Build vocabulary from logs"""
        for log in logs:
            tokens = self.parse_log(log)
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.next_id
                    self.id_to_token[self.next_id] = token
                    self.next_id += 1

    def encode(self, log_line: str, max_length: int = 128) -> Tuple[List[int], List[int]]:
        """Encode log line to token IDs"""
        tokens = self.parse_log(log_line)
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        # Convert to IDs
        token_ids = [self.token_to_id.get(t, self.token_to_id['[UNK]']) for t in tokens]

        # Padding
        attention_mask = [1] * len(token_ids)
        if len(token_ids) < max_length:
            padding_length = max_length - len(token_ids)
            token_ids += [self.token_to_id['[PAD]']] * padding_length
            attention_mask += [0] * padding_length
        else:
            token_ids = token_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        return token_ids, attention_mask

    def save(self, path: str):
        """Save tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'next_id': self.next_id
            }, f)

    def load(self, path: str):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.token_to_id = data['token_to_id']
            self.id_to_token = data['id_to_token']
            self.next_id = data['next_id']


class LogDataset(Dataset):
    """Dataset for log sequences"""
    def __init__(self, logs: List[str], tokenizer: LogTokenizer, max_length: int = 128):
        self.logs = logs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        log = self.logs[idx]
        token_ids, attention_mask = self.tokenizer.encode(log, self.max_length)

        # Create masked version for training (mask 15% of tokens)
        masked_ids = token_ids.copy()
        labels = [-100] * len(token_ids)  # -100 is ignored in loss

        for i in range(1, len(token_ids) - 1):  # Skip [CLS] and [SEP]
            if attention_mask[i] == 1 and np.random.random() < 0.15:
                labels[i] = token_ids[i]

                # 80% mask, 10% random, 10% keep
                rand = np.random.random()
                if rand < 0.8:
                    masked_ids[i] = self.tokenizer.token_to_id['[MASK]']
                elif rand < 0.9:
                    masked_ids[i] = np.random.randint(5, len(self.tokenizer.token_to_id))

        return {
            'input_ids': torch.tensor(masked_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'original_ids': torch.tensor(token_ids, dtype=torch.long)
        }


class LogBERT(nn.Module):
    """LogBERT model for anomaly detection"""
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 6,
                 num_heads: int = 8, max_length: int = 128):
        super().__init__()

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length
        )

        self.bert = BertModel(config)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Masked Language Model predictions
        mlm_logits = self.mlm_head(sequence_output)

        # Anomaly score (using [CLS] token)
        cls_output = sequence_output[:, 0, :]
        anomaly_score = self.anomaly_head(cls_output)

        return mlm_logits, anomaly_score


class LogBERTAnomalyDetector:
    """Main class for training and inference"""
    def __init__(self, max_length: int = 128, hidden_size: int = 256,
                 num_layers: int = 6, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.tokenizer = LogTokenizer()
        self.model = None
        self.threshold = None

    def train(self, normal_logs: List[str], epochs: int = 10, batch_size: int = 32, lr: float = 1e-4):
        """Train on normal logs"""
        print(f"Training on {len(normal_logs)} normal logs...")

        # Build vocabulary
        self.tokenizer.fit(normal_logs)
        print(f"Vocabulary size: {len(self.tokenizer.token_to_id)}")

        # Create dataset
        dataset = LogDataset(normal_logs, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        self.model = LogBERT(
            vocab_size=len(self.tokenizer.token_to_id),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            max_length=self.max_length
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                mlm_logits, _ = self.model(input_ids, attention_mask)

                loss = criterion(mlm_logits.view(-1, len(self.tokenizer.token_to_id)),
                               labels.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Calculate threshold
        self._calculate_threshold(normal_logs)
        print(f"Anomaly threshold set to: {self.threshold:.4f}")

    def _calculate_threshold(self, normal_logs: List[str]):
        """Calculate anomaly threshold from normal logs"""
        self.model.eval()
        scores = []

        with torch.no_grad():
            for log in normal_logs:
                score = self._get_anomaly_score(log)
                scores.append(score)

        # Set threshold at 95th percentile
        self.threshold = np.percentile(scores, 95)

    def _get_anomaly_score(self, log_line: str) -> float:
        """Calculate anomaly score for a single log"""
        token_ids, attention_mask = self.tokenizer.encode(log_line, self.max_length)

        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)

        with torch.no_grad():
            mlm_logits, anomaly_score = self.model(input_ids, attention_mask)

            # Calculate reconstruction error
            predicted_ids = torch.argmax(mlm_logits, dim=-1)
            original_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)

            mask = attention_mask == 1
            mismatch = (predicted_ids != original_ids) & mask
            reconstruction_error = mismatch.float().mean().item()

            # Combined score (0.5 * reconstruction error + 0.5 * anomaly detection head)
            combined_score = 0.5 * reconstruction_error + 0.5 * anomaly_score.item()

        return combined_score

    def detect_anomaly(self, log_line: str) -> Dict:
        """Detect if log is anomalous"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        score = self._get_anomaly_score(log_line)

        is_anomaly = score > self.threshold

        return {
            'log': log_line,
            'anomaly_score': float(score),
            'threshold': float(self.threshold),
            'is_anomaly': bool(is_anomaly),
            'confidence': float(abs(score - self.threshold) / self.threshold)
        }

    def batch_detect(self, logs: List[str]) -> List[Dict]:
        """Detect anomalies in batch"""
        return [self.detect_anomaly(log) for log in logs]

    def save_model(self, path: str):
        """Save model and tokenizer"""
        # Note: We save the state dict, which is safer and cleaner than saving the entire model.
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold': self.threshold,
            'max_length': self.max_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'vocab_size': len(self.tokenizer.token_to_id) # Save vocab size for loading
        }, f"{path}.pt")

        # Tokenizer is saved with the correct path
        self.tokenizer.save(f"{path}_tokenizer.pkl")
        print(f"Model saved to {path}.pt and {path}_tokenizer.pkl")

    def load_model(self, path: str):
        """Load model and tokenizer"""

        print(f"DEBUG: LogBERT loading requested path: {path}")

        # Load tokenizer first - this gives us the vocab size
        self.tokenizer.load(f"{path}_tokenizer.pkl")
        vocab_size_from_tokenizer = len(self.tokenizer.token_to_id)
        print(f"DEBUG: Tokenizer loaded, vocab_size={vocab_size_from_tokenizer}")

        # Load model weights
        checkpoint = torch.load(f"{path}.pt", map_location=self.device, weights_only=False)
        print(f"DEBUG: Checkpoint keys: {list(checkpoint.keys())}")

        # Extract checkpoint values with fallbacks for missing keys
        self.threshold = checkpoint.get('threshold', 0.5)
        self.max_length = checkpoint.get('max_length', 128)
        self.hidden_size = checkpoint.get('hidden_size', 256)
        self.num_layers = checkpoint.get('num_layers', 6)
        
        # Get vocab_size from checkpoint if available, otherwise from tokenizer
        vocab_size = checkpoint.get('vocab_size', vocab_size_from_tokenizer)
        print(f"DEBUG: Using vocab_size={vocab_size}, threshold={self.threshold}")

        # Initialize the model architecture before loading weights
        self.model = LogBERT(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            max_length=self.max_length
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"✓ Model loaded from {path}.pt")

# Example usage and sample data generator (omitted for brevity)

if __name__ == "__main__":
    # Example: Load and test the model
    detector = LogBERTAnomalyDetector()
    detector.load_model("logbert_ics_model")
    
    # Test detection
    test_log = "2024-01-01 12:00:00 MODBUS 192.168.1.1 -> 192.168.1.100 FC03 READ_HOLDING_REGISTERS"
    result = detector.detect_anomaly(test_log)
    print(f"Test result: {result}")
