import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Reduce logging verbosity
logging.basicConfig(level=logging.WARNING)

# Base path configuration
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Paths configuration
PATHS = {
    'train_data': os.path.join(BASE_PATH, "data", "labelled", "train_set.xlsx"),
    'output_folder': os.path.join(BASE_PATH, "data", "output", "codebert"),  # Output directory
    'model_file': os.path.join(BASE_PATH, "data", "output", "codebert", "models", "best_codebert_model.pth"),  # Model path
}

# Create necessary directories
os.makedirs(os.path.join(BASE_PATH, "data", "output", "codebert", "models"), exist_ok=True)

class CodeBertTransformer(nn.Module):
    def __init__(
        self,
        service_num_labels,
        activity_num_labels,
        model_name="microsoft/codebert-base"
    ):
        super().__init__()

        # Load CodeBERT model
        self.transformer = AutoModel.from_pretrained(model_name)

        # Freeze initial layers
        for param in list(self.transformer.parameters())[:6]:
            param.requires_grad = False

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Advanced classification heads
        self.service_classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, service_num_labels)
        )

        self.activity_classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, activity_num_labels)
        )

    def forward(self, input_ids, attention_mask):
        # Efficient forward pass
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Classification
        service_pred = self.service_classifier(pooled_output)
        activity_pred = self.activity_classifier(pooled_output)

        return service_pred, activity_pred

class CodeBertDataset(Dataset):
    def __init__(
        self,
        texts,
        service_labels,
        activity_labels,
        tokenizer,
        max_length=128
    ):
        # Efficient tokenization
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        self.service_labels = torch.tensor(service_labels, dtype=torch.long)
        self.activity_labels = torch.tensor(activity_labels, dtype=torch.long)

    def __len__(self):
        return len(self.service_labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'service_label': self.service_labels[idx],
            'activity_label': self.activity_labels[idx]
        }

class CodeBertClassifier:
    def __init__(
        self,
        training_data_path,
        model_name="microsoft/codebert-base"
    ):
        # Efficient device selection
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Load data efficiently
        self.training_df = pd.read_excel(
            training_data_path,
             engine='openpyxl'

        )

        # Prepare data
        self._prepare_data(model_name)

    def _prepare_data(self, model_name):
        # Validate and clean data
        self.training_df['service'] = self.training_df['service'].fillna('Unknown')
        self.training_df['activityType'] = self.training_df['activityType'].fillna('Unknown')

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Label encoding
        self.service_encoder = LabelEncoder()
        self.activity_encoder = LabelEncoder()

        # Encode labels
        self.encoded_services = self.service_encoder.fit_transform(
            self.training_df['service']
        )
        self.encoded_activities = self.activity_encoder.fit_transform(
            self.training_df['activityType']
        )

        # Prepare text features with technical context
        self.texts = self.training_df.apply(
            self._prepare_text_features,
            axis=1
        )

    def _prepare_text_features(self, row):
        # Enhanced feature extraction with technical context
        technical_features = [
            str(row.get('url', '')),
            str(row.get('method', '')),
            str(row.get('headers_Host', '')),
            str(row.get('requestHeaders_Content_Type', '')),
            str(row.get('responseHeaders_Content_Type', ''))
        ]

        # Join features, limit length
        return " ".join(technical_features)[:512]

    def train(
        self,
        test_size=0.2,
        batch_size=32,
        epochs=20,
        learning_rate=2e-5
    ):
        # Split data
        (train_texts, val_texts,
         train_service_labels, val_service_labels,
         train_activity_labels, val_activity_labels) = train_test_split(
            self.texts,
            self.encoded_services,
            self.encoded_activities,
            test_size=test_size,
            random_state=42
        )

        # Create datasets
        train_dataset = CodeBertDataset(
            train_texts,
            train_service_labels,
            train_activity_labels,
            self.tokenizer
        )
        val_dataset = CodeBertDataset(
            val_texts,
            val_service_labels,
            val_activity_labels,
            self.tokenizer
        )

        # DataLoaders with optimization
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2
        )

        # Model initialization
        service_num_labels = len(self.service_encoder.classes_)
        activity_num_labels = len(self.activity_encoder.classes_)

        model = CodeBertTransformer(
            service_num_labels,
            activity_num_labels
        ).to(self.device)

        # Loss and optimizer
        service_criterion = nn.CrossEntropyLoss()
        activity_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2
        )

        # Training loop
        best_val_accuracy = 0
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                service_labels = batch['service_label'].to(self.device)
                activity_labels = batch['activity_label'].to(self.device)

                service_pred, activity_pred = model(
                    input_ids, attention_mask
                )

                service_loss = service_criterion(
                    service_pred, service_labels
                )
                activity_loss = activity_criterion(
                    activity_pred, activity_labels
                )

                total_loss = service_loss + activity_loss
                total_loss.backward()
                optimizer.step()

                total_train_loss += total_loss.item()

            # Validation phase
            model.eval()
            val_service_preds, val_activity_preds = [], []
            val_service_true, val_activity_true = [], []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    service_pred, activity_pred = model(
                        input_ids, attention_mask
                    )

                    val_service_preds.extend(
                        torch.argmax(service_pred, dim=1).cpu().numpy()
                    )
                    val_activity_preds.extend(
                        torch.argmax(activity_pred, dim=1).cpu().numpy()
                    )

                    val_service_true.extend(batch['service_label'].numpy())
                    val_activity_true.extend(batch['activity_label'].numpy())

            # Accuracy calculation
            service_accuracy = np.mean(
                np.array(val_service_preds) == np.array(val_service_true)
            )
            activity_accuracy = np.mean(
                np.array(val_activity_preds) == np.array(val_activity_true)
            )

            print(f"Epoch {epoch+1}: "
                  f"Service Accuracy: {service_accuracy:.4f}, "
                  f"Activity Accuracy: {activity_accuracy:.4f}")

            # Update learning rate
            scheduler.step(service_accuracy + activity_accuracy)

            # Save best model
            current_accuracy = service_accuracy + activity_accuracy
            if current_accuracy > best_val_accuracy:
                best_val_accuracy = current_accuracy
                torch.save(model.state_dict(), PATHS['model_file'])
                print("Saved best model")

        return model

def main():
    try:
        # Initialize and train classifier
        classifier = CodeBertClassifier(PATHS['train_data'])
        model = classifier.train()

        print("Training completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()