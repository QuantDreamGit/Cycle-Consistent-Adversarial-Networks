from typing import List, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import nn, Tensor
import torch


class DiscriminatorModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pretrained_path: str = None,
        num_domains: int = 2,
        max_seq_length: int = 64,
        truncation: str = "longest_first",
        padding: str = "max_length",
    ):
        """
        Initialize the Discriminator for StarGAN.
        Args:
            model_name_or_path (str): Path to the pretrained model or its identifier.
            pretrained_path (str): Path to the fine-tuned weights (if any).
            num_domains (int): Number of possible target domains.
            max_seq_length (int): Maximum sequence length for inputs.
            truncation (str): Truncation strategy for tokenization.
            padding (str): Padding strategy for tokenization.
        """
        super(DiscriminatorModel, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding
        self.num_domains = num_domains

        # Base model for encoding
        if pretrained_path is None:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_path}tokenizer/")

        # Replace classifier with a new multi-task output:
        self.validity_head = nn.Linear(self.base_model.config.hidden_size, 1)  # Binary classification
        self.domain_head = nn.Linear(self.base_model.config.hidden_size, num_domains)  # Domain classification

    def train(self):
        """Set model to training mode."""
        self.base_model.train()
        self.validity_head.train()
        self.domain_head.train()

    def eval(self):
        """Set model to evaluation mode."""
        self.base_model.eval()
        self.validity_head.eval()
        self.domain_head.eval()

    def forward(
        self,
        sentences: List[str],
        validity_labels: Tensor = None,
        domain_labels: Tensor = None,
        device=None,
    ):
        """
        Forward pass of the discriminator.
        Args:
            sentences (List[str]): Input sentences.
            validity_labels (Tensor, optional): Labels for validity (real/fake).
            domain_labels (Tensor, optional): Labels for domain classification.
            device: Torch device (CPU/GPU).
        Returns:
            Tuple containing:
                - Validity output (real/fake classification).
                - Domain classification output.
                - Losses for validity and domain classification (if labels are provided).
        """
        # Tokenize input sentences
        inputs = self.tokenizer(
            sentences,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(device)

        # Get hidden states from the base model
        outputs = self.base_model.bert(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden_states = outputs.last_hidden_state[:, 0, :]  # CLS token representation

        # Compute predictions
        validity_logits = self.validity_head(hidden_states)  # Real/Fake logits
        domain_logits = self.domain_head(hidden_states)  # Domain classification logits

        # Loss calculation
        validity_loss = None
        domain_loss = None
        if validity_labels is not None:
            validity_loss_fn = nn.BCEWithLogitsLoss()
            validity_loss = validity_loss_fn(validity_logits.squeeze(), validity_labels.float())

        if domain_labels is not None:
            domain_loss_fn = nn.CrossEntropyLoss()
            domain_loss = domain_loss_fn(domain_logits, domain_labels)

        return {
            "validity_logits": validity_logits,
            "domain_logits": domain_logits,
            "validity_loss": validity_loss,
            "domain_loss": domain_loss,
        }

    def save_model(self, path: Union[str]):
        """
        Save the model and tokenizer.
        Args:
            path (str): Path to save the model.
        """
        self.base_model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer/")