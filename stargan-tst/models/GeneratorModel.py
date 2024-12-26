from typing import List, Union
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class GeneratorModel(nn.Module):
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
        Initialize the Generator for StarGAN.
        Args:
            model_name_or_path (str): Path to the pretrained model or its identifier.
            pretrained_path (str): Path to the fine-tuned weights (if any).
            num_domains (int): Number of possible target domains.
            max_seq_length (int): Maximum sequence length for inputs and outputs.
            truncation (str): Truncation strategy for tokenization.
            padding (str): Padding strategy for tokenization.
        """
        super(GeneratorModel, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding
        self.num_domains = num_domains

        # Load tokenizer and model
        if pretrained_path is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_path)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_path}tokenizer/")

        # Embedding layer to condition on target domains
        self.domain_embedding = nn.Embedding(num_domains, self.model.config.d_model)

    def train(self):
        """Set model to training mode."""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def forward(
        self,
        sentences: List[str],
        target_domains: List[int],
        target_sentences: List[str] = None,
        device=None,
    ):
        """
        Forward pass of the generator.
        Args:
            sentences (List[str]): Input sentences.
            target_domains (List[int]): List of target domain indices.
            target_sentences (List[str], optional): Ground-truth sentences for supervised training.
            device: Torch device (CPU/GPU).
        Returns:
            output: Generated token IDs.
            transferred_sentences: List of generated sentences.
            loss (if target_sentences is provided): Supervised loss.
        """
        # Tokenize input sentences
        inputs = self.tokenizer(
            sentences,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(device)

        # Generate domain embeddings
        domain_indices = torch.tensor(target_domains, dtype=torch.long, device=device)
        domain_embeds = self.domain_embedding(domain_indices)

        # Integrate domain embeddings into encoder input
        encoder_inputs = self.model.get_encoder()(inputs["input_ids"], return_dict=True)
        encoder_outputs = encoder_inputs.last_hidden_state + domain_embeds.unsqueeze(1)

        if target_sentences is not None:
            # Tokenize target sentences for supervised learning
            with self.tokenizer.as_target_tokenizer():
                target = self.tokenizer(
                    target_sentences,
                    truncation=self.truncation,
                    padding=self.padding,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(device)
            labels = target["input_ids"]

            # Generate output and compute supervised loss
            outputs = self.model(
                inputs_embeds=encoder_outputs,
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss

            # Generate text
            generated_ids = self.model.generate(inputs_embeds=encoder_outputs, max_length=self.max_seq_length)
            transferred_sentences = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_ids]

            return generated_ids, transferred_sentences, loss

        else:
            # Generate text without computing loss
            generated_ids = self.model.generate(inputs_embeds=encoder_outputs, max_length=self.max_seq_length)
            transferred_sentences = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_ids]

            return generated_ids, transferred_sentences

    def transfer(self, sentences: List[str], target_domains: List[int], device=None):
        """
        Transfer sentences to the specified target domain.
        Args:
            sentences (List[str]): Input sentences.
            target_domains (List[int]): Target domain indices.
            device: Torch device (CPU/GPU).
        Returns:
            List of generated sentences.
        """
        inputs = self.tokenizer(
            sentences,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(device)

        domain_indices = torch.tensor(target_domains, dtype=torch.long, device=device)
        domain_embeds = self.domain_embedding(domain_indices)

        encoder_inputs = self.model.get_encoder()(inputs["input_ids"], return_dict=True)
        encoder_outputs = encoder_inputs.last_hidden_state + domain_embeds.unsqueeze(1)

        generated_ids = self.model.generate(inputs_embeds=encoder_outputs, max_length=self.max_seq_length)
        transferred_sentences = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_ids]

        return transferred_sentences

    def save_model(self, path: Union[str]):
        """
        Save the model and tokenizer.
        Args:
            path (str): Path to save the model.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer/")