import torch
from torch import nn
from typing import List, Union

from stargan_tst.models.GeneratorModel import GeneratorModel
from stargan_tst.models.DiscriminatorModel import DiscriminatorModel

import logging


class StarGANModel(nn.Module):
    def __init__(
        self,
        G: Union[GeneratorModel, None],
        D: Union[DiscriminatorModel, None],
        device=None,
    ):
        """Initialization method for the StarGANModel

        Args:
            G (:obj:`GeneratorModel`): Generator model
            D (:obj:`DiscriminatorModel`): Discriminator model
        """
        super(StarGANModel, self).__init__()

        if G is None or D is None:
            logging.warning("StarGANModel: Models are not provided, please invoke 'load_models' to initialize them from a checkpoint")

        self.G = G
        self.D = D
        self.device = device
        logging.info(f"Device: {device}")

        # Move models to the device
        self.G.model.to(self.device)
        self.D.base_model.to(self.device)
        self.D.validity_head.to(self.device)
        self.D.domain_head.to(self.device)

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def get_optimizer_parameters(self):
        return list(self.G.model.parameters()) + list(self.D.model.parameters())

    def training_step(
        self,
        sentences: List[str],
        source_styles: List[int],
        target_styles: List[int],
        lambdas: List[float],
        comet_experiment=None,
        loss_logging=None,
        training_step=None,
    ):
        """
        Perform one training step for the StarGAN.

        Args:
            sentences: Input sentences
            domains: Source style indices for the sentences
            lambdas: List of lambda values for weighting the losses
        """

        # ----------------- Generator Step -----------------
        self.D.eval()  # Freeze the discriminator during generator training
        
        # Generate transformed sentences
        transferred_sentences, _ = self.G(sentences, target_styles, device=self.device)

        # Adversarial loss for generator (fooling the discriminator)
        zeros = torch.zeros(len(transferred_sentences))
        ones = torch.ones(len(transferred_sentences))
        labels_fake_sentences = torch.column_stack((ones, zeros))  # Class index 0 = Fake
        _, loss_g_adv = self.D(transferred_sentences, validity_labels=labels_fake_sentences, device=self.device)

        # Style classification loss for generator
        target_styles_tensor = torch.tensor(target_styles, dtype=torch.long, device=self.device)
        _, loss_g_style = self.D(transferred_sentences, domain_labels=target_styles_tensor, device=self.device)

        # Cycle-consistency loss (optional)
        reconstructed_sentences, _, loss_cycle = self.G(transferred_sentences, source_styles, device=self.device)

        # Total generator loss
        total_loss_g = lambdas[0] * loss_g_adv + lambdas[1] * loss_g_style + lambdas[2] * loss_cycle
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Generator Adversarial Loss", lambdas[0] * loss_g_adv, step=training_step)
                comet_experiment.log_metric("Generator Style Loss", lambdas[1] * loss_g_style, step=training_step)
                comet_experiment.log_metric("Cycle Consistency Loss", lambdas[2] * loss_cycle, step=training_step)
        loss_logging["Generator Adversarial Loss"].append(lambdas[0] * loss_g_adv.item())
        loss_logging["Generator Style Loss"].append(lambdas[1] * loss_g_style.item())
        loss_logging["Cycle Consistency Loss"].append(lambdas[2] * loss_cycle.item())

        # Backward pass for generator
        total_loss_g.backward()

        # ----------------- Discriminator Step -----------------
        self.D.train()  # Train the discriminator

        # Discriminator loss for real sentences
        labels_real_sentences = torch.column_stack((ones, zeros))  # Class index 0 = Real
        _, loss_d_real = self.D(sentences, validity_labels=labels_real_sentences, device=self.device)

        # Discriminator loss for fake sentences
        labels_fake_sentences = torch.column_stack((zeros, ones))  # Class index 1 = Fake
        _, loss_d_fake = self.D(transferred_sentences.detach(), validity_labels=labels_fake_sentences, device=self.device)

        # Style classification loss for discriminator
        _, loss_d_style = self.D(sentences, domain_labels=source_styles, device=self.device)

        # Total discriminator loss
        total_loss_d = lambdas[3] * (loss_d_real + loss_d_fake) + lambdas[4] * loss_d_style
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Discriminator Real/Fake Loss", lambdas[3] * (loss_d_real + loss_d_fake), step=training_step)
                comet_experiment.log_metric("Discriminator Style Loss", lambdas[4] * loss_d_style, step=training_step)
        loss_logging["Discriminator Real/Fake Loss"].append(lambdas[3] * (loss_d_real + loss_d_fake).item())
        loss_logging["Discriminator Style Loss"].append(lambdas[4] * loss_d_style.item())

        # Backward pass for discriminator
        total_loss_d.backward()

    def save_models(self, base_path: Union[str]):
        """Save the models."""
        self.G.save_model(base_path + "/G/")
        self.D.save_model(base_path + "/D/")

    def transfer(self, sentences: List[str], target_styles: List[int]):
        """Perform style transfer."""
        transferred_sentences = self.G.transfer(sentences, target_styles, device=self.device)
        return transferred_sentences
