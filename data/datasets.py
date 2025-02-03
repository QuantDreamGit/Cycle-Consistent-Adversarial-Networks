import csv
from typing import List
import logging
import random

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, transform=None, max_samples=None, target_label_fn=None, n_styles=2):
        """
        Args:
            file_path (str): Percorso del file contenente le frasi e le classi.
            transform (callable, optional): Funzione di trasformazione da applicare alle frasi.
            max_samples (int, optional): Numero massimo di campioni da includere nel dataset.
            target_label_fn (callable, optional): Funzione per calcolare lo stile di trasformazione.
        """
        self.data = []
        self.transform = transform
        self.additional_int_fn = target_label_fn  # Funzione per calcolare il terzo valore
        
        # Legge il file e salva le righe come tuple (frase, classe, int aggiuntivo)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                phrase, label = line.strip().split('[ENDSTRING]')
                source_style = int(label)  # Converte la classe in un intero
                target_style = target_label_fn(source_style, n_styles) if target_label_fn else 0
                self.data.append((phrase, source_style, target_style))
                
                # Interrompi la lettura se il limite di campioni è raggiunto
                if max_samples is not None and len(self.data) >= max_samples:
                    break

    def __len__(self):
        # Restituisce il numero di campioni nel dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Restituisce il campione corrispondente all'indice idx
        phrase, source_style, target_style = self.data[idx]
        if self.transform:
            phrase = self.transform(phrase)
        return phrase, source_style, target_style


class ParallelDataset(Dataset):
    def __init__(self, validation_file, style_files, max_dataset_samples=None):
        self.sentences = []
        self.styles = []
        self.parallel_sentences = [[] for _ in style_files]
        
        # Load the validation file
        with open(validation_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_dataset_samples is not None and len(self.sentences) >= max_dataset_samples:
                    break
                sentence, style_id = line.rsplit("['ENDSTRING']", 1)
                self.sentences.append(sentence.strip())
                self.styles.append(int(style_id.strip()))
        
        # Load the style files
        for style_idx, file_path in enumerate(style_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_dataset_samples is not None and i >= max_dataset_samples:
                        break
                    self.parallel_sentences[style_idx].append(line.strip())
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        original_sentence = self.sentences[idx]
        original_style = self.styles[idx]
        parallel_sentences = [self.parallel_sentences[style_idx][idx] for style_idx in range(len(self.parallel_sentences))]
        return original_sentence, original_style, parallel_sentences
    
    @staticmethod
    def customCollate(batch):
        original_sentences, original_styles, parallel_sentences = zip(*batch)
        return list(original_sentences), list(original_styles), list(parallel_sentences)
