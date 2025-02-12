import numpy as np
import pandas as pd
import random
import pickle

import os

import torch
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stargan_tst.models import DiscriminatorModel

class Evaluator():

    def __init__(self, starGAN, args, experiment):
        """ Class for evaluation """
        super(Evaluator, self).__init__()

        self.starGAN = starGAN
        self.args = args
        self.experiment = experiment

        self.bleu = evaluate.load('sacrebleu')
        self.rouge = evaluate.load('rouge')
        if args.bertscore: self.bertscore = evaluate.load('bertscore')
    

    def __compute_metric__(self, predictions, references, metric_name, direction=None):
        # predictions = list | references = list of lists
        scores = []
        if metric_name in ['bleu', 'rouge', 'bertscore']:
            for pred, ref in zip(predictions, references):
                if metric_name == 'bleu':
                    res = self.bleu.compute(predictions=[pred], references=[ref])
                    scores.append(res['score'])
                elif metric_name == 'rouge':
                    tmp_rouge1, tmp_rouge2, tmp_rougeL = [], [], []
                    for r in ref:
                        res = self.rouge.compute(predictions=[pred], references=[r], use_aggregator=False)
                        tmp_rouge1.append(res['rouge1'][0])
                        tmp_rouge2.append(res['rouge2'][0])
                        tmp_rougeL.append(res['rougeL'][0])
                    scores.append([max(tmp_rouge1), max(tmp_rouge2), max(tmp_rougeL)])
                elif metric_name == 'bertscore':
                    res = self.bertscore.compute(predictions=[pred], references=[ref], lang=self.args.lang)
                    scores.extend(res['f1'])
        else:
            raise Exception(f"Metric {metric_name} is not supported.")
        return scores
    

    def __compute_classif_metrics__(self, pred_sentences, true_domains):
        if len(pred_sentences)!=len(true_domains):
            print("Error size in input")

        device = self.starGAN.device
        truncation, padding = 'longest_first', 'max_length'
        
        # Verifica se usare il classificatore pre-addestrato o il discriminatore
        if 'lambdas' not in vars(self.args) or self.args.lambdas[4] == 0:
            # Classificatore pre-addestrato
            classifier = AutoModelForSequenceClassification.from_pretrained(self.args.pretrained_classifier_eval)
            classifier_tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_classifier_eval)
            classifier.to(device)
        else:
            # Usa il discriminatore
            classifier = self.starGAN.D  # DiscriminatorModel
            classifier_tokenizer = classifier.tokenizer
        
        classifier.eval()

        y_pred, y_true = [], true_domains

        # Elaborazione batch
        for i in range(0, len(pred_sentences), self.args.batch_size):
            batch_sentences = pred_sentences[i:i + self.args.batch_size]
            
            # Tokenizzazione delle frasi in batch
            inputs = classifier_tokenizer(
                batch_sentences,
                truncation=truncation,
                padding=padding,
                max_length=self.args.max_sequence_length,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                # Forward pass
                if isinstance(classifier, DiscriminatorModel.DiscriminatorModel):
                    # Usa il discriminatore
                    outputs = classifier(sentences=batch_sentences, device=device)
                    domain_logits = outputs["domain_logits"]
                else:
                    # Usa il classificatore pre-addestrato
                    outputs = classifier(**inputs)
                    domain_logits = outputs.logits  # Logits per la classificazione del dominio
                
                # Calcolo delle predizioni
                batch_predictions = torch.argmax(domain_logits, dim=1).cpu().tolist()
                y_pred.extend(batch_predictions)
        
        # Convertire y_true in una lista se non lo è già
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        elif isinstance(y_true, list):
            y_true = np.array(y_true)  # Per compatibilità con sklearn
        y_pred = np.array(y_pred)  # y_pred è già su CPU in formato lista

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, prec, rec, f1

    def run_eval_no_ref(self, epoch, current_training_step, phase, dataset):
        """
        Evaluates the model on a single dataset structured as 
        (sentence, original_style_code, new_style_code).
        """
        print(f'Start {phase}...')
        self.starGAN.eval()  # set evaluation mode

        if self.args.comet_logging:
            if phase == 'validation':
                context = self.experiment.validate
            elif phase == 'test':
                context = self.experiment.test

        real_sentences, pred_sentences, all_target_styles = [], [], []
        scores_bleu_self, scores_r1_self, scores_r2_self, scores_rL_self = [], [], [], []

        for batch in dataset:
            sentences, source_styles, target_styles = batch

            with torch.no_grad():
                transferred = self.starGAN.transfer(sentences=sentences, target_styles=target_styles)
            
            real_sentences.extend(sentences)
            pred_sentences.extend(transferred)
            all_target_styles.extend(target_styles)
            
            # Prepare for metric computation
            references = [[s] for s in sentences]
            scores_bleu_self.extend(self.__compute_metric__(transferred, references, 'bleu'))
            scores_rouge_self = np.array(self.__compute_metric__(transferred, references, 'rouge'))
            scores_r1_self.extend(scores_rouge_self[:, 0].tolist())
            scores_r2_self.extend(scores_rouge_self[:, 1].tolist())
            scores_rL_self.extend(scores_rouge_self[:, 2].tolist())

        # Calculate averages for the metrics
        avg_bleu_self = np.mean(scores_bleu_self)
        avg_r1_self, avg_r2_self, avg_rL_self = (
            np.mean(scores_r1_self),
            np.mean(scores_r2_self),
            np.mean(scores_rL_self),
        )

        # Calculate accuracy metrics
        acc, _, _, _ = self.__compute_classif_metrics__(real_sentences, all_target_styles)
        acc_scaled = acc * 100
        avg_acc_bleu_self = (avg_bleu_self + acc_scaled) / 2
        avg_acc_bleu_self_geom = (avg_bleu_self * acc_scaled) ** 0.5
        avg_acc_bleu_self_h = 2 * avg_bleu_self * acc_scaled / (avg_bleu_self + acc_scaled + 1e-6)

        # Save the metrics
        metrics = {
            'epoch': epoch,
            'step': current_training_step,
            'self-BLEU avg': avg_bleu_self,
            'self-ROUGE-1 avg': avg_r1_self,
            'self-ROUGE-2 avg': avg_r2_self,
            'self-ROUGE-L avg': avg_rL_self,
            'style accuracy': acc,
            'acc-BLEU': avg_acc_bleu_self,
            'g-acc-BLEU': avg_acc_bleu_self_geom,
            'h-acc-BLEU': avg_acc_bleu_self_h,
        }

        # Determine file paths
        base_path = f"{self.args.save_base_folder}epoch_{epoch}/"
        if phase == 'validation':
            suffix = f'epoch{epoch}'
            if self.args.eval_strategy == 'epochs':
                suffix += f'_step{current_training_step}'
        else:
            base_path = f"{self.args.save_base_folder}test/epoch_{epoch}/"
            suffix = f'epoch{epoch}_test'
        
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        pickle.dump(metrics, open(f"{base_path}metrics_{suffix}.pickle", 'wb'))

        for m, v in metrics.items():
            if m not in ['epoch', 'step']:
                print(f'{m}: {v}')

        # Save results in CSV format
        df = pd.DataFrame()
        df['Source Sentence'] = real_sentences
        df['Generated Sentence'] = pred_sentences
        df.to_csv(f"{base_path}results_{suffix}.csv", sep=',', header=True)

        # Log to Comet if enabled
        if self.args.comet_logging:
            with context():
                self.experiment.log_table(f'./results_{suffix}.csv', tabular_data=df, headers=True)
                for m, v in metrics.items():
                    if m not in ['epoch', 'step']:
                        self.experiment.log_metric(m, v, step=current_training_step, epoch=epoch)
        
        del df
        print(f'End {phase}...')

    
    def run_eval_ref(self, epoch, current_training_step, phase, parallel_dl_eval):
        print(f'Start {phase}...')
        self.starGAN.eval()  # Imposta il modello in modalità valutazione

        if self.args.comet_logging:
            if phase == 'validation':
                context = self.experiment.validate
            elif phase == 'test':
                context = self.experiment.test

        real_sentences, pred_sentences, ref_sentences = [], [], []
        scores_bleu_self, scores_bleu_ref = [], []
        scores_r1_self, scores_r2_self, scores_rL_self = [], [], []
        scores_r1_ref, scores_r2_ref, scores_rL_ref = [], [], []
        scores_bscore = []

        for batch in parallel_dl_eval:
            original_sentences = list(batch[0])  
            original_styles = list(batch[1])  
            references = list(batch[2])  # List(str)
            
            # If needed, eliminate uppercase letters
            if self.args.lowercase_ref:
                references = [[ref.lower() for ref in refs] for refs in references]
            
            # Randomly select a target style for each sentence
            target_styles = [random.choice([s for s in range(len(references[0])) if s != orig]) for orig in original_styles]
            
            # Save the corresponding reference phrase 
            selected_references = [references[i][target_styles[i]] for i in range(len(original_sentences))]
            
            with torch.no_grad():
                transferred = self.starGAN.transfer(sentences=original_sentences, target_styles=target_styles)
            
            real_sentences.extend(original_sentences)
            pred_sentences.extend(transferred)
            ref_sentences.extend(selected_references)
            
            # Compute the metrics
            real_wrapped = [[s] for s in original_sentences]  
            scores_bleu_self.extend(self.__compute_metric__(transferred, real_wrapped, 'bleu'))
            scores_bleu_ref.extend(self.__compute_metric__(transferred, selected_references, 'bleu'))
            scores_rouge_self = np.array(self.__compute_metric__(transferred, real_wrapped, 'rouge'))
            scores_r1_self.extend(scores_rouge_self[:, 0].tolist())
            scores_r2_self.extend(scores_rouge_self[:, 1].tolist())
            scores_rL_self.extend(scores_rouge_self[:, 2].tolist())
            scores_rouge_ref = np.array(self.__compute_metric__(transferred, selected_references, 'rouge'))
            scores_r1_ref.extend(scores_rouge_ref[:, 0].tolist())
            scores_r2_ref.extend(scores_rouge_ref[:, 1].tolist())
            scores_rL_ref.extend(scores_rouge_ref[:, 2].tolist())
            
            if self.args.bertscore:
                scores_bscore.extend(self.__compute_metric__(transferred, selected_references, 'bertscore'))
            else:
                scores_bscore.extend([0] * len(transferred))

        # Compute the average values
        avg_bleu_self, avg_bleu_ref = np.mean(scores_bleu_self), np.mean(scores_bleu_ref)
        avg_bleu_geom = (avg_bleu_self * avg_bleu_ref) ** 0.5
        avg_r1_self, avg_r2_self, avg_rL_self = np.mean(scores_r1_self), np.mean(scores_r2_self), np.mean(scores_rL_self)
        avg_r1_ref, avg_r2_ref, avg_rL_ref = np.mean(scores_r1_ref), np.mean(scores_r2_ref), np.mean(scores_rL_ref)
        avg_bscore = np.mean(scores_bscore)

        metrics = {
            'epoch': epoch, 'step': current_training_step,
            'self-BLEU': avg_bleu_self, 'ref-BLEU': avg_bleu_ref,
            'g-BLEU': avg_bleu_geom,
            'self-ROUGE-1': avg_r1_self, 'self-ROUGE-2': avg_r2_self, 'self-ROUGE-L': avg_rL_self,
            'ref-ROUGE-1': avg_r1_ref, 'ref-ROUGE-2': avg_r2_ref, 'ref-ROUGE-L': avg_rL_ref,
            'BERTScore': avg_bscore
        }

        if phase == 'test':
            acc, prec, rec, f1 = self.__compute_classif_metrics__(pred_sentences)
            metrics.update({'style accuracy': acc, 'style precision': prec, 'style recall': rec, 'style F1 score': f1})

        base_path = f"{self.args.save_base_folder}{'test/' if phase == 'test' else f'epoch_{epoch}/'}"
        suffix = f"epoch{epoch}{'_test' if phase == 'test' else ''}"
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        pickle.dump(metrics, open(f"{base_path}metrics_{suffix}.pickle", 'wb'))

        for m, v in metrics.items():
            if m not in ['epoch', 'step']:
                print(f'{m}: {v}')

        df = pd.DataFrame({'Original': real_sentences, 'Generated': pred_sentences, 'Reference': ref_sentences})
        df.to_csv(f"{base_path}translations_{suffix}.csv", sep=',', header=True)

        if self.args.comet_logging:
            with context():
                self.experiment.log_table(f'./translations_{suffix}.csv', tabular_data=df, headers=True)
                for m, v in metrics.items():
                    if m not in ['epoch', 'step']:
                        self.experiment.log_metric(m, v, step=current_training_step, epoch=epoch)

        del df
        print(f'End {phase}...')

    
