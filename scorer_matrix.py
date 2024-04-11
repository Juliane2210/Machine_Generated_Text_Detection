from format_checker import check_format
import logging.handlers
import argparse
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import sys

import os
import cv2
import shutil
import sys
import pandas as pd
from sklearn.metrics import precision_score, recall_score, mean_squared_error
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from skimage.feature import hog
from skimage import exposure
from skimage.feature import local_binary_pattern
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


sys.path.append('.')

"""
Scoring of SEMEVAL-Task-8--subtask-A-and-B  with the metrics f1-macro, f1-micro and accuracy. 
"""


def evaluate(pred_fpath, gold_fpath):
    """
      Evaluates the predicted classes w.r.t. a gold file.
      Metrics are: f1-macro, f1-micro and accuracy

      :param pred_fpath: a json file with predictions, 
      :param gold_fpath: the original annotated gold file.

      The submission of the result file should be in jsonl format. 
      It should be a lines of objects:
      {
        id     -> identifier of the test sample,
        labels -> labels (0 or 1 for subtask A and from 0 to 5 for subtask B),
      }
    """

    pred_labels = pd.read_json(pred_fpath, lines=True)[['id', 'label']]
    gold_labels = pd.read_json(gold_fpath, lines=True)[['id', 'label']]

    merged_df = pred_labels.merge(
        gold_labels, on='id', suffixes=('_pred', '_gold'))

    macro_f1 = f1_score(
        merged_df['label_gold'], merged_df['label_pred'], average="macro", zero_division=0)
    micro_f1 = f1_score(
        merged_df['label_gold'], merged_df['label_pred'], average="micro", zero_division=0)
    accuracy = accuracy_score(merged_df['label_gold'], merged_df['label_pred'])

    visualize_predictions_confusionMatrix(
        pred_fpath, merged_df['label_gold'], merged_df['label_pred'])
    return macro_f1, micro_f1, accuracy


def validate_files(pred_files):
    if not check_format(pred_files):
        logging.error(
            'Bad format for pred file {}. Cannot score.'.format(pred_files))
        return False
    return True


#
# Visualizations
# Helper functions that will be called to perform visualizations of model evaluation scores
#

def visualize_predictions_confusionMatrix(title, true_labels, predicted_labels):
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
                'human', 'machine'], yticklabels=[
                'human', 'machine'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title + ": Confusion Matrix")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file_path", '-g', type=str,
                        required=True, help="Paths to the file with gold annotations.")
    parser.add_argument("--pred_file_path", '-p', type=str,
                        required=True, help="Path to the file with predictions")
    args = parser.parse_args()

    pred_file_path = args.pred_file_path
    gold_file_path = args.gold_file_path

    if validate_files(pred_file_path):
        logging.info('Prediction file format is correct')
        macro_f1, micro_f1, accuracy = evaluate(pred_file_path, gold_file_path)
        logging.info(
            "macro-F1={:.5f}\tmicro-F1={:.5f}\taccuracy={:.5f}".format(macro_f1, micro_f1, accuracy))
