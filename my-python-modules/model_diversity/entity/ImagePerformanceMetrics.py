# ImagePerformanceMetrics Class
# This class computes the performance metrics from predictions of one image

# Importing python libraries 
import math                         
import pandas as pd
import numpy as np
import torch 
import torchvision
from torchvision.ops import *  

# Importing python modules
from common.manage_log import *

class ImagePerformanceMetrics:
    def __init__(self, dataset_name=None, model_name=None, image_name=None,
                 classes=None, number_of_classes=None,
                 iou_threshold=None, ground_truths=None, predictions=None,
                 true_positive=None, false_positive=None, true_negative=None, false_negative=None, 
                 classes_precision=None, classes_recall=None, classes_f1_score=None                 ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.image_name = image_name
        self.classes = classes
        self.number_of_classes = number_of_classes
        self.iou_threshold = iou_threshold
        self.ground_truths = ground_truths
        self.predictions = predictions

        self.confusion_matrix = None
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.true_negative = true_negative
        self.false_negative = false_negative
        self.classes_precision = classes_precision
        self.classes_recall = classes_recall
        self.classes_f1_score = classes_f1_score

        self.number_of_images = None
        self.number_of_bounding_boxes_target = None
        self.number_of_bounding_boxes_predicted = None
        self.number_of_bounding_boxes_predicted_with_target = None
        self.number_of_background_predictions = None
        self.number_of_undetected_objects = None

        # contain data of all image bounding boxes of ground truths and predictions of the image
        self.image_bounding_boxes = []


    def compute_metrics(self):

        # logging_info(f'Computing performance metrics: of the image: {self.image_name}')
        # logging_info(f'Dataset: {self.dataset_name}')
        # logging_info(f'Image: {self.image_name}')

        # print(f'')
        # print(f'Computing performance metrics: of the image: {self.image_name}')
        # print(f'Dataset: {self.dataset_name}')
        # print(f'Model: {self.model_name}')
        # print(f'Image: {self.image_name}')

        # preparing confusion matrix of the image 
        self.confusion_matrix = np.zeros((self.number_of_classes + 2, self.number_of_classes + 2))

        # setting row index of "undetected objects (false negatives)" and "background "
        undetected_objects_fn_index = background_fp_index = self.number_of_classes + 1

        # setting counters 
        self.number_of_bounding_boxes_target = 0
        self.number_of_bounding_boxes_predicted = 0        
        self.number_of_bounding_boxes_predicted_with_target = 0
        self.number_of_background_predictions = 0
        self.number_of_undetected_objects = 0

        # logging_info(f'image ground_truths: {self.ground_truths}')
        # logging_info(f'image predictions {self.predictions}')

        # preparing ground truths and predictions 
        ground_truths_boxes = self.ground_truths['boxes']
        ground_truths_labels = self.ground_truths['labels']
        self.number_of_bounding_boxes_target += len(ground_truths_boxes)

        if len(self.predictions) == 0: 
            predictions_boxes = []
            predictions_labels = []
            predictions_scores = []
        else:
            predictions_boxes = self.predictions['boxes']
            predictions_labels = self.predictions['labels']
            predictions_scores = self.predictions['scores']

        # initializing variables
        matched_gt = set()

        # if self.image_name == "ds-2023-09-07-santa-helena-de-goias-go-fazenda-sete-ilhas-pivo-04-IMG_3868-bbox-1527539646.jpg":
        #     print(f'self.image_name 1: {self.image_name}')
        #     print(f'self.ground_truths 1: {self.ground_truths}')
        #     print(f'self.predictions 1: {self.predictions}')                 
        
        # computing iou of the model bbox and groudn truth bbox 
        for pred_ind, (p_bbox, p_label, p_score) in enumerate(zip(predictions_boxes, predictions_labels, predictions_scores)):

            self.number_of_bounding_boxes_predicted += 1

            # initializing variables
            best_iou = 0
            best_gt_ind = -1
            
            for gt_ind, (t_box, t_label) in enumerate(zip(ground_truths_boxes, ground_truths_labels)):

                # computing IoU of two boxes
                tensor_p_bbox = torch.as_tensor([p_bbox]) # add one new dimension
                tensor_gt_bbox = torch.as_tensor([t_box]) # add one new dimension

                # Both sets of boxes are expected to be in (x1, y1, x2, y2)
                # evaluate IoU threshold and labels
                iou = box_iou(tensor_p_bbox, tensor_gt_bbox).numpy()[0][0]    
                # if self.image_name == "ds-2023-09-07-santa-helena-de-goias-go-fazenda-sete-ilhas-pivo-04-IMG_3868-bbox-1527539646.jpg":
                #     print(f'IoU for {self.image_name}: {iou}')
                #     print(f'self.iou_threshold: {self.iou_threshold}')
                #     print(f'best_iou: {best_iou}')
                #     print(f'Prediction: {p_bbox}, {p_label}, {p_score}')
                #     print(f'Ground Truth: {t_box}, {t_label}')

                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_ind = gt_ind

            # if best_iou is found, add to confusion matrix
            if best_gt_ind >= 0:
                self.number_of_bounding_boxes_predicted_with_target += 1
                matched_gt.add(best_gt_ind)
                best_t_label = ground_truths_labels[best_gt_ind]

                # evaluating of the prediction and ground truth labels to set TP or FP
                if p_label == best_t_label:
                    # True Positive 
                    self.confusion_matrix[best_t_label, p_label] += 1
                    status = 'target-tp'
                else:                        
                    # False Positive 
                    self.confusion_matrix[best_t_label, p_label] += 1
                    status = 'target-fp'

                # adding bounding box details 
                self.add_image_bounding_box(self.image_name,
                                            ground_truths_boxes[best_gt_ind], ground_truths_labels[best_gt_ind],
                                            p_bbox, p_label, p_score, 0, best_iou, self.iou_threshold, status, self.model_name)

            else:
                # Counting prediction as background FP using the prediction label
                self.number_of_background_predictions += 1                         
                self.confusion_matrix[p_label, background_fp_index] += 1
                status = 'err-pred-fp'

                # adding bounding box details 
                self.add_image_bounding_box(self.image_name, ground_truths_boxes, ground_truths_labels,
                                            p_bbox, p_label, p_score, 0, 0, self.iou_threshold, status, self.model_name)
        
        # counting undetected objects (false negatives)
        for gt_ind, t_label in enumerate(ground_truths_labels):
            if gt_ind not in matched_gt:
                self.number_of_undetected_objects += 1
                self.confusion_matrix[undetected_objects_fn_index, t_label] += 1
                status='undetected-fn'  

                # adding bounding box details 
                self.add_image_bounding_box(self.image_name,
                                            ground_truths_boxes[gt_ind], ground_truths_labels[gt_ind],
                                            [], [], [], 0, 0, self.iou_threshold, status, self.model_name)


    # def compute_metrics_backup_15_08_2025(self):

    #     # logging_info(f'Computing performance metrics: of the image: {self.image_name}')
    #     # logging_info(f'Dataset: {self.dataset_name}')
    #     # logging_info(f'Image: {self.image_name}')

    #     # print(f'')
    #     # print(f'Computing performance metrics: of the image: {self.image_name}')
    #     # print(f'Dataset: {self.dataset_name}')
    #     # print(f'Model: {self.model_name}')
    #     # print(f'Image: {self.image_name}')

    #     # preparing confusion matrix of the image 
    #     self.confusion_matrix = np.zeros((self.number_of_classes + 2, self.number_of_classes + 2))

    #     # setting row index of "undetected objects (false negatives)" and "background "
    #     undetected_objects_fn_index = background_fp_index = self.number_of_classes + 1

    #     # setting counters 
    #     self.number_of_bounding_boxes_target = 0
    #     self.number_of_bounding_boxes_predicted = 0        
    #     self.number_of_bounding_boxes_predicted_with_target = 0
    #     self.number_of_background_predictions = 0
    #     self.number_of_undetected_objects = 0

    #     # logging_info(f'image ground_truths: {self.ground_truths}')
    #     # logging_info(f'image predictions {self.predictions}')

    #     # preparing ground truths and predictions 
    #     ground_truths_boxes = self.ground_truths['boxes']
    #     ground_truths_labels = self.ground_truths['labels']
    #     ground_truths_has_tp = []
    #     for label in ground_truths_labels:
    #         ground_truths_has_tp.append(False)
    #     self.number_of_bounding_boxes_target += len(ground_truths_boxes)

    #     if len(self.predictions) == 0: 
    #         predictions_boxes = []
    #         predictions_labels = []
    #         predictions_scores = []
    #     else:
    #         predictions_boxes = self.predictions['boxes']
    #         predictions_labels = self.predictions['labels']
    #         predictions_scores = self.predictions['scores']

    #     if self.image_name in 'MVI_9431_380.jpg':
    #         print(f'')
    #         print(f'Rubens - self.image_name: {self.image_name}')                
    #         print(f'ground_truths_boxes: {ground_truths_boxes}')
    #         print(f'ground_truths_labels: {ground_truths_labels}')
    #         print(f'ground_truths_has_tp: {ground_truths_has_tp}')
    #         print(f'predictions_boxes: {predictions_boxes}')
    #         print(f'predictions_labels: {predictions_labels}')
    #         print(f'predictions_scores: {predictions_scores}')
    #         print(f'')



    #     # initializing variables
    #     matched_gt = set()
        
    #     # computing iou of the model bbox and groudn truth bbox 
    #     for pred_ind, (p_bbox, p_label, p_score) in enumerate(zip(predictions_boxes, predictions_labels, predictions_scores)):
    #         if self.image_name in 'MVI_9431_380.jpg':
    #             print(f'')
    #             print(f'Rubens - self.image_name: {self.image_name}')                
    #             print(f'Rubens - p_box: {p_bbox}  p_label: {p_label}  p_score: {p_score}')
    #             print(f'')

    #         # logging_info(f'predictions rubens: {index}')
    #         self.number_of_bounding_boxes_predicted += 1

    #         # print(f'')
    #         best_iou = 0
    #         best_gt_idx = -1
            
    #         for gt_ind, (t_box, t_label) in enumerate(zip(ground_truths_boxes, ground_truths_labels)):

    #             if self.image_name in 'MVI_9431_380.jpg':
    #                 print(f'Rubens - processing GT t_box: {t_box}  t_label: {t_label}')
    #                 print(f'')

    #             # if ground truth is processed, next ground truth
    #             if gt_ind in matched_gt:
    #                 if self.image_name in 'MVI_9431_380.jpg':
    #                     print(f'Rubens - image - gt_ind: {gt_ind} and matched_gt: {matched_gt}')
    #                     print(f'descartou GT t_box: {t_box}  t_label: {t_label}')
    #                     print(f'')
    #                 continue

    #             if self.image_name in 'MVI_9431_380.jpg':
    #                 print(f'Rubens - image - gt_ind: {gt_ind} and matched_gt: {matched_gt}')
    #                 print(f'vai comparar GT t_box: {t_box}  t_label: {t_label}')
    #                 print(f'')

    #             # computing IoU of two boxes
    #             tensor_p_bbox = torch.as_tensor([p_bbox]) # add one new dimension
    #             tensor_gt_bbox = torch.as_tensor([t_box]) # add one new dimension

    #             # Both sets of boxes are expected to be in (x1, y1, x2, y2)
    #             # evaluate IoU threshold and labels
    #             iou = box_iou(tensor_p_bbox, tensor_gt_bbox).numpy()[0][0]
    #             # print(f'prediction: {p_bbox}    label: {p_label}    score: {p_score}')
    #             # print(f'ground truth: {t_box}    label: {t_label}')
    #             # print(f'iou: {iou}    iou_threshold: {self.iou_threshold}')
    #             # logging_info(f'iou: {iou}    iou_threshold: {self.iou_threshold}')
    #             # logging_info(f'p_label: {p_label}    t_label: {t_label}')
    #             if self.image_name in 'MVI_9431_380.jpg':
    #                 print(f'Rubens iou: {iou}    iou_threshold: {self.iou_threshold}')
    #                 print(f'')
                
    #             if iou >= self.iou_threshold and iou > best_iou:
    #                 best_iou = iou
    #                 best_gt_idx = gt_ind

    #         # if best_iou is found, add to confusion matrix
    #         if best_gt_idx >= 0:
    #             self.number_of_bounding_boxes_predicted_with_target += 1
    #             matched_gt.add(best_gt_idx)

    #             # evaluating of the prediction and ground truth labels to set TP or FP
    #             if p_label == t_label:
    #                 # True Positive 
    #                 self.confusion_matrix[t_label, p_label] += 1
    #                 status = 'target-tp'
    #             else:                        
    #                 # False Positive 
    #                 self.confusion_matrix[t_label, p_label] += 1
    #                 status = 'target-fp'
    #         else:
    #             # Counting prediction as background FP
    #             self.number_of_background_predictions += 1                         
    #             self.confusion_matrix[t_label, background_fp_index] += 1
    #             status = 'err-pred-fp'

    #         # adding bounding box details 
    #         self.add_image_bounding_box(self.image_name,
    #                                     ground_truths_boxes[best_gt_idx], ground_truths_labels[best_gt_idx],
    #                                     p_bbox, p_label, p_score, 0, best_iou, self.iou_threshold, status, self.model_name)
        
    #     # counting undetected objects (false negatives)
    #     for gt_ind, t_label in enumerate(ground_truths_labels):
    #         if gt_ind not in matched_gt:
    #             self.number_of_undetected_objects += 1
    #             self.confusion_matrix[undetected_objects_fn_index, t_label] += 1
    #             status='undetected-fn'  

    #             # adding bounding box details 
    #             self.add_image_bounding_box(self.image_name,
    #                                         ground_truths_boxes[gt_ind], ground_truths_labels[gt_ind],
    #                                         [], [], [], 0, 0, self.iou_threshold, status, self.model_name)

    #     # logging_info(f'Image confusion matrix: {self.confusion_matrix}')
    #     # print(f'Image confusion matrix: \n {self.confusion_matrix}')
    #     # print(f'self.number_of_images: {self.number_of_images}')
    #     # print(f'self.number_of_bounding_boxes_target: {self.number_of_bounding_boxes_target}')
    #     # print(f'self.number_of_bounding_boxes_predicted: {self.number_of_bounding_boxes_predicted}')
    #     # print(f'self.number_of_bounding_boxes_predicted_with_target: {self.number_of_bounding_boxes_predicted_with_target}')
    #     # print(f'self.number_of_ghost_predictions: {self.number_of_background_predictions}')
    #     # print(f'self.number_of_undetected_objects: {self.number_of_undetected_objects}')
    #     if self.image_name in 'MVI_9431_380.jpg':
    #         print(f'Rubens Processing image: {self.image_name}')
    #         print(f'self.image_bounding_boxes: {self.image_bounding_boxes}')


    # def compute_metrics_old_version(self):

    #     # logging_info(f'Computing performance metrics: of the image: {self.image_name}')
    #     # logging_info(f'Dataset: {self.dataset_name}')
    #     # logging_info(f'Image: {self.image_name}')

    #     print(f'Computing performance metrics: of the image: {self.image_name}')
    #     print(f'Dataset: {self.dataset_name}')
    #     print(f'Model: {self.model_name}')
    #     print(f'Image: {self.image_name}')

    #     # preparing confusion matrix of the image 
    #     self.confusion_matrix = np.zeros((self.number_of_classes + 2, self.number_of_classes + 2))

    #     # setting row index of "undetected objects (false negatives)" and "background "
    #     undetected_objects_fn_index = background_fp_index = self.number_of_classes + 1

    #     # setting counters 
    #     self.number_of_bounding_boxes_target = 0
    #     self.number_of_bounding_boxes_predicted = 0        
    #     self.number_of_bounding_boxes_predicted_with_target = 0
    #     self.number_of_ghost_predictions = 0
    #     self.number_of_undetected_objects = 0

    #     logging_info(f'image ground_truths: {self.ground_truths}')
    #     logging_info(f'image predictions {self.predictions}')

    #     # preparing ground truths and predictions 
    #     ground_truths_boxes = self.ground_truths['boxes']
    #     ground_truths_labels = self.ground_truths['labels']
    #     self.number_of_bounding_boxes_target += len(ground_truths_boxes)

    #     # if self.model_name == "consensus":
    #     #     logging_info(f'rubens predictions: {len(self.predictions)} - {self.predictions}')

    #     if len(self.predictions) == 0: 
    #         predictions_boxes = []
    #         predictions_labels = []
    #         predictions_scores = []
    #     else:
    #         predictions_boxes = self.predictions['boxes']
    #         predictions_labels = self.predictions['labels']
    #         predictions_scores = self.predictions['scores']

    #     # evaluating image predictions       
    #     if len(predictions_boxes) == 0:
    #         # counting undetected objects (false negatives)
    #         # number_of_false_negatives = len(self.ground_truths['labels'])
    #         self.number_of_undetected_objects += 1
    #         for gt_label_index in ground_truths_labels:
    #             self.confusion_matrix[undetected_objects_fn_index, gt_label_index] += 1

    #     else:
    #         # counting predictions against of ground truths
            
    #         # computing iou of the model bbox and groudn truth bbox 
    #         # for i, (model_bbox, model_label) in enumerate(zip(model_bboxes, model_labels)):
    #         for index, (p_bbox, p_label, p_score) in enumerate(zip(predictions_boxes, predictions_labels, predictions_scores)):
    #             # logging_info(f'predictions rubens: {index}')
    #             self.number_of_bounding_boxes_predicted += 1

    #             print(f'')

    #             # number_of_bounding_boxes_predicted += 1
    #             for t_box, t_label in zip(ground_truths_boxes, ground_truths_labels):

    #                 # computing IoU of two boxes
    #                 tensor_p_bbox = torch.as_tensor([p_bbox]) # add one new dimension
    #                 tensor_gt_bbox = torch.as_tensor([t_box]) # add one new dimension

    #                 # Both sets of boxes are expected to be in (x1, y1, x2, y2)
    #                 # iou = torchvision.ops.box_iou(p_box.unsqueeze(0), t_box.unsqueeze(0))
    #                 # iou = torchvision.ops.box_iou(p_box, t_box)

    #                 # logging_info(f'tensor_gt_bbox: {tensor_gt_bbox}')
    #                 # logging_info(f'tensor_p_bbox: {tensor_p_bbox}')
    #                 iou = box_iou(tensor_p_bbox, tensor_gt_bbox).numpy()[0][0]
    #                 print(f'prediction: {p_bbox}    label: {p_label}    score: {p_score}')
    #                 print(f'ground truth: {t_box}    label: {t_label}')
    #                 print(f'iou: {iou}    iou_threshold: {self.iou_threshold}')
    #                 # logging_info(f'iou: {iou}    iou_threshold: {self.iou_threshold}')
    #                 # logging_info(f'p_label: {p_label}    t_label: {t_label}')

    #                 # evaluate IoU threshold and labels
    #                 if iou >= self.iou_threshold:
    #                     self.number_of_bounding_boxes_predicted_with_target += 1                      
    #                     if p_label == t_label:
    #                         # True Positive 
    #                         self.confusion_matrix[t_label, p_label] += 1
    #                         # status = 'target-tp'
    #                     else:                        
    #                         # False Positive 
    #                         self.confusion_matrix[t_label, p_label] += 1
    #                         # status = 'target-fp'
    #                 else:
    #                     # Counting ghost predictions   
    #                     self.number_of_ghost_predictions += 1                         
    #                     self.confusion_matrix[t_label, background_fp_index] += 1
    #                     # status = 'err-pred-fp'

    #                 # setting model name 
    #                 # logging_info(f'pred rubens: {pred}')
    #                 # if 'model' in pred:
    #                 #     logging_info(f'rubens pred: {pred}')
    #                 #     logging_info(f'rubens pred[model]: {pred["model"]}')
    #                 #     prediction_model_name = pred['model']
    #                 # else:
    #                 #     prediction_model_name = ''

    #     # logging_info(f'Image confusion matrix: {self.confusion_matrix}')
    #     print(f'Image confusion matrix: \n {self.confusion_matrix}')
    #     print(f'self.number_of_images: {self.number_of_images}')
    #     print(f'self.number_of_bounding_boxes_target: {self.number_of_bounding_boxes_target}')
    #     print(f'self.number_of_bounding_boxes_predicted: {self.number_of_bounding_boxes_predicted}')
    #     print(f'self.number_of_bounding_boxes_predicted_with_target: {self.number_of_bounding_boxes_predicted_with_target}')
    #     print(f'self.number_of_ghost_predictions: {self.number_of_ghost_predictions}')
    #     print(f'self.number_of_undetected_objects: {self.number_of_undetected_objects}')

    def add_image_bounding_box(self, image_name, 
        ground_truth_bbox=None, ground_truth_label=None, 
        predicted_bbox=None, predicted_label=None, predicted_score=None, 
        score_threshold=None, iou=None, iou_threshold=None, status=None, model_name=None):

        # adding one bounding box of an image 
        image_bounding_box = []
        image_bounding_box.append(image_name)
        image_bounding_box.append(ground_truth_bbox)
        image_bounding_box.append(ground_truth_label)
        image_bounding_box.append(predicted_bbox)
        image_bounding_box.append(predicted_label)
        image_bounding_box.append(predicted_score)
        image_bounding_box.append(score_threshold)
        image_bounding_box.append(iou)
        image_bounding_box.append(iou_threshold)
        image_bounding_box.append(status)        
        image_bounding_box.append(model_name)
        self.image_bounding_boxes.append(image_bounding_box)
