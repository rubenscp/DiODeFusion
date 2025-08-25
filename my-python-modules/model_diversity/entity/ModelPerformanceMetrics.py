# ModelPerformanceMetrics Class
# This class computes the model performance metrics from predictions of the object detection model

# Importing python libraries 
# import math
import pandas as pd
import json 

# Importing python modules
from common.manage_log import *
from model_diversity.entity.ImagePerformanceMetrics import * 

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class ModelPerformanceMetrics:
    def __init__(self, dataset_name=None, model_name=None, 
                 classes=None, number_of_classes=None,
                 iou_threshold=None, predictions=None
                 ):
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.classes = classes
        self.number_of_classes = number_of_classes
        self.iou_threshold = iou_threshold
        self.predictions = predictions

        self.image_performance_metrics_list = None 
        self.confusion_matrix = None
        self.normalized_confusion_matrix = None
        self.confusion_matrix_summary = {
            'number_of_images': 0,
            'number_of_bounding_boxes_target': 0,
            'number_of_bounding_boxes_predicted': 0,
            'number_of_bounding_boxes_predicted_with_target': 0,
            'number_of_ghost_predictions': 0,
            'number_of_undetected_objects': 0,
        }
        
        # self.classes_precision = classes_precision
        # self.classes_recall = classes_recall
        # self.classes_f1_score = classes_f1_score
        # self.model_precision = model_precision
        # self.model_recall = model_recall
        # self.model_f1_score = model_f1_score

        self.tp_per_class = []
        self.fp_per_class = []
        self.fn_per_class = []
        self.tn_per_class = []
        self.tp_model = 0
        self.fp_model = 0
        self.fn_model = 0
        self.tn_model = 0
        self.accuracy_per_class = []
        self.precision_per_class = []
        self.recall_per_class = []
        self.f1_score_per_class = []
        self.dice_per_class = []

    # preparing object to add ground truth and predictions per image 
    def prepare_performance_metric(self, input_dataset_type):
        self.predictions = {
            "model_name": self.model_name, 
            "input_dataset_type": input_dataset_type,
            "number_of_images": 0,
            "number_of_predictions": 0,
            "images": {}
        }      
    
    # adding one image with its ground truths and predictions 
    def add_image(self, image_filename, ground_truths, predictions):
        self.predictions["images"][image_filename] = {}
        self.predictions["images"][image_filename]["ground_truths"] = ground_truths
        self.predictions["images"][image_filename]["predictions"] = predictions

    # computes the performance metrics for the model
    def compute_metrics(self):

        # preparing images list 
        self.image_performance_metrics_list = []

        # looping predictions to compute performance metrics
        for image_name_key, value in self.predictions["images"].items():
            ground_truths = value['ground_truths']
            predictions = value ['predictions']
            # logging_info(f'Processing prediction of image {image_name_key}')
            # logging_info(f'ground_truths: {ground_truths}')
            # logging_info(f'predictions {predictions}')

            image_performance_metrics = ImagePerformanceMetrics()
            image_performance_metrics.dataset_name = self.dataset_name
            image_performance_metrics.model_name = self.model_name
            image_performance_metrics.image_name = image_name_key
            image_performance_metrics.classes = self.classes
            image_performance_metrics.number_of_classes = self.number_of_classes
            image_performance_metrics.iou_threshold = self.iou_threshold
            image_performance_metrics.ground_truths = ground_truths
            image_performance_metrics.predictions = predictions
            image_performance_metrics.true_positive = 0
            image_performance_metrics.false_positive = 0
            image_performance_metrics.true_negative = 0
            image_performance_metrics.false_negative = 0
            image_performance_metrics.classes_precision = 0
            image_performance_metrics.classes_recall = 0
            image_performance_metrics.classes_f1_score = 0

            # computing performance metrics for the image predictions
            image_performance_metrics.compute_metrics()

            # adding image performance metrics
            self.image_performance_metrics_list.append(image_performance_metrics)

            # summary of confusion matrix        
            self.confusion_matrix_summary["number_of_images"] += 1
            self.confusion_matrix_summary["number_of_bounding_boxes_target"] += image_performance_metrics.number_of_bounding_boxes_target         
            self.confusion_matrix_summary["number_of_bounding_boxes_predicted"] += image_performance_metrics.number_of_bounding_boxes_predicted
            self.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"] += image_performance_metrics.number_of_bounding_boxes_predicted_with_target
            self.confusion_matrix_summary["number_of_ghost_predictions"] += image_performance_metrics.number_of_background_predictions
            self.confusion_matrix_summary["number_of_undetected_objects"] += image_performance_metrics.number_of_undetected_objects


        # computing performance metrics for the model
        self.confusion_matrix = np.zeros((self.number_of_classes + 2, self.number_of_classes + 2))
        for image_performance_metrics in self.image_performance_metrics_list:
            for i in range(self.number_of_classes + 2):
                for j in range(self.number_of_classes + 2):
                    self.confusion_matrix[i][j] += image_performance_metrics.confusion_matrix[i][j]

        # logging_info(f'')
        # logging_info(f'Dataset: {self.dataset_name} - Model: {self.model_name}')
        # logging_info(f'Model confusion matrix: {self.confusion_matrix}')
        # logging_info(f'')
        # logging_info(f'Dataset: {self.dataset_name} - Model: {self.model_name}')
        # logging_info(f'Model confusion matrix: {self.confusion_matrix}')

        # normalizing confusion matrix 
        self.normalize_confusion_matrix()

        # computing performance metrics from confusion matrix 
        self.compute_performance_metrics_from_confusion_matrix()


    def normalize_confusion_matrix(self):
        
        # normalizing values summarizing by rows
        self.confusion_matrix_normalized = np.copy(self.confusion_matrix)
        sum_columns_aux_1 = np.sum(self.confusion_matrix_normalized,axis=0)
        row, col = self.confusion_matrix_normalized.shape
        for i in range(col):
            if sum_columns_aux_1[i] > 0:
                self.confusion_matrix_normalized[:,i] = self.confusion_matrix_normalized[:,i] / sum_columns_aux_1[i]

        # logging_info(f'Model confusion matrix normalized: {LINE_FEED} {self.confusion_matrix_normalized}')
        # logging_info(f'Model confusion matrix normalized: {LINE_FEED} {self.confusion_matrix_normalized}')

    def compute_performance_metrics_from_confusion_matrix(self):
        # Obtain TP, FN FP, and TN for each class in the confusion matrix
        # logging_info(f'Confusion matrix: {LINE_FEED}{self.confusion_matrix}')

        # initializing variable 
        self.tp_per_class = [0 for i in range(self.number_of_classes + 1)]
        self.fp_per_class = [0 for i in range(self.number_of_classes + 1)]
        self.fn_per_class = [0 for i in range(self.number_of_classes + 1)]
        self.tn_per_class = [0 for i in range(self.number_of_classes + 1)]
        self.tp_model = 0
        self.fp_model = 0
        self.fn_model = 0
        self.tn_model = 0
        
        cm_fp = self.confusion_matrix[1:-1, 1:]
        self.tp_per_class = cm_fp.diagonal()
        self.fp_per_class = cm_fp.sum(1) - self.tp_per_class
        
        cm_fn = self.confusion_matrix[1:, 1:-1]
        # DO NOT USE below because it consider all values of matrix, but it must
        # use just the last row to calculate the false negatives
        # self.fn_per_class = cm_fn.sum(0) - self.tp_per_class
        self.fn_per_class = cm_fn[-1:,].squeeze()

        # TN will be calculate just per class as every bounding boxes predicted of other classes,
        # all predicted bbox minus the bounding boxes predicted considering TP and FP of that class.
        # logging_info(f'self.confusion_matrix: {LINE_FEED} {self.confusion_matrix}')
        cm_tn = self.confusion_matrix[1:-1, 1:]

        # computing all values of confusion matrix for true negative 
        cm_tn_all = cm_tn.sum()

        # logging_info(f'cm_tn: {LINE_FEED} {cm_tn}')
        # logging_info(f'number_of_classes: {self.number_of_classes}')
        for i in range(self.number_of_classes):
            # sum all elements of the row "i"
            cm_tn_row = cm_tn[i,].sum()
            # sum all elements of the column "i"
            cm_tn_col = cm_tn[:,i].sum()
            # getting the true positive of the element "i"
            element_tp = cm_tn[i,i]
            # compute the true negative value 
            self.tn_per_class[i] = cm_tn_all - cm_tn_row - cm_tn_col + element_tp

        # summarizing TP, FP, FN       
        self.tp_model = self.tp_per_class.sum()
        self.fp_model = self.fp_per_class.sum()
        self.fn_model = self.fn_per_class.sum()
        self.tn_model = 0 

        # logging_info(f'TP per class: {self.tp_per_class}')
        # logging_info(f'FP per class: {self.fp_per_class}')
        # logging_info(f'FN per class: {self.fn_per_class}')
        # logging_info(f'TN per class: {self.tn_per_class}')
        # logging_info(f'')
        # logging_info(f'TP: {self.tp_model}')
        # logging_info(f'FP: {self.fp_model}')
        # logging_info(f'FN: {self.fn_model}')
        # logging_info(f'TN: {self.tn_model}')

        # computing metrics accuracy, precision, recall, f1-score and dice per classes
        self.compute_accuracy_per_class()
        self.compute_precision_per_class()
        self.compute_recall_per_class()
        self.compute_f1_score_per_class()
        self.compute_dice_per_class()

        # logging_info(f'accuracy_per_class : {self.accuracy_per_class}')
        # logging_info(f'precision_per_class: {self.precision_per_class}')
        # logging_info(f'recall_per_class   : {self.recall_per_class}')
        # logging_info(f'f1_score_per_class : {self.f1_score_per_class}')
        # logging_info(f'dice_per_class     : {self.dice_per_class}')

        # logging_info(f'') 
        # logging_info(f'self.dataset_name: {self.dataset_name}')
        # logging_info(f'self.model_name  : {self.model_name}')
        # logging_info(f'') 
        # logging_info(f'-----------------------------------------------') 


    # compute accuracy per classes and for the model
    def compute_accuracy_per_class(self):
        # initializing variable 
        self.accuracy_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute precision for each class
        for i in range(len(self.tp_per_class)):
            denominator = self.tp_per_class[i] + self.tn_per_class[i] + \
                          self.fp_per_class[i] + self.fn_per_class[i]
            if denominator > 0:
                self.accuracy_per_class[i] = (self.tp_per_class[i] + self.tn_per_class[i]) / denominator

    # compute precision per classes and for the model
    def compute_precision_per_class(self):
        # initializing variable 
        self.precision_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute precision for each class
        for i in range(len(self.tp_per_class)):
            denominator = self.tp_per_class[i] + self.fp_per_class[i]
            if denominator > 0:
                self.precision_per_class[i] = (self.tp_per_class[i]) / denominator

        # compute precision of the model
        # self.precision_per_class[i+1] = np.sum(self.precision_per_class) / self.number_of_classes
        self.precision_per_class[i+1] = self.get_model_precision()

    # compute recall per classes and for the model
    def compute_recall_per_class(self):
        # initializing variable 
        self.recall_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute recall for each class
        for i in range(len(self.tp_per_class)):
            denominator = self.tp_per_class[i] + self.fn_per_class[i]
            if denominator > 0:
                self.recall_per_class[i] = (self.tp_per_class[i]) / denominator

        # compute recall of the model
        # self.recall_per_class[i+1] = np.sum(self.recall_per_class) / self.number_of_classes
        self.recall_per_class[i+1] = self.get_model_recall()

    # compute f1-score per classes and for the model
    def compute_f1_score_per_class(self):
        # initializing variable 
        self.f1_score_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute f1-score for each class
        for i in range(len(self.tp_per_class)):
            denominator = self.precision_per_class[i] + self.recall_per_class[i]
            if denominator > 0:
                self.f1_score_per_class[i] = \
                (2 * self.precision_per_class[i] * self.recall_per_class[i]) / denominator

        # compute f1-score of the model
        # self.f1_score_per_class[i+1] = np.sum(self.f1_score_per_class) / self.number_of_classes
        self.f1_score_per_class[i+1] = self.get_model_f1_score()

    # compute dice per classes and for the model
    def compute_dice_per_class(self):
        # initializing variable 
        self.dice_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute dice for each class
        for i in range(len(self.tp_per_class)):
            denominator = (2 * self.tp_per_class[i]) + self.fp_per_class[i] + self.fn_per_class[i]
            if denominator > 0:
                self.dice_per_class[i] = (2 * self.tp_per_class[i]) / denominator

        # compute dice of the model
        # self.dice_per_class[i+1] = np.sum(self.dice_per_class) / self.number_of_classes
        self.dice_per_class[i+1] = self.get_model_dice()


    # https://docs.kolena.io/metrics/accuracy/
    def get_model_accuracy(self):
        # accuracy = (self.tp_model + self.tn_model) /  \
        #            (self.tp_model + self.tn_model + self.fp_model + self.fn_model)
        accuracy = 0                   
        return accuracy

    # https://docs.kolena.io/metrics/precision/
    def get_model_precision(self):
        precision = (self.tp_model) /  \
                    (self.tp_model + self.fp_model)
        return precision

    # https://docs.kolena.io/metrics/recall/
    def get_model_recall(self):
        recall = (self.tp_model) /  \
                 (self.tp_model + self.fn_model)
        return recall

    # https://docs.kolena.io/metrics/f1-score/
    def get_model_f1_score(self):
        f1_score = (2.0 * self.get_model_precision() * self.get_model_recall()) /  \
                   (self.get_model_precision() + self.get_model_recall())
        return f1_score

    # https://docs.kolena.io/metrics/specificity/
    def get_model_specificity(self):
        specificity = (self.tn_model) /  \
                      (self.tn_model + self.fp_model)
        return specificity

    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    def get_model_dice(self):
        dice = (2 * self.tp_model) /  \
               ((2 * self.tp_model) + self.fp_model + self.fn_model)
        return dice
                  
    # def save_json(self, matrices_folder):

    #     logging_info(f'matrices_folder: {matrices_folder}')

    #     models_diversity_measures_dic = {}

    #     # initializing dictionary and variables 
    #     for dataset_name_key, models_pair_relationship_matrix in self.dataset_relationship_matrix_list.items():
    #         logging_info(f'dataset_name_key: {dataset_name_key}')
    #         logging_info(f'models_pair_relationship_matrix: {models_pair_relationship_matrix}')

    #         # check and create new folder for image relationship matrix 
    #         dataset_name_folder = os.path.join(matrices_folder, dataset_name_key)
    #         if not os.path.exists(dataset_name_folder):
    #             Utils.create_directory(dataset_name_folder)

    #         # creating new dicionary for dataset_name_key
    #         models_diversity_measures_dic[dataset_name_key] = {}

    #         # looping models pair relationship matrix 
    #         for models_pair_key, models_pair_relationship_matrix_item in models_pair_relationship_matrix.items():
    #             logging_info(f'')
    #             logging_info(f'models_pair_key: {models_pair_key}')
    #             logging_info(f'models_pair_relationship_matrix_item.cor: {models_pair_relationship_matrix_item.cor}')
    #             logging_info(f'models_pair_relationship_matrix_item.dfm: {models_pair_relationship_matrix_item.dfm}')
    #             logging_info(f'models_pair_relationship_matrix_item.dm: {models_pair_relationship_matrix_item.dm}')
    #             logging_info(f'models_pair_relationship_matrix_item.ia: {models_pair_relationship_matrix_item.ia}')
    #             logging_info(f'models_pair_relationship_matrix_item.qsta: {models_pair_relationship_matrix_item.qstat}')

    #             # creating new dicionary for dataset_name_key
    #             models_diversity_measures_dic[dataset_name_key][models_pair_key] = {}
    #             models_diversity_measures_dic[dataset_name_key][models_pair_key] = {
    #                 "correlation_coefficient" : models_pair_relationship_matrix_item.cor,
    #                 "double_fault_measure" : models_pair_relationship_matrix_item.dfm,
    #                 "disagreement_measure" : models_pair_relationship_matrix_item.dm,
    #                 "interrater_agreement" : models_pair_relationship_matrix_item.ia,
    #                 "q_statistic" : models_pair_relationship_matrix_item.qstat,
    #             }               
              
    #         # setting filename 
    #         json_filename = os.path.join(
    #             dataset_name_folder,
    #             ("models-diversity-" + dataset_name_key + ".json")
    #         )

    #         # Save as JSON
    #         with open(json_filename, "w") as out_f:
    #                 json.dump(models_diversity_measures_dic, out_f, indent=2)


    def save_confusion_matrix(self, result_folder, filename, title, cm_classes):
        # setting to do not show graph in screen
        show_graph = False
              
        x_labels_names = cm_classes.copy()
        y_labels_names = cm_classes.copy()
        x_labels_names.append('Background False Positives')    
        y_labels_names.append('Undetected objects')
        format='.0f'
        path_and_filename = os.path.join(result_folder, filename + '_confusion_matrix.png')
        Utils.save_plot_confusion_matrix(self.confusion_matrix, 
                                         path_and_filename, title, format,
                                         x_labels_names, y_labels_names, show_graph=show_graph)

        # path_and_filename = os.path.join(result_folder, filename + '_confusion_matrix_normalized.png')
        # Utils.save_plot_confusion_matrix(model_performance_metrics.normalized_confusion_matrix,
        #                                  path_and_filename, title, format,
        #                                  x_labels_names, y_labels_names, show_graph=show_graph)

        path_and_filename = os.path.join(result_folder, filename + '_confusion_matrix.xlsx')
        Utils.save_confusion_matrix_excel(self.confusion_matrix, path_and_filename, x_labels_names, y_labels_names, 
                                          self.tp_per_class, self.fp_per_class, self.fn_per_class, self.tn_per_class)
        

    # saving performance metrics (Excel sheet format) from confusion matrix 
    def save_performance_metrics(self, result_folder, filename, title, cm_classes, 
                                 iou_threshold_for_grouping, iou_threshold_for_inference, non_maximum_suppression, 
                                 selected_detectors):

        # setting path and filename 
        path_and_filename = os.path.join(result_folder, filename + '_confusion_matrix_metrics.xlsx')

        sheet_name='metrics_summary'
        sheet_list = []
        sheet_list.append(['Metrics Results calculated by application', ''])
        sheet_list.append(['', ''])
        sheet_list.append(['Model', f'{title}'])
        sheet_list.append(['Selected Detector', f'{selected_detectors}'])
        sheet_list.append(['', ''])
        sheet_list.append(['Threshold',  f"{0:.2f}"])
        sheet_list.append(['IoU Threshold Grouping',  f"{iou_threshold_for_grouping:.2f}"])
        sheet_list.append(['IoU Threshold Inference',  f"{iou_threshold_for_inference:.2f}"])
        sheet_list.append(['Non-Maximum Supression',  f"{non_maximum_suppression:.2f}"])
        sheet_list.append(['', ''])

        sheet_list.append(['TP / FP / FN / TN per Class', ''])
        # cm_classes = classes[1:(number_of_classes+1)]

        # setting values of TP, FP, and FN per class
        sheet_list.append(['Class', 'TP', 'FP', 'FN', 'TN'])
        # for i, class_name in enumerate(classes[1:(number_of_classes+1)]):
        for i, class_name in enumerate(cm_classes):
            row = [class_name, 
                f'{self.tp_per_class[i]:.0f}',
                f'{self.fp_per_class[i]:.0f}',
                f'{self.fn_per_class[i]:.0f}',
                f'{self.tn_per_class[i]:.0f}',
                ]
            sheet_list.append(row)

        i += 1
        row = ['Total',
            f'{self.tp_model:.0f}',
            f'{self.fp_model:.0f}',
            f'{self.fn_model:.0f}',
            f'{self.tn_model:.0f}',
            ]
        sheet_list.append(row)    
        sheet_list.append(['', ''])

        # setting values of metrics precision, recall, f1-score and dice per class
        sheet_list.append(['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Dice'])
        for i, class_name in enumerate(cm_classes):
            row = [class_name, 
                f'{self.accuracy_per_class[i]:.8f}',
                f'{self.precision_per_class[i]:.8f}',
                f'{self.recall_per_class[i]:.8f}',
                f'{self.f1_score_per_class[i]:.8f}',
                f'{self.dice_per_class[i]:.8f}',
                ]
            sheet_list.append(row)

        i += 1
        row = ['Model Metrics',
                f'{self.get_model_accuracy():.8f}',
                f'{self.get_model_precision():.8f}',
                f'{self.get_model_recall():.8f}',
                f'{self.get_model_f1_score():.8f}',
                f'{self.get_model_dice():.8f}',
            ]
        sheet_list.append(row)
        sheet_list.append(['', ''])

        # metric measures 
        sheet_list.append(['Metric measures', ''])
        sheet_list.append(['number_of_images', f'{self.confusion_matrix_summary["number_of_images"]:.0f}'])
        sheet_list.append(['number_of_bounding_boxes_target', f'{self.confusion_matrix_summary["number_of_bounding_boxes_target"]:.0f}'])
        sheet_list.append(['number_of_bounding_boxes_predicted', f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted"]:.0f}'])
        sheet_list.append(['number_of_bounding_boxes_predicted_with_target', f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"]:.0f}'])
        sheet_list.append(['number_of_incorrect_predictions', f'{self.confusion_matrix_summary["number_of_ghost_predictions"]:.0f}'])
        sheet_list.append(['number_of_undetected_objects', f'{self.confusion_matrix_summary["number_of_undetected_objects"]:.0f}'])

        # saving metrics sheet
        Utils.save_metrics_excel(path_and_filename, sheet_name, sheet_list)
        logging_sheet(sheet_list)


    def save_inferenced_images(self, result_folder, filename):

        # print(f'save_inferenced_images')
        # print(f'result_folder: {result_folder}')

        # preparing bounding boxes to save 
        all_bounding_boxes = []
        for image_performance_metric in self.image_performance_metrics_list:
            for bbox in image_performance_metric.image_bounding_boxes:
                # item = []
                # for bbox_item in range(len(bbox)):
                #     item.append(bbox[bbox_item])
                item = [bbox[bbox_item] for bbox_item in range(len(bbox))]
                all_bounding_boxes.append(item)

        # setting path and filename 
        path_and_filename = os.path.join(result_folder, filename + '_image_bounding_boxes.xlsx')

        # preparing columns name to list        
        column_names = [
            "image_name",
            "ground_truth_bbox",
            "ground_truth_label",
            "predicted_bbox",
            "predicted_label",
            "predicted_score",
            "score_threshold",
            "iou",
            "iou_threshold",
            "status",
            "model_name"
        ]

        # creating dataframe from list 
        df = pd.DataFrame(all_bounding_boxes, columns=column_names)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name='bounding_boxes', index=False)


    def save_inferenced_images_old_(self, result_folder, filename):

        # print(f'save_inferenced_images')
        # print(f'result_folder: {result_folder}')

        # preparing list of inferenced bounding boxes to save 
        i = 0
        images_ground_truths_and_predictions = []
        for image_filename, image_prediction in self.predictions["images"].items():
            # print(f'')
            # print(f'summarize predictions {i} - image_filename: {image_filename}')
            # print(f'{self.predictions["images"][image_filename]["ground_truths"]}')
            # print(f'{self.predictions["images"][image_filename]["predictions"]}')
            item = []
            item.append(image_filename)
            item.append(str(self.predictions["images"][image_filename]["ground_truths"]["boxes"]))
            item.append(str(self.predictions["images"][image_filename]["ground_truths"]["labels"]))
            if "boxes" in self.predictions["images"][image_filename]["predictions"]:
                item.append(str(self.predictions["images"][image_filename]["predictions"]["boxes"]))
                item.append(str(self.predictions["images"][image_filename]["predictions"]["labels"]))
                item.append(str(self.predictions["images"][image_filename]["predictions"]["scores"]))
                item.append("threshold")
                # item.append(str(self.predictions["images"][image_filename]["predictions"]["iou"]))
                item.append("")
                item.append("iou threshold")
            else:
                item.append("")
                item.append("")
                item.append("")
                item.append("")
                item.append("")
                item.append("")
            item.append("status")
            if "model" in self.predictions["images"][image_filename]["predictions"]:
                item.append(str(self.predictions["images"][image_filename]["predictions"]["model"]))
            else:
                item.append("")
            images_ground_truths_and_predictions.append(item)
            # print(f'{i} {images_ground_truths_and_predictions[i-1]}')
            i += 1

        # setting path and filename 
        path_and_filename = os.path.join(result_folder, filename + '_images_bounding_boxes.xlsx')

        # preparing columns name to list
        column_names = [
            'image name',
            'target bbox',
            'target label',
            'predict bbox',
            'predict label',
            'predict score',
            'threshold',
            'iou',
            'iou threshold',
            'status',
            'model'
        ]
        # if len(self.images_bounding_boxes[0]) == 11:
        #     column_names.append('model name')

        # creating dataframe from list 
        df = pd.DataFrame(images_ground_truths_and_predictions, columns=column_names)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name='bounding_boxes', index=False)





    # # summarize predicted detections per model name  
    # def summarize_inferenced_images(self, result_folder, filename, selected_detectors):

    #     i = 1
    #     for image_filename, image_prediction in self.predictions["images"].items():
    #         print(f'summarize predictions {i} - image_filename: {image_filename}')
    #         print(f'{self.predictions["images"][image_filename]["ground_truths"]}')
    #         print(f'{self.predictions["images"][image_filename]["predictions"]}')
    #         i += 1

        
    #     # initializing list 
    #     summarized_inferenced_images_by_model = {}

    #     # processing inferenced predictions
    #     for image_bounding_box in self.images_bounding_boxes:
    #         # get model name
    #         model_name = image_bounding_box[10]

    #         # summarize number of predicted detections by model name 
    #         if (model_name != ''):

    #             if model_name in summarized_inferenced_images_by_model:
    #                 summarized_inferenced_images_by_model[model_name] += 1
    #             else:
    #                 summarized_inferenced_images_by_model[model_name] = 1

    #     print(f'summarized_inferenced_images_by_model: {summarized_inferenced_images_by_model}')

    #     model_names = []
    #     for key, value in summarized_inferenced_images_by_model.items():
    #         model_names.append([key, value])

    #     print(f'model_names: {model_names}')

    #     # just return if there is nothing to record
    #     if len(model_names) == 0:
    #         return

    #        # preparing columns name to list
    #     column_names = [
    #         'Model Name',
    #         'Predicted Detections',
    #     ]

    #     # creating dataframe from list 
    #     df = pd.DataFrame(model_names, columns=column_names)

    #     # writing excel file from dataframe
    #     df.to_excel(path_and_filename, sheet_name='summarize_bounding_boxes', index=False)


    # Converting the predictions
    def convert_predictions_format(self, predictions):

        # initializing objects 
        boxes = []
        scores = []
        labels = []
        label_names = []
        model = []
        number_of_fused_models = []

        # converting list of predictions into a detailed list 
        for prediction in predictions:
            boxes.append(prediction["box"])
            scores.append(prediction["score"])
            labels.append(prediction["label"])
            model.append(prediction["model"])
            number_of_fused_models.append(prediction["number_of_fused_models"])

        converted_predictions = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "label_names": label_names,
            "model": model,
            "number_of_fused_models": number_of_fused_models
        }

        # returning the converted predictions
        return converted_predictions