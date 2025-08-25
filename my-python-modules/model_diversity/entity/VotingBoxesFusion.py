# VotingBoxesFusion Class
# This class provides services for fusion of multiple detectors bounding boxes based on voting schemes.


# Importing python libraries 
import math
# import pandas as pd
# import matplotlib.pyplot as plt
# from argon2 import Parameters

import numpy as np
import torchvision
from torchvision.ops import * 

# Importing python modules
from common.manage_log import *
from common.metrics import *
from model_diversity.entity.ModelPerformanceMetrics import * 

# from common.utils  import *
# from common.entity.ImageAnnotation import ImageAnnotation
# from model_ensemble.image_utils import ImageUtils

# from model_diversity.entity.ModelsPairRelationshipMatrix import * 

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class VotingBoxesFusion:
    def __init__(self, models_selection_method_name=None, dataset_name=None, selected_models=None,  all_predictions=None, 
                 iou_threshold_for_grouping=None, iou_threshold_for_inference=None, non_maximum_suppression=None, 
                 number_of_classes=None, classes=None, input_dataset_type=None
                ):
                #  method_short_name=None, method_description=None, 
                #  dataset_name=None, models_diversity_measures=None,
                #  method_results_folder=None, models_dic=None, top_t=None):
        
        # input attributes 
        self.models_selection_method_name = models_selection_method_name
        self.dataset_name = dataset_name
        self.selected_models = selected_models
        self.all_predictions_list = all_predictions
        self.iou_threshold_for_grouping = iou_threshold_for_grouping
        self.iou_threshold_for_inference = iou_threshold_for_inference
        self.non_maximum_suppression = non_maximum_suppression
        self.number_of_classes = number_of_classes
        self.classes = classes
        self.input_dataset_type = input_dataset_type

        # output attributes 
        self.fusioned_bboxes_per_image = {}
        self.affirmative_performance_metrics = None
        self.consensus_performance_metrics = None
        self.unanimous_performance_metrics = None

        # self.method_complement_text = ''
        self.top_t = ''
        self.f1_score_threshold = ''
        self.minimum_number_of_models = ''
        
    def execute(self):

        # logging_info(f'Performing Bounding Boxes Fusion based on Voting Schemes')
        # logging_info(f'')

        # logging_info(f'self.models_selection_method: {self.models_selection_method_name}')
        # logging_info(f'self.selected_models: {self.selected_models}')
        # logging_info(f'self.all_preditcions_list: {self.all_predictions_list}')
        # logging_info(f'len(self.all_predictions_list): {len(self.all_predictions_list)}')
        # logging_info(f'')
        # for i in range(len(self.all_predictions_list)):
            # logging_info(f'========================================')
            # logging_info(f'{i} {self.all_predictions_list[i]}')
            # logging_info(f'')
       
        # initializing variables and constants
        # service_name = "voting-boxes-fusion"
        affirmative_prediction_type = "affirmative"    
        consensus_prediction_type = "consensus"    
        unanimous_prediction_type = "unanimous"

        # creating objects for model performance metrics 
        self.affirmative_performance_metrics = ModelPerformanceMetrics()
        self.affirmative_performance_metrics.dataset_name = self.dataset_name
        self.affirmative_performance_metrics.model_name = affirmative_prediction_type
        self.affirmative_performance_metrics.classes = self.classes
        self.affirmative_performance_metrics.number_of_classes = self.number_of_classes
        self.affirmative_performance_metrics.iou_threshold = self.iou_threshold_for_inference
        self.affirmative_performance_metrics.prepare_performance_metric(self.input_dataset_type)

        self.consensus_performance_metrics = ModelPerformanceMetrics()
        self.consensus_performance_metrics.dataset_name = self.dataset_name
        self.consensus_performance_metrics.model_name = consensus_prediction_type 
        self.consensus_performance_metrics.classes = self.classes
        self.consensus_performance_metrics.number_of_classes = self.number_of_classes
        self.consensus_performance_metrics.iou_threshold = self.iou_threshold_for_inference
        self.consensus_performance_metrics.prepare_performance_metric(self.input_dataset_type)

        self.unanimous_performance_metrics = ModelPerformanceMetrics()
        self.unanimous_performance_metrics.dataset_name = self.dataset_name
        self.unanimous_performance_metrics.model_name = unanimous_prediction_type
        self.unanimous_performance_metrics.classes = self.classes
        self.unanimous_performance_metrics.number_of_classes = self.number_of_classes
        self.unanimous_performance_metrics.iou_threshold = self.iou_threshold_for_inference
        self.unanimous_performance_metrics.prepare_performance_metric(self.input_dataset_type)

        # creating inference metrics for ensemble strategies
        # affirmative_prediction_type = "Affirmative"    
        # self.affirmative_inference_metric = Metrics(
        #     model=service_name + ' - ' + affirmative_prediction_type,
        #     number_of_classes=self.number_of_classes,
        # )
        # consensus_prediction_type = "Consensus"    
        # self.consensus_inference_metric = Metrics(
        #     model=service_name + ' - ' + consensus_prediction_type,
        #     number_of_classes=self.number_of_classes,
        # )
        # unanimous_prediction_type = "Unanimous"    
        # self.unanimous_inference_metric = Metrics(
        #     model=service_name + ' - ' + unanimous_prediction_type,
        #     number_of_classes=self.number_of_classes,
        # )

        # getting all predictions keys as image filename for all inferenced models
        test_image_filenames = self.get_all_test_image_filenames(self.all_predictions_list)            
        # logging_info(f'len(test_image_filenames): {len(test_image_filenames)}')
        
        # ensembling the inference results for each test image 
        image_number = 0
        for test_image_filename in test_image_filenames:

            # show_logging_info = False
            # if test_image_filename == "ds-2023-09-07-santa-helena-de-goias-go-fazenda-sete-ilhas-pivo-01-IMG_2530-bbox-1505679995.jpg":
            #     show_logging_info = True

            show = False
            # if test_image_filename == "8125.jpg":
            #     show = True    

            image_number += 1
            # logging_info(f'')
            # logging_info(f'----------------------------------------------------')
            # logging_info(f'test_image_filename #{image_number}: {test_image_filename}')

            # flattening list of predictions
            # logging_info(f'')
            # logging_info(f'get_flattened_predictons')
            flattened_predictions = self.get_flattened_predictons(self.all_predictions_list, test_image_filename)
            all_predictions_size = len(flattened_predictions)

            # if show_logging_info:
            #     logging_info(f'len(flattened_predictions): {len(flattened_predictions)}')
            #     logging_info(f'flattened_predictions: {flattened_predictions}')

            # grouping detections based on overlapping of their bounding boxes and classes
            # by using the IoU metric
            # logging_info(f'')
            # logging_info(f'get_grouped_predictions')
            grouped_predictions = self.get_grouped_predictions(flattened_predictions, self.iou_threshold_for_grouping)
            
            # if show_logging_info:
            #     logging_info(f'len(grouped_predictions): {len(grouped_predictions)}')

            # i = 1
            # for grouped_predicton in grouped_predictions:
            #     logging_info(f'grouped_predicton: {i}) {len(grouped_predicton)} - {grouped_predicton}')
            #     i += 1

            # applying voting strategies in the grouped predictions: affirmative, consensus, and unanimous
            affirmative_predictions, consensus_predictions, unanimous_predictions = \
                self.apply_voting_strategies(grouped_predictions, all_predictions_size, self.selected_models, 
                                             self.non_maximum_suppression, show)
            
            # logging_info(f'')
            # logging_info(f'Voting Strategies Results:')
            # logging_info(f'')
            # logging_info(f'affirmative_predictions: {len(affirmative_predictions)} - {affirmative_predictions}')
            # logging_info(f'consensus_predictions: {len(consensus_predictions)} - {consensus_predictions}')
            # logging_info(f'unanimous_predictions: {len(unanimous_predictions)} - {unanimous_predictions}')
            # logging_info(f'')

            # applying non-maximum supression (nms) on  the predictions 

            if show:
                print(f'')
                print(f'')
                print(f'Before NMS on {test_image_filename} - affirmative predictions')
                print(f'self.non_maximum_suppression: {self.non_maximum_suppression}')
                print(f'len(affirmative_predictions): {len(affirmative_predictions)}')
                print(f'affirmative_predictions: {affirmative_predictions}')
            affirmative_nms_predictions = self.apply_nms_into_predictions(affirmative_predictions, self.non_maximum_suppression, show)
            if show:
                print(f'After NMS on {test_image_filename} - affirmative predictions')
                # print(f'len(affirmative_nms_predictions): {len(affirmative_nms_predictions["boxes"])}')
                print(f'affirmative_nms_predictions: {affirmative_nms_predictions}')

            if show:
                print(f'Before NMS on {test_image_filename} - consensus predictions')
                print(f'self.non_maximum_suppression: {self.non_maximum_suppression}')
                print(f'len(consensus_predictions): {len(consensus_predictions)}')
                print(f'consensus_predictions: {consensus_predictions}')
            consensus_nms_predictions = self.apply_nms_into_predictions(consensus_predictions, self.non_maximum_suppression, show)
            if show:
                print(f'After NMS on {test_image_filename} - consensus predictions')
                # print(f'len(consensus_nms_predictions): {len(consensus_nms_predictions["boxes"])}')
                print(f'consensus_nms_predictions: {consensus_nms_predictions}')

            if show:
                print(f'Before NMS on {test_image_filename} - unanimous predictions')
                print(f'self.non_maximum_suppression: {self.non_maximum_suppression}')
                print(f'len(unanimous_predictions): {len(unanimous_predictions)}')
                print(f'unanimous_predictions: {unanimous_predictions}')
            unanimous_nms_predictions = self.apply_nms_into_predictions(unanimous_predictions, self.non_maximum_suppression, show)
            if show:
                print(f'After NMS on {test_image_filename} - unanimous predictions')
                # print(f'len(unanimous_nms_predictions): {len(unanimous_nms_predictions["boxes"])}')
                print(f'unanimous_nms_predictions: {unanimous_nms_predictions}')

            # getting target annotations of the test image 
            ground_truths = self.get_ground_truth_of_test_image(self.all_predictions_list, test_image_filename)                  
            # logging_info(f'image_filename - ground_truths : {ground_truths}')
            
            # adding ensembled predictions to the performance metrics
            self.affirmative_performance_metrics.add_image(test_image_filename, ground_truths, affirmative_nms_predictions)
            self.consensus_performance_metrics.add_image(test_image_filename, ground_truths, consensus_nms_predictions)
            self.unanimous_performance_metrics.add_image(test_image_filename, ground_truths, unanimous_nms_predictions)

            # # adding ensembled predictions to the performance metrics
            # self.affirmative_inference_metric.set_details_of_inferenced_image(test_image_filename, ground_truths, affirmative_nms_predictions)
            # self.consensus_inference_metric.set_details_of_inferenced_image(test_image_filename, ground_truths, consensus_nms_predictions)
            # self.unanimous_inference_metric.set_details_of_inferenced_image(test_image_filename, ground_truths, unanimous_nms_predictions)

            # logging_info(f'add_ensembled_predictions_to_performance_metrics - affirmative_inference_metric')
            # self.add_ensembled_predictions_to_performance_metrics(
            #     affirmative_inference_metric, test_image_filename, ground_truths, affirmative_nms_predictions)
            # logging_info(f'add_ensembled_predictions_to_performance_metrics - consensus_inference_metric')
            # self.add_ensembled_predictions_to_performance_metrics(
            #     consensus_inference_metric, test_image_filename, ground_truths, consensus_nms_predictions)
            # logging_info(f'add_ensembled_predictions_to_performance_metrics - unanimous_inference_metric')
            # self.add_ensembled_predictions_to_performance_metrics(
            #     unanimous_inference_metric, test_image_filename, ground_truths, unanimous_nms_predictions)
            # logging_info(f'')

            # saving predicted image with bounding boxes
            # save_predicted_image(parameters, test_image_filename, affirmative_nms_predictions, affirmative_prediction_type)
            # save_predicted_image(parameters, test_image_filename, consensus_nms_predictions, consensus_prediction_type)
            # save_predicted_image(parameters, test_image_filename, unanimous_nms_predictions, unanimous_prediction_type)

        # computing model performance metric object 
        # logging_info(f'')
        # logging_info(f'Computing performance metrics for voting schemes')
        # logging_info(f'self.affirmative_performance_metrics.predictions')
        # logging_info(f'{self.affirmative_performance_metrics.predictions}')
        self.affirmative_performance_metrics.compute_metrics()
        self.consensus_performance_metrics.compute_metrics()
        self.unanimous_performance_metrics.compute_metrics()

        # computing the performance metrics 
        # logging_info(f'')
        # logging_info(f'affirmative_inference_metric: {len(self.affirmative_inference_metric.inferenced_images)}')
        # logging_info(f'affirmative_inference_metric: {self.affirmative_inference_metric.inferenced_images}')
        # # computing_performance_metrics(parameters, affirmative_inference_metric, affirmative_prediction_type)

        # logging_info(f'')
        # logging_info(f'consensus_inference_metric: {len(self.consensus_inference_metric.inferenced_images)}')
        # logging_info(f'consensus_inference_metric: {self.consensus_inference_metric.inferenced_images}')
        # # computing_performance_metrics(parameters, consensus_inference_metric, consensus_prediction_type)

        # logging_info(f'')   
        # logging_info(f'unanimous_inference_metric: {len(self.unanimous_inference_metric.inferenced_images)}')
        # logging_info(f'unanimous_inference_metric: {self.unanimous_inference_metric.inferenced_images}')   
        # # computing_performance_metrics(parameters, unanimous_inference_metric, unanimous_prediction_type)
        # logging_info(f'')


    def get_all_test_image_filenames(self,all_predictions):

        # logging_info(f'get_all_test_image_filenames')
        # logging_info(f'len(all_predictions): {len(all_predictions)}')
        
        # getting all test image filenames with no duplicates
        test_image_filenames = {}
        for model in all_predictions:
            # for image_filename in list(all_predictions[model].keys()):
            for i in range(len(all_predictions)):
                # logging_info(f'all_predictions: {i} {all_predictions[i]}')
                
                for image_filename_key, image_value in all_predictions[i]["images"].items():
                    # logging_info(f'image_filename_key: {image_filename_key}')
                    test_image_filenames[image_filename_key] = image_filename_key

        # returning test image filenames 
        return test_image_filenames

    def get_flattened_predictons(self, predictions_list, test_image_filename):

        # creating flattened predictions of one image from all models 
        flattened_predictions = []

        for model_ind in range(len(predictions_list)):
            # logging_info(f'model_ind: {model_ind}')
            # logging_info(f'{predictions_list[model_ind]["model_name"]}')
            model_name = predictions_list[model_ind]["model_name"]

            # Change date: 08-07-2025 - Handle case when model_name does not exist in the image list 
            if test_image_filename in predictions_list[model_ind]["images"]:
                if len(predictions_list[model_ind]["images"][test_image_filename]['predictions']) > 0:
                    test_image_predictions = predictions_list[model_ind]["images"][test_image_filename]['predictions']
                    # logging_info(f'test_image_predictions: {test_image_predictions}')

                    # get all valid predictions of the image 
                    if len(test_image_predictions['boxes']) == 0: 
                        continue 
                    
                    # flattening image predictions 
                    for box, score, label in zip(test_image_predictions['boxes'], 
                                                    test_image_predictions['scores'], 
                                                    test_image_predictions['labels']):
                        # setting data of the one detection 
                        detection = {}
                        detection['box'] = box 
                        detection['score'] = score
                        detection['label'] = label
                        detection['model'] = model_name
                        
                        # adding detection to flattened predictions 
                        flattened_predictions.append(detection)

        # returning flattened predictions
        return flattened_predictions
       
    # grouping detections based on overlapping of their bounding boxes and classes
    # by using the IoU metric
    def get_grouped_predictions(self, flattened_predictions, iou_threshold_for_grouping):

        # logging_info(f'get_grouped_predictions - starting')
        # logging_info(f'iou_threshold_for_grouping: {iou_threshold_for_grouping}')

        # setting flattened predictions as NOT removed
        for prediction in flattened_predictions:
            prediction['removed'] = False        

        # creating grouped predictions of one image from flattened predictions 
        grouped_predictions = []
        one_grouped_prediction = []

        # calculating the IoU between all predictions of the same class for grouping 
        for i in range(len(flattened_predictions)):
            # getting bounding box reference to compare with the others
            bounding_box_reference = flattened_predictions[i]
            if bounding_box_reference['removed']:
                continue

            # initialize IoU of the reference bounding box
            bounding_box_reference['iou'] = 0
            bounding_box_reference['iou_threshold_for_grouping'] = iou_threshold_for_grouping

            # setting one grouped prediction with first prediction (reference)
            one_grouped_prediction.append(bounding_box_reference)

            # removing bounding box reference from the flatted predictions 
            bounding_box_reference['removed'] = True
            
            # comparing with others bounding boxes 
            for j in range(i+1, len(flattened_predictions)):
                # getting bounding box for comparision 
                bounding_box_next = flattened_predictions[j]
                if bounding_box_next['removed']:
                    continue

                # calculating IoU of the two bounding boxes: reference and next
                # Both sets of boxes are expected to be in (x1, y1, x2, y2)
                box_reference = torch.IntTensor(bounding_box_reference['box'])
                box_next = torch.IntTensor(bounding_box_next['box'])
                # logging_info(f'box_reference: {box_reference}')
                # logging_info(f'box_next: {box_next}')
                iou = box_iou(box_reference.unsqueeze(0), box_next.unsqueeze(0))
                # logging_info(f'computing iou {box_reference} and {box_next}: {iou}')

                # setting IoU value 
                bounding_box_next['iou'] = iou
                bounding_box_next['iou_threshold_for_grouping'] = iou_threshold_for_grouping

                # evaluating overlapping of bounding boxes reference and next
                if (bounding_box_reference['label'] ==  bounding_box_next['label']) and \
                   (iou >= iou_threshold_for_grouping):

                    # removing bounding box reference from the flatted predictions 
                    bounding_box_next['removed'] = True

                    # adding bounding box to group
                    one_grouped_prediction.append(bounding_box_next)

            # adding one new grouped prediction
            grouped_predictions.append(one_grouped_prediction)

            # initializing one new grouped prediction
            one_grouped_prediction = []

        # returning grouped predictions         
        return grouped_predictions

    # # applying voting strategies in the grouped predictions that can be of three strategies:
    # # 1) affirmative: all grouped predictions 
    # # 2) consensus: the group size must be greater than m/2, where m is the size of flattened predeictions
    # # 3) unanimous: the group size must be equal to m size
    # def apply_voting_strategies_original(self, grouped_predictions, all_predictions_size, show):

    #     # creating results list for each strategy
    #     affirmative_predictions = []
    #     consensus_predictions = []
    #     unanimous_predictions = []

    #     if show:
    #         print(f'Processing grouped predictions: {grouped_predictions}') 

    #     # processing all grouped predictions
    #     for grouped_prediction in grouped_predictions:

    #         # affirmative strategy
    #         for prediction in grouped_prediction:
    #             affirmative_predictions.append(prediction)

    #         # consensus strategy
    #         if len(grouped_prediction) >= math.ceil(all_predictions_size / 2.0):
    #             for prediction in grouped_prediction:
    #                 consensus_predictions.append(prediction)

    #         # unanimous strategy 
    #         if len(grouped_prediction) == all_predictions_size:
    #             for prediction in grouped_prediction:
    #                 unanimous_predictions.append(prediction)

    #     # returning results of ensembling 
    #     return affirmative_predictions, consensus_predictions, unanimous_predictions


    # applying voting strategies in the grouped predictions that can be of three strategies:
    # 1) affirmative: all grouped predictions 
    # 2) consensus: the group size must be greater than m/2, where m is the size of flattened predeictions
    # 3) unanimous: the group size must be equal to m size
    def apply_voting_strategies(self, grouped_predictions, all_predictions_size, selected_models, non_maximum_suppression, show):

        # creating results list for each strategy
        affirmative_predictions = []
        consensus_predictions = []
        unanimous_predictions = []

        if show:
            print(f'Processing grouped predictions: {grouped_predictions}')
            print(f'selected_models: {selected_models}')

        # processing all grouped predictions
        for grouped_prediction in grouped_predictions:

            # computing number of models for one prediction group
            group_models = {}
            for prediction in grouped_prediction:
                model_name = prediction['model']
                if model_name not in group_models:
                    group_models[model_name] = 1
                else:
                    group_models[model_name] += 1
            if show:
                print(f'grouped_prediction: {grouped_prediction}')
                print(f'models_from_predictions: {group_models}')
                
            # affirmative strategy
            for prediction in grouped_prediction:
                affirmative_predictions.append(prediction)

            # consensus strategy
            if len(group_models) >= math.ceil(len(selected_models) / 2.0):
                for prediction in grouped_prediction:
                    consensus_predictions.append(prediction)

            # unanimous strategy 
            if len(group_models) == len(selected_models):
                for prediction in grouped_prediction:
                    unanimous_predictions.append(prediction)

        # returning results of ensembling 
        return affirmative_predictions, consensus_predictions, unanimous_predictions

    # apply non-maximum supressoion in the predicitons list to remove overlapping bounding boxes
    def apply_nms_into_predictions(self, predictions, iou_threshold, show_print=False):
        
        # initializing kept prediction 
        kept_predictions = {}

        # evaluating predictions size
        if len(predictions) == 0:
            # logging_info(f'predictions list for nms is empty: {predictions}')
            return kept_predictions
        
        # preparing boxes and scores to apply nms
        bounding_boxes = []
        scores = []
        for prediction in predictions:
            # bounding_boxes.append(prediction['box'].numpy())
            bounding_boxes.append(prediction['box'])
            scores.append(prediction['score']) 

        bounding_boxes = torch.Tensor(bounding_boxes)        
        scores = torch.Tensor(scores)
        # logging_info(f'')
        # logging_info(f'apply nms - bounding_boxes: {bounding_boxes}')
        # logging_info(f'apply nms - scores: {scores}')
        # logging_info(f'apply nms - iou_threshold: {iou_threshold}')
        # logging_info(f'')

    
        if show_print:
            print(f'')
            print(f'apply nms - bounding_boxes: {bounding_boxes}')
            print(f'apply nms - scores: {scores}')
            print(f'apply nms - iou_threshold: {iou_threshold}')
            print(f'')

        # applying nms in the predictions
        keep_indexes = nms(bounding_boxes, scores, iou_threshold)
        if show_print:
            print(f'apply nms - keep_indexes: {keep_indexes}')

        # logging_info(f'apply nms - keep_indexes: {keep_indexes}')

        # preparing kept predictions 
        keep_predictions = []
        for keep_index in keep_indexes:
            keep_predictions.append(predictions[keep_index])

        keep_predictions = {
            "boxes": [],
            "scores": [],
            "labels": [],
            "label_names": [],
            "model": [],
            "removed": [],
            "iou": [],
            "iou_threshold_for_grouping": [],
        }
        for keep_index in keep_indexes:
            # logging_info(f'rubens predictions[index]: {predictions[keep_index]}')
            keep_predictions["boxes"].append(predictions[keep_index]['box'])
            keep_predictions["scores"].append(predictions[keep_index]['score'])
            keep_predictions["labels"].append(predictions[keep_index]['label'])
            keep_predictions["model"].append(predictions[keep_index]['model'])
            keep_predictions["removed"].append(predictions[keep_index]['removed'])
            keep_predictions["iou"].append(predictions[keep_index]['iou'])
            keep_predictions["iou_threshold_for_grouping"].append(predictions[keep_index]['iou_threshold_for_grouping'])

        # for i in range(len(predictions)):
        #     if i in keep_index:
        #         kept_predictions.append(predictions[i])

        # logging_info(f'apply nms - keep_predictions: {keep_predictions}')
        if show_print:
            print(f'apply nms - keep_predictions: {keep_predictions}')
            print(f'')

        return keep_predictions

    # getting target annotations of the test image 
    def get_ground_truth_of_test_image(self, all_predictions_list, test_image_filename):

       	# creating ground truth list of one image from all models 
        ground_truths = {     
            "boxes": [],
            "labels": [],
            "labels_names": [],
        }

        for model_ind in range(len(all_predictions_list)):
            # logging_info(f'model_ind: {model_ind}')
            # logging_info(f'{all_predictions_list[model_ind]["model_name"]}')
            model_name = all_predictions_list[model_ind]["model_name"]

            if len(all_predictions_list[model_ind]["images"][test_image_filename]['ground_truths']['boxes']) > 0:
                ground_truths = all_predictions_list[model_ind]["images"][test_image_filename]['ground_truths']
                ground_truths["model"] = model_name
                break 

        # returning ground truth list
        return ground_truths

    # # getting target annotations of the test image 
    # def get_ground_truth_of_test_image(self, all_predictions_list, test_image_filename):

    #    	# creating ground truth list of one image from all models 
    #     ground_truths = {}
    #     boxes = []
    #     labels = []
    #     labels_names = []

    #     for model_ind in range(len(all_predictions_list)):
    #         logging_info(f'model_ind: {model_ind}')
    #         logging_info(f'{all_predictions_list[model_ind]["model_name"]}')
    #         model_name = all_predictions_list[model_ind]["model_name"]

    #         found_ground_truth = False 

    #         if len(all_predictions_list[model_ind]["images"][test_image_filename]['ground_truths']) > 0:
    #             image_ground_truth = all_predictions_list[model_ind]["images"][test_image_filename]['ground_truths']
    #             logging_info(f'image_ground_truth: {image_ground_truth}')

    #             # get all valid predictions of the image 
    #             if len(image_ground_truth['boxes']) == 0: 
    #                 continue 
                
    #             # ground truth list of an image
    #             for box, label in zip(image_ground_truth['boxes'], image_ground_truth['labels']):
    #                 # setting data of the one detection 
    #                 ground_truth = {}
    #                 ground_truth['boxes'] = box 
    #                 ground_truth['labels'] = label
    #                 ground_truth['model'] = model_name
                    
    #                 # adding ground truth to list
    #                 ground_truths.append(ground_truth)

    #                 # set ground truth found 
    #                 found_ground_truth = True

    #         # end loop wheter ground truth is found 
    #         if found_ground_truth: 
    #             break 

    #     # returning ground truth list
    #     return ground_truths

    # # add ensembled predictions to the performance metrics
    # def add_ensembled_predictions_to_performance_metrics(self, inference_metric, image_name, targets, predictions):

    #     # creating adjusted predictions for inference metrics 
    #     adjusted_predictions = []
        
    #     # if len(predictions) == 0:
    #     #     logging_info(f'predictions empty')
    #     #     return 

    #     # adjusting key names of the predictions for performance metric 
    #     for prediction in predictions:
    #         item = {}
    #         item['boxes'] = torch.unsqueeze(prediction['box'], 0)
    #         item['scores'] = torch.unsqueeze(prediction['score'], 0)
    #         item['labels'] = torch.unsqueeze(prediction['label'], 0)
    #         item['model'] = prediction['model']
    #         adjusted_predictions.append(item)
        
    #     logging_info(f'add_ensembled_predictions_to_performance_metrics:')
    #     logging_info(f'targets: {len(targets)} - {targets}')
    #     logging_info(f'adjusted_predictions: {len(adjusted_predictions)} - {adjusted_predictions}')
    #     logging_info(f'')
        
    #     # adding targets and predictions
    #     inference_metric.set_details_of_inferenced_image(image_name, targets, adjusted_predictions)


    def save_results(self, result_folder, running_id_text):

        # -------------------------
        # Affirmative voting scheme
        # -------------------------
        voting_scheme = 'affirmative'
        affirmative_bboxes_fusion_folder = os.path.join(result_folder, voting_scheme)      
        Utils.create_directory(affirmative_bboxes_fusion_folder)
        title = 'Confusion Matrix' + \
                ' - Voting scheme: ' + voting_scheme + \
                ' - Models Selection Method: ' + self.models_selection_method_name + \
                ' - Dataset: ' + self.dataset_name + \
                ' - Type: ' + self.input_dataset_type
        title += LINE_FEED + \
                'Confidence threshold: ' + \
                '   IoU grouping: ' + f"{self.iou_threshold_for_grouping:.1f}" + \
                '   IoU inference: ' + f"{self.iou_threshold_for_inference:.1f}" + \
                '   Non maximum Supression: ' + f"{self.non_maximum_suppression:.1f}"
        filename = voting_scheme + '-' + self.models_selection_method_name + '-' + running_id_text
        cm_classes = self.classes[0:(self.number_of_classes+1)]
        # self.save_confusion_matrix(affirmative_bboxes_fusion_folder, filename,
        #                            title, cm_classes, self.affirmative_performance_metrics)        
        self.affirmative_performance_metrics.save_confusion_matrix(affirmative_bboxes_fusion_folder, 
                                                                   filename, title, cm_classes)
        
        # saving performance metrics 
        cm_classes_for_metrics = self.classes[1:(self.number_of_classes+1)]
        self.affirmative_performance_metrics.save_performance_metrics(
            affirmative_bboxes_fusion_folder, filename, title, cm_classes_for_metrics, 
            self.iou_threshold_for_grouping, self.iou_threshold_for_inference, self.non_maximum_suppression,
            self.selected_models)
        
        # saving bounding boxes used in the fusion 
        self.affirmative_performance_metrics.save_inferenced_images(affirmative_bboxes_fusion_folder, filename)

        # -------------------------
        # Consensus voting scheme
        # -------------------------
        voting_scheme = 'consensus'
        consensus_bboxes_fusion_folder = os.path.join(result_folder, voting_scheme)      
        Utils.create_directory(consensus_bboxes_fusion_folder)
        title = 'Confusion Matrix' + \
                ' - Voting scheme: ' + voting_scheme + \
                ' - Models Selection Method: ' + self.models_selection_method_name + \
                ' - Dataset: ' + self.dataset_name + \
                ' - Type: ' + self.input_dataset_type
        title += LINE_FEED + \
                'Confidence threshold: ' + \
                '   IoU grouping: ' + f"{self.iou_threshold_for_grouping:.1f}" + \
                '   IoU inference: ' + f"{self.iou_threshold_for_inference:.1f}" + \
                '   Non maximum Supression: ' + f"{self.non_maximum_suppression:.1f}"
        filename = voting_scheme + '-' + self.models_selection_method_name + '-' + running_id_text
        cm_classes = self.classes[0:(self.number_of_classes+1)]
        # self.save_confusion_matrix(consensus_bboxes_fusion_folder, filename,
                                #    title, cm_classes, self.consensus_performance_metrics)        
        self.consensus_performance_metrics.save_confusion_matrix(consensus_bboxes_fusion_folder,
                                                                 filename, title, cm_classes)
        
        # saving performance metrics 
        cm_classes_for_metrics = self.classes[1:(self.number_of_classes+1)]
        self.consensus_performance_metrics.save_performance_metrics(
            consensus_bboxes_fusion_folder, filename, title, cm_classes_for_metrics, 
            self.iou_threshold_for_grouping, self.iou_threshold_for_inference, self.non_maximum_suppression,
            self.selected_models)

        # saving bounding boxes used in the fusion 
        self.consensus_performance_metrics.save_inferenced_images(consensus_bboxes_fusion_folder, filename)


        # -------------------------
        # Unanimous voting scheme
        # -------------------------
        voting_scheme = 'unanimous'
        unanimous_bboxes_fusion_folder = os.path.join(result_folder, voting_scheme)      
        Utils.create_directory(unanimous_bboxes_fusion_folder)
        title = 'Confusion Matrix' + \
                ' - Voting scheme: ' + voting_scheme + \
                ' - Models Selection Method: ' + self.models_selection_method_name + \
                ' - Dataset: ' + self.dataset_name + \
                ' - Type: ' + self.input_dataset_type
        title += LINE_FEED + \
                'Confidence threshold: ' + \
                '   IoU grouping: ' + f"{self.iou_threshold_for_grouping:.1f}" + \
                '   IoU inference: ' + f"{self.iou_threshold_for_inference:.1f}" + \
                '   Non maximum Supression: ' + f"{self.non_maximum_suppression:.1f}"
        filename = voting_scheme + '-' + self.models_selection_method_name + '-' + running_id_text
        cm_classes = self.classes[0:(self.number_of_classes+1)]
        # self.save_confusion_matrix(unanimous_bboxes_fusion_folder, filename,
        #                            title, cm_classes, self.consensus_performance_metrics)        
        self.unanimous_performance_metrics.save_confusion_matrix(unanimous_bboxes_fusion_folder, 
                                                                 filename, title, cm_classes)        

        # saving performance metrics 
        cm_classes_for_metrics = self.classes[1:(self.number_of_classes+1)]
        self.unanimous_performance_metrics.save_performance_metrics(
            unanimous_bboxes_fusion_folder, filename, title, cm_classes_for_metrics, 
            self.iou_threshold_for_grouping, self.iou_threshold_for_inference, self.non_maximum_suppression,
            self.selected_models)

        # saving bounding boxes used in the fusion 
        self.unanimous_performance_metrics.save_inferenced_images(unanimous_bboxes_fusion_folder, filename)


    # def save_confusion_matrix(self, result_folder, filename, title,
    #                           cm_classes, model_performance_metrics):
    #     # setting to do not show graph in screen
    #     show_graph = False
              
    #     x_labels_names = cm_classes.copy()
    #     y_labels_names = cm_classes.copy()
    #     x_labels_names.append('Background False Positives')    
    #     y_labels_names.append('Undetected objects')
    #     format='.0f'
    #     path_and_filename = os.path.join(result_folder, filename + '_confusion_matrix.png')
    #     Utils.save_plot_confusion_matrix(model_performance_metrics.confusion_matrix, 
    #                                      path_and_filename, title, format,
    #                                      x_labels_names, y_labels_names, show_graph=show_graph)

    #     # path_and_filename = os.path.join(result_folder, filename + '_confusion_matrix_normalized.png')
    #     # Utils.save_plot_confusion_matrix(model_performance_metrics.normalized_confusion_matrix,
    #     #                                  path_and_filename, title, format,
    #     #                                  x_labels_names, y_labels_names, show_graph=show_graph)

    #     path_and_filename = os.path.join(result_folder, filename + '_confusion_matrix.xlsx')
    #     Utils.save_confusion_matrix_excel(
    #         model_performance_metrics.confusion_matrix,
    #         path_and_filename, x_labels_names, y_labels_names, 
    #         model_performance_metrics.tp_per_class,
    #         model_performance_metrics.fp_per_class,
    #         model_performance_metrics.fn_per_class,
    #         model_performance_metrics.tn_per_class
    #     )

