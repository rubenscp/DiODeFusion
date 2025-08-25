# ImageRelationshipMatrix Class
# This class represents the relationship matrix for one image

# Importing python modules
from common.manage_log import *

import os 
import json 
import torch 
import torchvision
from torchvision.ops import *  

class ImageRelationshipMatrix:
    def __init__(self, image_name=None, ground_truths_image=None,
                 model_1_name=None, predictions_image_model_1=None, model_2_name=None, predictions_image_model_2=None,
                 iou_threshold=0, a=0, b=0, c=0, d=0):
        self.image_name = image_name
        self.ground_truths_image = ground_truths_image
        self.model_1_name = model_1_name
        self.predictions_image_model_1 = predictions_image_model_1
        self.model_2_name = model_2_name
        self.predictions_image_model_2 = predictions_image_model_2
        self.iou_threshold = iou_threshold
        
        # Relationship matrix is related to the ground truth and the models pair
        # a + b + c + d = total number of ground truths bounding boxes of the image 
        self.a = a # DM1 and DM2 are correct related to GT 
        self.b = b # DM1 correct and DM2 wrong related to GT
        self.c = c # DM1 wrong and DM2 correct related to GT
        self.d = d # DM1 and DM2 are wrong related to GT

        self.number_of_gt_bboxes = 0

    def to_string(self):
        text = 'Image: ' + self.image_name + ' a: ' + str(self.a) + '  b: ' + str(
            self.b) + '  c: ' + str(self.c) + '  d: ' + str(self.d)
        return text

    def compute(self, iou_threshold):

        # compute the relationship matrix for one image using ground truths, and the 
        # predictions of the two models considering the class as grouped criteria

        # logging_info(f'')
        # logging_info(f'Computing relationship matrix for one image:')
        # logging_info(f'image_name: {self.image_name}')
        # logging_info(f'ground_truths_image: {self.ground_truths_image}')
        # logging_info(f'predictions_image_model_1: {self.predictions_image_model_1}')
        # logging_info(f'predictions_image_model_2: {self.predictions_image_model_2}')
        # logging_info(f'')

        # walking through the ground truths
        gt_bboxes = self.ground_truths_image['boxes']
        gt_labels = self.ground_truths_image['labels']
        gt_label_names = self.ground_truths_image['label_names']

        # print(f'rubens predictions_image_model_1: {self.predictions_image_model_1}')


        m1_bboxes = self.predictions_image_model_1['boxes']
        m1_labels = self.predictions_image_model_1['labels']
        m1_label_names = self.predictions_image_model_1['label_names']

        m2_bboxes = self.predictions_image_model_2['boxes']
        m2_labels = self.predictions_image_model_2['labels']
        m2_label_names = self.predictions_image_model_2['label_names']    

        # evaluate model predictions using the ground truth 
        self.evaluate_model_predictions_using_ground_truth(gt_bboxes, gt_labels,
                m1_bboxes, m1_labels, m2_bboxes, m2_labels, iou_threshold)


    def evaluate_model_predictions_using_ground_truth(self, 
            gt_bboxes, gt_labels,
            m1_bboxes, m1_labels, m2_bboxes, m2_labels, 
            iou_threshold):

        # # initializing objects 
        # m1_predictions_status = []
        # m2_predictions_status = []

        # setting number of ground truth bounding boxes 
        self.number_of_gt_bboxes = len(gt_bboxes)

        # looping through the ground truth bounding boxes
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):

            # print(f'rubens evaluate_bounding_box - img {self.image_name} - model 1: {self.model_1_name}')
            m1_detection_status = self.evaluate_bounding_box(
                                    gt_bbox, gt_label, m1_bboxes, m1_labels, iou_threshold)
            # print(f'rubens evaluate_bounding_box - img {self.image_name} - model 2: {self.model_2_name}')
            m2_detection_status = self.evaluate_bounding_box(
                                    gt_bbox, gt_label, m2_bboxes, m2_labels, iou_threshold)

            if m1_detection_status and m2_detection_status:
                self.a += 1
            if m1_detection_status and not m2_detection_status:
                self.b += 1
            if not m1_detection_status and m2_detection_status:
                self.c += 1
            if not m1_detection_status and not m2_detection_status:
                self.d += 1

            # logging_info(f'Relationship Matrix for image: {self.image_name}')
            # logging_info(f'a: {self.a}     b: {self.b}')
            # logging_info(f'c: {self.c}     d: {self.d}')    

    def evaluate_bounding_box(self, gt_bbox, gt_label, model_bboxes, model_labels, iou_threshold):

        # computing iou of the model bbox and groudn truth bbox 
        tensor_gt_bbox = torch.as_tensor([gt_bbox]) # add one new dimension

        # evaluating if model has bounding boxes
        if len(model_bboxes) <= 0:
            # print(f'rubens evaluate_bounding_box for image {self.image_name} - model: {model_labels} - no bounding boxes')
            # prediction wrong
            return False

        # evaluating if model has correct or wrong predictions  
        for i, (model_bbox, model_label) in enumerate(zip(model_bboxes, model_labels)):

            tensor_model_bbox = torch.as_tensor([model_bbox]) # add one new dimension

            # computing IoU of model bouding box and ground truth 
            iou = box_iou(tensor_gt_bbox, tensor_model_bbox).numpy()[0][0]

            # evaluating prediction is correct or wrong
            if iou >= iou_threshold and model_label == gt_label:
                # print(f'rubens prediction correct - image {self.image_name} - tensor_gt_bbox: {tensor_gt_bbox} versus tensor_model_bbox: {tensor_model_bbox}     model_label: {model_label} versus gt_label: {gt_label}      iou: {iou} versus iou_threshold: {iou_threshold}')
                # prediction correct
                return True
            else:
                # print(f'rubens prediction wrong - image {self.image_name} - tensor_gt_bbox: {tensor_gt_bbox} versus tensor_model_bbox: {tensor_model_bbox}     model_label: {model_label} versus gt_label: {gt_label}      iou: {iou} versus iou_threshold: {iou_threshold}')
                # prediction wrong 
                return False

    def save_json(self, dataset_name, model_1_name, model_2_name, iou_threshold, models_pair_name_folder):

        # check and create new folder for image relationship matrix 
        image_models_pair_name_folder = os.path.join(models_pair_name_folder, "image_rm")
        if not os.path.exists(image_models_pair_name_folder):
            Utils.create_directory(image_models_pair_name_folder)

        # initializing dictionary and variables 
        image_relationship_matrix_dic = {
            "dataset_name" : dataset_name,
            "model_1_name" : model_1_name,
            "model_2_name" : model_2_name,
            "image_name" : self.image_name,
            "number_of_gt_bboxes" : self.number_of_gt_bboxes,
            "relationship_matrix:" : {
                "a" : self.a,
                "b" : self.b,
                "c" : self.c,
                "d" : self.d
            }
        }

        # setting filename 
        json_filename = os.path.join(
            image_models_pair_name_folder,
            ("rm-" + self.image_name.replace(".jpg", "") + ".json")
        )

        # Save as JSON
        with open(json_filename, "w") as out_f:
                json.dump(image_relationship_matrix_dic, out_f, indent=2)            

