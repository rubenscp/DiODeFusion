# RelationshipMatrix Class
# This class represents the relationship matrix for two object detection models


# Importing python libraries 
import math
import copy 

# Importing python modules
from common.manage_log import *
from model_diversity.entity.ImageRelationshipMatrix import * 
import time   


class ModelsPairRelationshipMatrix:
    def __init__(self, dataset_name=None, 
                 model_1_name=None, model_1_predictions=None, model_1_short_filename=None,
                 model_2_name=None, model_2_predictions=None, model_2_short_filename=None,
                 iou_threshold=None, a=0, b=0, c=0, d=0, 
                 image_relationship_matrix_list=None, 
                 models_pair_name_folder=None):
        self.dataset_name = dataset_name
        self.model_1_name = model_1_name
        self.model_1_predictions = model_1_predictions
        self.model_1_short_filename = model_1_short_filename
        self.model_2_name = model_2_name
        self.model_2_predictions = model_2_predictions
        self.model_2_short_filename = model_2_short_filename
        self.iou_threshold_for_divertsity_measure = iou_threshold
        self.image_relationship_matrix_list = image_relationship_matrix_list
        self.a = a # DM1 and DM2 are correct related to GT 
        self.b = b # DM1 correct and DM2 wrong related to GT
        self.c = c # DM1 wrong and DM2 correct related to GT
        self.d = d # DM1 and DM2 are wrong related to GT

        self.a_norm = 0
        self.b_norm = 0
        self.c_norm = 0
        self.d_norm = 0

        self.models_pair_name_folder = models_pair_name_folder

        self.cor = 0
        self.dfm = 0
        self.dm = 0
        self.ia = 0
        self.qstat = 0       

    def to_string(self):
        text = 'model_name_1: ' + self.model_name_1 + \
               'model_name_2: ' + self.model_name_2 + \
               ' a: ' + str(self.a) + '  b: ' + str(self.b) + \
               ' c: ' + str(self.c) + '  d: ' + str(self.d)
        return text


    def compute(self):

        logging_info(f'')
        logging_info(f'Computing relationship matrix for two models and one dataset:')
        logging_info(f'Image dataset             : {self.dataset_name}')
        logging_info(f'Object detection model 1  : {self.model_1_name}')
        logging_info(f'Object detection model 2  : {self.model_2_name}')
        logging_info(f'IoU Threshold             : {self.iou_threshold_for_divertsity_measure}')
        logging_info(f'Models Pairs Matrix folder: {self.models_pair_name_folder}')
        logging_info(f'')

        print(f'')
        print(f'Computing relationship matrix for two models and one dataset:')
        print(f'Image dataset             : {self.dataset_name}')
        print(f'Object detection model 1  : {self.model_1_name}')
        print(f'Object detection model 2  : {self.model_2_name}')
        print(f'IoU Threshold             : {self.iou_threshold_for_divertsity_measure}')
        print(f'Models Pairs Matrix folder: {self.models_pair_name_folder}')
        print(f'')

        # creating one dictionary from all images list of the two models
        all_images = {}
        for key, value in self.model_1_predictions["images"].items():
            all_images[key] = ''
        # print(f'all_images apos carga do modelo 1 {self.model_1_name}: {len(all_images)}')

        for key, value in self.model_2_predictions["images"].items():
            all_images[key] = ''
        # print(f'all_images apos carga do modelo 2 {self.model_2_name}: {len(all_images)}')
        
        # number_of_images_all_models = len(all_images)
        # number_of_images_model_1 = len(self.model_1_predictions["images"])
        # number_of_images_model_2 = len(self.model_2_predictions["images"])  
        # print(f'Rubens Equalizing images in models {self.model_1_name} and {self.model_2_name}')
        # print(f'Rubens Before - Number of all images: {number_of_images_all_models}')
        # print(f'Number of model 1 images: {len(self.model_1_predictions["images"])}')
        # print(f'Number of model 2 images: {len(self.model_2_predictions["images"])}')

        # if number_of_images_all_models != number_of_images_model_1 or \
        #    number_of_images_all_models != number_of_images_model_2 or \
        #    number_of_images_model_1 != number_of_images_model_2:
        #     self.equalize_images_in_models(all_images)
        #     print(f'Rubens after')
        #     print(f'Number of model 1 images: {len(self.model_1_predictions["images"])}')
        #     print(f'Number of model 2 images: {len(self.model_2_predictions["images"])}')            


        # for each image, compute the relationship matrix based on the predictions of the 
        # two object detection models and the ground truth
        for image_name, _ in all_images.items():

            # getting predictions from the two images 
            ground_truths_image = self.model_1_predictions["images"][image_name]['ground_truths']
            predictions_image_model_1 = self.model_1_predictions["images"][image_name]['predictions']
            predictions_image_model_2 = self.model_2_predictions["images"][image_name]['predictions']

            # creating new instance of objects 
            image_relationship_matrix = ImageRelationshipMatrix()

            # setting fields
            image_relationship_matrix.image_name = image_name
            image_relationship_matrix.ground_truths_image = ground_truths_image
            image_relationship_matrix.model_1_name = self.model_1_name
            image_relationship_matrix.predictions_image_model_1 = predictions_image_model_1
            image_relationship_matrix.model_2_name = self.model_2_name
            image_relationship_matrix.predictions_image_model_2 = predictions_image_model_2
            image_relationship_matrix.iou_threshold = self.iou_threshold_for_divertsity_measure
            image_relationship_matrix.a = 0
            image_relationship_matrix.b = 0
            image_relationship_matrix.c = 0
            image_relationship_matrix.d = 0

            # computing relationship matrix for the models pair
            image_relationship_matrix.compute(self.iou_threshold_for_divertsity_measure)

            # saving image relationship matrix 
            image_relationship_matrix.save_json(self.dataset_name, self.model_1_name, self.model_2_name, self.iou_threshold_for_divertsity_measure, self.models_pair_name_folder)

            # adding image relationship matrix in the models pair 
            self.image_relationship_matrix_list.append(image_relationship_matrix)
            

        # computing the relationship matrix for the models pair
        self.compute_relationship_matrix_for_models_pair()

        # computing the relationship matrix for the models pair
        self.normalize_measures()

        # computing diversity measures
        self.cor = self.get_correlation_coefficient()
        self.dfm = self.get_double_fault_measure()
        self.dm = self.get_disagreement_measure()
        self.ia = self.get_interrater_agreement()
        self.qstat = self.get_q_statistic()
        

    # def equalize_images_in_models(self, all_images):
    #     # print(f'------------------------------------------------')
    #     # logging_info(f'Equalizing images in models {self.model_1_name} and {self.model_2_name}')
    #     # print(f'Rubens - Equalizing images in models {self.model_1_name} and {self.model_2_name}')

    #     # equalizing images in model 1
    #     for image_name, _ in all_images.items():
    #         if image_name not in self.model_1_predictions["images"]:
    #             # print(f'Image {image_name} not found in model 1 predictions')
    #             new_image = copy.deepcopy(self.model_2_predictions["images"][image_name])
    #             self.model_1_predictions["images"][image_name] = new_image
    #             self.model_1_predictions["images"][image_name]['predictions']['boxes'] = []
    #             self.model_1_predictions["images"][image_name]['predictions']['scores'] = []
    #             self.model_1_predictions["images"][image_name]['predictions']['labels'] = []
    #             self.model_1_predictions["images"][image_name]['predictions']['label_names'] = []

    #     # equalizing images in model 2
    #     for image_name, _ in all_images.items():
    #         if image_name not in self.model_2_predictions["images"]:
    #             # print(f'Image {image_name} not found in model 2 predictions')
    #             new_image = copy.deepcopy(self.model_1_predictions["images"][image_name])
    #             self.model_2_predictions["images"][image_name] = new_image
    #             self.model_2_predictions["images"][image_name]['predictions']['boxes'] = []
    #             self.model_2_predictions["images"][image_name]['predictions']['scores'] = []
    #             self.model_2_predictions["images"][image_name]['predictions']['labels'] = []
    #             self.model_2_predictions["images"][image_name]['predictions']['label_names'] = []
    #             # print(f'rubens adding image {image_name} to model 2 predictions: {self.model_2_predictions["images"][image_name]}')
    #             if image_name == 'MVI_6675.MP4#t=33.86.jpg':
    #                 print(f'rubens adding image {image_name} to model 1 predictions: {self.model_1_predictions["images"][image_name]}')
    #                 print(f'rubens adding image {image_name} to model 2 predictions: {self.model_2_predictions["images"][image_name]}')


    #     # print(f'------------------------------------------------')


    def compute_relationship_matrix_for_models_pair(self):

        # initializing relation matrix for the models pair 
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0

        # looping images relationhip matrix
        for image_rm in self.image_relationship_matrix_list:
            # print(f'image_rm {image_rm.image_name}')
            self.a += image_rm.a
            self.b += image_rm.b
            self.c += image_rm.c
            self.d += image_rm.d

        # logging_info(f'Relationship Matrix for the models pair: {self.model_1_name} and {self.model_2_name}')
        # logging_info(f'a: {self.a}     b: {self.b}')
        # logging_info(f'c: {self.c}     d: {self.d}')    

        print(f'Relationship Matrix Not Normalized for the models pair: {self.model_1_name} and {self.model_2_name} ')
        print(f'a: {self.a}     b: {self.b}')
        print(f'c: {self.c}     d: {self.d}')    

    def normalize_measures(self):
        print(f'')
        total = ( self.a + self.b + self.c + self.d ) * 1.0
        self.a_norm = self.a / total
        self.b_norm = self.b / total
        self.c_norm = self.c / total
        self.d_norm = self.d / total

        print(f'Relationship Matrix Normalized for the models pair: {self.model_1_name} and {self.model_2_name} ')
        print(f'total: {total}')
        print(f'a: {self.a_norm}     b: {self.b_norm}')
        print(f'c: {self.c_norm}     d: {self.d_norm}')    


    def print(self):
        logging_info(f'Relationship Matrix for the models pair: {self.model_1_name} and {self.model_2_name}')
        logging_info(f'a: {self.a}     b: {self.b}')
        logging_info(f'c: {self.c}     d: {self.d}')    

        print(f'Relationship Matrix for the models pair: {self.model_1_name} and {self.model_2_name}')
        print(f'a: {self.a}     b: {self.b}')
        print(f'c: {self.c}     d: {self.d}')    
        
    def get_correlation_coefficient(self):
        try:
            corr = ((self.a_norm * self.d_norm) - (self.b_norm * self.c_norm)) /    \
                    (math.sqrt( (self.a_norm + self.b_norm) * (self.c_norm + self.d_norm) * \
                            (self.a_norm + self.c_norm) * (self.b_norm + self.d_norm) ))
        except Exception as e:
            corr = 0.0
            logging_info(f"Error computing correlation coefficient: {e}")
            print(f"Error computing correlation coefficient: {e}")
            self.print()
        return corr

    def get_double_fault_measure(self):
        dfm = self.d_norm
        return dfm

    def get_disagreement_measure(self):
        dm = (self.b_norm + self.c_norm)  /  (self.a_norm + self.b_norm + self.c_norm + self.d_norm)
        return dm

    def get_interrater_agreement(self):
        ia = ( 2 * ( (self.a_norm * self.c_norm) - (self.b_norm * self.d_norm) ) ) /    \
               ( ((self.a_norm + self.b_norm) * (self.b_norm + self.d_norm)) + ((self.a_norm + self.c_norm) * (self.c_norm + self.d_norm)) )       
        return ia

    def get_q_statistic(self):
        try:
            qstat = ( (self.a_norm * self.d_norm) - (self.b_norm * self.c_norm) )  /  \
                    ( (self.a_norm * self.d_norm) + (self.b_norm * self.c_norm) )
        except Exception as e:
            qstat = 0.0
            logging_info(f"Error computing Q statistic: {e}")
            print(f"Error computing Q statistic: {e}")
            self.print()
        return qstat


    def save_json(self):

        # check and create new folder for image relationship matrix 
        models_pair_name_folder = os.path.join(self.models_pair_name_folder, "models_pair_rm")
        if not os.path.exists(models_pair_name_folder):
            Utils.create_directory(models_pair_name_folder)

        # initializing dictionary and variables 
        models_pair_relationship_matrix_dic = {
            "dataset_name" : self.dataset_name,
            "model_1_name" : self.model_1_name,
            "model_2_name" : self.model_2_name,
            "iou_threshold" : self.iou_threshold_for_divertsity_measure,
            "relationship_matrix:" : {
                "a" : self.a,
                "b" : self.b,
                "c" : self.c,
                "d" : self.d
            },
            "relationship_matrix_normalized:" : {
                "a_norm" : self.a_norm,
                "b_norm" : self.b_norm,
                "c_norm" : self.c_norm,
                "d_norm" : self.d_norm
            },
            "diversity_measures" : {
                "correlation_coefficient" : self.cor,
                "double_fault_measure" : self.dfm,
                "disagreement_measure" : self.dm,
                "interrater_agreement" : self.ia,
                "q_statistic" : self.qstat,
            }
        }

        # setting filename 
        # json_filename = os.path.join(
        #     models_pair_name_folder,
        #     ("models-pair-rm-" + self.dataset_name + "-" + self.model_1_name + "-" + self.model_2_name + ".json")
        # )

        # Get current time as a 6-digit number formatted as text
        time_text = time.strftime("%H%M%S")
        json_filename = os.path.join(
            models_pair_name_folder,
            ("models-pair-rm-" + self.dataset_name + "-" + time_text + ".json")
        )

        # Save as JSON
        with open(json_filename, "w") as out_f:
                json.dump(models_pair_relationship_matrix_dic, out_f, indent=2)            

