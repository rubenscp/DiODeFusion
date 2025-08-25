# ModelsDiversityMeasures Class
# This class represents the the diversity measures for the object detection models

# Importing python libraries 
# import math
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Importing python modules
from common.manage_log import *
from model_diversity.entity.ModelsPairRelationshipMatrix import * 

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class ModelsDiversityMeasures:
    def __init__(self, dataset_relationship_matrix_list=None, dataset_name=None, relationship_matrix_list=None, 
                 dataset_name_folder=None):
        self.dataset_relationship_matrix_list = { dataset_name: {} }
        # self.top_t = top_t
        # self.selected_models = selected_models
        
        self.dataset_model_performance_metrics_dic = {}
        self.dataset_diversity_measures_dic = {}
        self.diversity_measures = []
        self.diversity_measures_df = pd.DataFrame()
        self.models_occurence_results_df = pd.DataFrame()
        self.models_occurence_results_df_sorted = pd.DataFrame()

    # def to_string(self):
    #     text = 'dataset_name: ' + self.dataset_name + \
    #            'relationship_matrix_list: ' + len(self.relationship_matrix_list)               
    #    return text

    def add_models_pair_relationship_matrix(self, dataset_name, models_pair_relationship_matrix):
        # print(f'adicionou models_pair_relationship_matrix')
        models_pair_name = models_pair_relationship_matrix.model_1_name + '-x-' + models_pair_relationship_matrix.model_2_name
        # print(f'models_pair_name (key): {models_pair_name}')        
        self.dataset_relationship_matrix_list[dataset_name][models_pair_name] = {}
        self.dataset_relationship_matrix_list[dataset_name][models_pair_name] = models_pair_relationship_matrix

    def create_diversity_measures(self):
        
        self.dataset_diversity_measures = {}

        # looping models pair relationship matrix 
        for dataset_name_key, models_pair_relationship_matrix in self.dataset_relationship_matrix_list.items():
            print(f'dataset_name_key: {dataset_name_key}')
            print(f'models_pair_relationship_matrix: {models_pair_relationship_matrix}')

            # creating new dicionary for dataset_name_key
            self.dataset_diversity_measures[dataset_name_key] = {}
            self.dataset_diversity_measures[dataset_name_key]['correlation_coefficient'] = {}
            self.dataset_diversity_measures[dataset_name_key]['double_fault_measure'] = {}
            self.dataset_diversity_measures[dataset_name_key]['disagreement_measure'] = {}
            self.dataset_diversity_measures[dataset_name_key]['interrater_agreement'] = {}
            self.dataset_diversity_measures[dataset_name_key]['q_statistic'] = {}
            self.dataset_diversity_measures[dataset_name_key]['model_1_f1_score'] = {}
            self.dataset_diversity_measures[dataset_name_key]['model_2_f1_score'] = {}

            # looping models pair relationship matrix 
            for models_pair_key, models_pair_relationship_matrix_item in models_pair_relationship_matrix.items():

                # creating new dicionary for diversity measures
                self.dataset_diversity_measures[dataset_name_key]['correlation_coefficient'][models_pair_key] = {}
                self.dataset_diversity_measures[dataset_name_key]['correlation_coefficient'][models_pair_key] = models_pair_relationship_matrix_item.cor
                            
                self.dataset_diversity_measures[dataset_name_key]['double_fault_measure'][models_pair_key] = {}
                self.dataset_diversity_measures[dataset_name_key]['double_fault_measure'][models_pair_key] = models_pair_relationship_matrix_item.dfm
                
                self.dataset_diversity_measures[dataset_name_key]['disagreement_measure'][models_pair_key] = {}
                self.dataset_diversity_measures[dataset_name_key]['disagreement_measure'][models_pair_key] = models_pair_relationship_matrix_item.dm
                
                self.dataset_diversity_measures[dataset_name_key]['interrater_agreement'][models_pair_key] = {}
                self.dataset_diversity_measures[dataset_name_key]['interrater_agreement'][models_pair_key] = models_pair_relationship_matrix_item.ia
                
                self.dataset_diversity_measures[dataset_name_key]['q_statistic'][models_pair_key] = {}
                self.dataset_diversity_measures[dataset_name_key]['q_statistic'][models_pair_key] = models_pair_relationship_matrix_item.qstat
                
                dataset_model = dataset_name_key + '-' + 'model_' + models_pair_relationship_matrix_item.model_1_name
                model_1_f1_score = self.dataset_model_performance_metrics_dic[dataset_model].get_model_f1_score()
                self.dataset_diversity_measures[dataset_name_key]['model_1_f1_score'][models_pair_key] = {}
                self.dataset_diversity_measures[dataset_name_key]['model_1_f1_score'][models_pair_key] = model_1_f1_score
    
                dataset_model = dataset_name_key + '-' + 'model_' + models_pair_relationship_matrix_item.model_2_name
                model_2_f1_score = self.dataset_model_performance_metrics_dic[dataset_model].get_model_f1_score()
                self.dataset_diversity_measures[dataset_name_key]['model_2_f1_score'][models_pair_key] = {}
                self.dataset_diversity_measures[dataset_name_key]['model_2_f1_score'][models_pair_key] = model_2_f1_score

        print(f'self.dataset_diversity_measures: {self.dataset_diversity_measures}')
    
    def create_summary_diversity_measures(self, matrices_folder):
        
        # creating new list 
        self.diversity_measures = []        

        # setting column names 
        column_names = ['dataset_name', 'model_1', 'model_2',
                'correlation_coefficient', 'double_fault_measure', 'disagreement_measure',
                'interrater_agreement', 'q_statistic', 'model_1_f1_score', 'model_2_f1_score',
                'average_f1_score']        

        # looping models pair relationship matrix 
        for dataset_name_key, models_pair_relationship_matrix in self.dataset_relationship_matrix_list.items():
            print(f'dataset_name_key: {dataset_name_key}')
            print(f'models_pair_relationship_matrix: {models_pair_relationship_matrix}')
        
            # looping models pair relationship matrix 
            for models_pair_key, models_pair_relationship_matrix_item in models_pair_relationship_matrix.items():

                # getting performance metrics for models 
                dataset_model = dataset_name_key + '-' + 'model_' + models_pair_relationship_matrix_item.model_1_name
                model_1_precision = self.dataset_model_performance_metrics_dic[dataset_model].get_model_precision()
                model_1_recall = self.dataset_model_performance_metrics_dic[dataset_model].get_model_recall()
                model_1_f1_score = self.dataset_model_performance_metrics_dic[dataset_model].get_model_f1_score()

                dataset_model = dataset_name_key + '-' + 'model_' + models_pair_relationship_matrix_item.model_2_name
                model_2_precision = self.dataset_model_performance_metrics_dic[dataset_model].get_model_precision()
                model_2_recall = self.dataset_model_performance_metrics_dic[dataset_model].get_model_recall()
                model_2_f1_score = self.dataset_model_performance_metrics_dic[dataset_model].get_model_f1_score()

                average_f1_score = (model_1_f1_score + model_2_f1_score) / 2

                # creating new item 
                item = [dataset_name_key, 
                        models_pair_relationship_matrix_item.model_1_name,
                        models_pair_relationship_matrix_item.model_2_name,
                        models_pair_relationship_matrix_item.cor,
                        models_pair_relationship_matrix_item.dfm,
                        models_pair_relationship_matrix_item.dm,
                        models_pair_relationship_matrix_item.ia,
                        models_pair_relationship_matrix_item.qstat,
                        model_1_f1_score,
                        model_2_f1_score,
                        average_f1_score
                        ]

                # adding new item 
                self.diversity_measures.append(item)

        # saving list 

        # setting filename 
        path_and_filename = os.path.join(
            matrices_folder,
            "diversity-measures.xlsx"
        )

        # creating dataframe from list 
        self.diversity_measures_df = pd.DataFrame(self.diversity_measures, columns=column_names)
        # print(f'diversity measures dataframe')
        # print(self.diversity_measures_df.head())

        # writing excel file from dataframe
        self.diversity_measures_df.to_excel(path_and_filename, sheet_name='diversity_measures', index=False)

              
    def save_json(self, matrices_folder):

        print(f'matrices_folder: {matrices_folder}')

        models_diversity_measures_dic = {}

        # initializing dictionary and variables 
        for dataset_name_key, models_pair_relationship_matrix in self.dataset_relationship_matrix_list.items():
            print(f'dataset_name_key: {dataset_name_key}')
            print(f'models_pair_relationship_matrix: {models_pair_relationship_matrix}')

            # check and create new folder for image relationship matrix 
            dataset_name_folder = os.path.join(matrices_folder, dataset_name_key)
            if not os.path.exists(dataset_name_folder):
                Utils.create_directory(dataset_name_folder)

            # creating new dicionary for dataset_name_key
            models_diversity_measures_dic[dataset_name_key] = {}

            # looping models pair relationship matrix 
            for models_pair_key, models_pair_relationship_matrix_item in models_pair_relationship_matrix.items():
                # print(f'')
                print(f'models_pair_key: {models_pair_key}')
                # print(f'models_pair_relationship_matrix_item.cor: {models_pair_relationship_matrix_item.cor}')
                # print(f'models_pair_relationship_matrix_item.dfm: {models_pair_relationship_matrix_item.dfm}')
                # print(f'models_pair_relationship_matrix_item.dm: {models_pair_relationship_matrix_item.dm}')
                # print(f'models_pair_relationship_matrix_item.ia: {models_pair_relationship_matrix_item.ia}')
                # print(f'models_pair_relationship_matrix_item.qsta: {models_pair_relationship_matrix_item.qstat}')

                # getting performance metrics for models 
                print(f'self.dataset_model_performance_metrics_dic.keys: {self.dataset_model_performance_metrics_dic.keys()}')
                dataset_model = dataset_name_key + '-' + 'model_' + models_pair_relationship_matrix_item.model_1_name
                model_1_precision = self.dataset_model_performance_metrics_dic[dataset_model].get_model_precision()
                model_1_recall = self.dataset_model_performance_metrics_dic[dataset_model].get_model_recall()
                model_1_f1_score = self.dataset_model_performance_metrics_dic[dataset_model].get_model_f1_score()

                dataset_model = dataset_name_key + '-' + 'model_' + models_pair_relationship_matrix_item.model_2_name
                model_2_precision = self.dataset_model_performance_metrics_dic[dataset_model].get_model_precision()
                model_2_recall = self.dataset_model_performance_metrics_dic[dataset_model].get_model_recall()
                model_2_f1_score = self.dataset_model_performance_metrics_dic[dataset_model].get_model_f1_score()

                # creating new dicionary for dataset_name_key                
                models_diversity_measures_dic[dataset_name_key][models_pair_key] = {}
                models_diversity_measures_dic[dataset_name_key][models_pair_key] = {
                    models_pair_relationship_matrix_item.model_1_name : {
                        "precision" : model_1_precision,
                        "recall" : model_1_recall,
                        "f1-score" : model_1_f1_score,
                    },
                    models_pair_relationship_matrix_item.model_2_name : {
                        "precision" : model_2_precision,
                        "recall" : model_2_recall,
                        "f1-score" : model_2_f1_score,
                    },
                    "correlation_coefficient" : models_pair_relationship_matrix_item.cor,
                    "double_fault_measure" : models_pair_relationship_matrix_item.dfm,
                    "disagreement_measure" : models_pair_relationship_matrix_item.dm,
                    "interrater_agreement" : models_pair_relationship_matrix_item.ia,
                    "q_statistic" : models_pair_relationship_matrix_item.qstat,
                }               
              
            # setting filename 
            json_filename = os.path.join(
                dataset_name_folder,
                ("models-diversity-" + dataset_name_key + ".json")
            )

            # Save as JSON
            with open(json_filename, "w") as out_f:
                    json.dump(models_diversity_measures_dic, out_f, indent=2)


    def create_graphs_diversity_measures(self, matrices_folder):

        logging_info(f'')
        logging_info(f'Creating graphs for diversity measures')
        # print(f'matrices_folder: {matrices_folder}')

        # setting show graph status 
        show_graph = False

        title = ''

        # setting labels for graph points 
        labels_text, labels_text_full = self.get_labels_text(self.diversity_measures_df['model_1'], self.diversity_measures_df['model_2'])

        # setting dataset names for graph title 
        dataset_names = self.get_dataset_names(self.diversity_measures_df['dataset_name'])

        # plotting graph of correlation coeficient versus average F1-score
        diversity_measure = 'correlation_coefficient' 
        average_f1_score  = 'average_f1_score' 
        filename = os.path.join(matrices_folder, diversity_measure + "-x-" + average_f1_score + ".png")
        x_label = "Correlation Coefficient (COR)"
        y_label = "Average F1-score"
        # title = "Correlation coefficient (low value) versus Average F1-score " + LINE_FEED + \
                # "Dataset: " + str(dataset_names)
        xlim_low = -0.1
        xlim_high = 1.0
        self.create_graph(self.diversity_measures_df[diversity_measure], 
                          self.diversity_measures_df[average_f1_score], 
                          filename, title, x_label, y_label, labels_text,  
                          xlim_low, xlim_high, show_graph
                          )
        
        # plotting graph of double fault measure versus average F1-score
        diversity_measure = 'double_fault_measure' 
        average_f1_score  = 'average_f1_score' 
        filename = os.path.join(matrices_folder, diversity_measure + "-x-" + average_f1_score + ".png")
        x_label = "Double Fault Measure (DFM)"
        y_label = "Average F1-score"
        # title = "Double fault measure (low value) versus Average F1-score " + LINE_FEED + \
        #         "Dataset: " + str(dataset_names)
        xlim_low = 0.0
        xlim_high = 1.0
        self.create_graph(self.diversity_measures_df[diversity_measure], 
                          self.diversity_measures_df[average_f1_score], 
                          filename, title, x_label, y_label, labels_text, 
                          xlim_low, xlim_high, show_graph
                          )

        # diversity_measure = 'double_fault_measure' 
        # average_f1_score  = 'average_f1_score' 
        # filename = os.path.join(matrices_folder, diversity_measure + "-x-" +
        #                         average_f1_score + "-compact.png")
        # x_label = ''
        # y_label = ''
        # title = ''
        # xlim_low = 0.1
        # xlim_high = 0.5
        # self.create_graph(self.diversity_measures_df[diversity_measure], 
        #                   self.diversity_measures_df[average_f1_score], 
        #                   filename, title, x_label, y_label, labels_text, 
        #                   xlim_low, xlim_high, show_graph
        #                   )

        # plotting graph of disagreement_measure versus average F1-score
        diversity_measure = 'disagreement_measure' 
        average_f1_score  = 'average_f1_score' 
        filename = os.path.join(matrices_folder, diversity_measure + "-x-" + average_f1_score + ".png")
        x_label = "Disagreement Measure (DM)"
        y_label = "Average F1-score"
        # title = "Disagreement measure (high value) versus Average F1-score" + LINE_FEED + \
        #         "Dataset: " + str(dataset_names)
        xlim_low = -0.1
        xlim_high = 1.0
        self.create_graph(self.diversity_measures_df[diversity_measure], 
                          self.diversity_measures_df[average_f1_score], 
                          filename, title, x_label, y_label, labels_text, 
                          xlim_low, xlim_high, show_graph
                          )

        # plotting graph of interrater_agreement versus average F1-score
        diversity_measure = 'interrater_agreement' 
        average_f1_score  = 'average_f1_score' 
        filename = os.path.join(matrices_folder, diversity_measure + "-x-" + average_f1_score + ".png")
        x_label = "Interrater Agreement (IA)"
        y_label = "Average F1-score"
        # title = "Interrater agreement (low value) versus Average F1-score" + LINE_FEED + \
        #         "Dataset: " + str(dataset_names)
        xlim_low = -0.1
        xlim_high = 1.0
        self.create_graph(self.diversity_measures_df[diversity_measure], 
                          self.diversity_measures_df[average_f1_score], 
                          filename, title, x_label, y_label, labels_text, 
                          xlim_low, xlim_high, show_graph
                          )           

        # plotting graph of q_statistic versus average F1-score
        diversity_measure = 'q_statistic' 
        average_f1_score  = 'average_f1_score' 
        filename = os.path.join(matrices_folder, diversity_measure + "-x-" + average_f1_score + ".png")
        x_label = "Q-Statistic"
        y_label = "Average F1-score"
        # title = "Q-Statistic (low value) versus Average F1-score " + LINE_FEED + \
        #         "Dataset: " + str(dataset_names)
        xlim_low = -0.1
        xlim_high = 1.0
        self.create_graph(self.diversity_measures_df[diversity_measure], 
                          self.diversity_measures_df[average_f1_score], 
                          filename, title, x_label, y_label, labels_text, 
                          xlim_low, xlim_high, show_graph
                          )

                
    def create_graph(self, df_x, df_y, filename, title, x_label, y_label, labels_text, 
                     xlim_low, xlim_high,
                     show_graph):

        # plt.figure(figsize=(15,7))
        plt.figure()

        # Step 2: Create a scatter plot
        # plt.figure(figsize=(8, 6))  # Optional: Set the figure size
        plt.scatter(df_x, df_y, color='blue', marker='o')

        # Step 3: Add labels and title
        plt.xlabel(x_label, fontsize=8)
        plt.ylabel(y_label, fontsize=8)
        plt.title(title, fontsize=8)
        plt.xlim(xlim_low, xlim_high)
        plt.ylim(0.5, 1.0)

        # Optional: Add point labels
        texts = []
        for row_x, row_y, label in zip(df_x, df_y, labels_text):
            # plt.text(row_x + 0.015, row_y, label, fontsize=6)
            texts.append(plt.text(row_x + 0.015, row_y, label, fontsize=6))

        # adjusting label text to avoid overlapping 
        adjust_text(texts, 
                    arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
                    expand_text=(1.5, 1.5),  # allow more space around texts
                    expand_points=(2.0, 2.0),  # increase distance from points
                    )

        # set grid on 
        # plt.grid()

        # Step 4: Save the plot to a file
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save to PNG file

        # Step 5: Show the plot (optional)
        if show_graph:
            plt.show()
        
        # closing plotter for next graph
        plt.close()

    # def create_graph(self, df_x, df_y, filename, title, x_label, y_label, labels_text, show_graph):

    #     # plt.figure(figsize=(15,7))
    #     plt.figure()

    #     # Step 2: Create a scatter plot
    #     # plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    #     plt.scatter(df_x, df_y, color='blue', marker='o')

    #     # Step 3: Add labels and title
    #     plt.xlabel(x_label, fontsize=8)
    #     plt.ylabel(y_label, fontsize=8)
    #     plt.title(title, fontsize=8)
    #     plt.xlim(-0.1, 1.0)
    #     plt.ylim(0.5, 1.0)

    #     # Optional: Add point labels
    #     for row_x, row_y, label in zip(df_x, df_y, labels_text):
    #         plt.text(row_x + 0.015, row_y, label, fontsize=6)

    #     # set grid on 
    #     # plt.grid()

    #     # Step 4: Save the plot to a file
    #     plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save to PNG file

    #     # Step 5: Show the plot (optional)
    #     if show_graph:
    #         plt.show()
        
    #     # closing plotter for next graph
    #     plt.close()

    def get_labels_text(self, model_1_list, model_2_list):

        # creating list of labels text 
        labels_text = []
        labels_text_full = []

        print(f'get labels - model_1_list: {model_1_list}    model_2_list: {model_2_list}')

        # looping rows of model 1 and model 2 to create labels text 
        for model_1, model_2 in zip(model_1_list, model_2_list):
            
            print(f'get labels - model_1: {model_1}    model_2: {model_2}')
            model_1_label = ''
            model_2_label = ''

            if model_1 == 'ssd': 
                model_1_label = 's'
                model_1_label_full = 'SSD'
            if model_1 == 'faster_rcnn': 
                model_1_label = 'f'
                model_1_label_full = 'Faster R-CNN'
            if model_1 == 'yolov8': 
                model_1_label = 'y8' 
                model_1_label_full = 'YOLOv8'
            if model_1 == 'yolov9': 
                model_1_label = 'y9' 
                model_1_label_full = 'YOLOv9'
            if model_1 == 'yolov10': 
                model_1_label = 'y10' 
                model_1_label_full = 'YOLOv10'
            if model_1 == 'detr': 
                model_1_label = 'd' 
                model_1_label_full = 'DETR'
            if model_1 == 'transunet': 
                model_1_label = 'tu' 
                model_1_label_full = 'TransUNet'
            if model_1 == 'cascade_rcnn_r101': 
                model_1_label = '11' 
                model_1_label_full = 'cascade_rcnn_r101'
            if model_1 == 'deformable_detr': 
                model_1_label = '12' 
                model_1_label_full = 'deformable_detr'
            if model_1 == 'faster_rcnn_nauplus_101': 
                model_1_label = '13' 
                model_1_label_full = 'faster_rcnn_nauplus_101'
            if model_1 == 'faster_rcnn_nauplus': 
                model_1_label = '14' 
                model_1_label_full = 'faster_rcnn_nauplus'
            if model_1 == 'fast_retinanet_r101_fpn_2x_coco': 
                model_1_label = '15' 
                model_1_label_full = 'fast_retinanet_r101_fpn_2x_coco'
            if model_1 == 'fast_retinanet_r50_2X': 
                model_1_label = '16' 
                model_1_label_full = 'fast_retinanet_r50_2X'
            if model_1 == 'fast_sparse-rcnn_r101_fpn_ms-480-800-3x_coco': 
                model_1_label = '17' 
                model_1_label_full = 'fast_sparse-rcnn_r101_fpn_ms-480-800-3x_coco'
            if model_1 == 'fast_sparse-rcnn_r50_fpn_ms-480-800-3x_coco': 
                model_1_label = '18' 
                model_1_label_full = 'fast_sparse-rcnn_r50_fpn_ms-480-800-3x_coco'
            if model_1 == 'fast_ssd300': 
                model_1_label = '19' 
                model_1_label_full = 'fast_ssd300'
            if model_1 == 'fast_ssd512': 
                model_1_label = '20' 
                model_1_label_full = 'fast_ssd512'
            if model_1 == 'fast_yolov3': 
                model_1_label = '21' 
                model_1_label_full = 'fast_yolov3'
            if model_1 == 'fast_yolox_s_8xb8-300e_coco': 
                model_1_label = '22' 
                model_1_label_full = 'fast_yolox_s_8xb8-300e_coco'
            if model_1 == 'yolov5su': 
                model_1_label = '23' 
                model_1_label_full = 'yolov5su'
            if model_1 == 'yolov7': 
                model_1_label = '24' 
                model_1_label_full = 'yolov7'
            if model_1 == 'yolov8n': 
                model_1_label = '25' 
                model_1_label_full = 'yolov8n'


            if model_2 == 'ssd': 
                model_2_label = 's'
                model_2_label_full = 'SSD'
            if model_2 == 'faster_rcnn': 
                model_2_label = 'f'
                model_2_label_full = 'Faster R-CNN'
            if model_2 == 'yolov8': 
                model_2_label = 'y8' 
                model_2_label_full = 'YOLOv8'
            if model_2 == 'yolov9': 
                model_2_label = 'y9' 
                model_2_label_full = 'YOLOv9'
            if model_2 == 'yolov10': 
                model_2_label = 'y10' 
                model_2_label_full = 'YOLOv10'
            if model_2 == 'detr': 
                model_2_label = 'd' 
                model_2_label_full = 'DETR'
            if model_2 == 'transunet': 
                model_2_label = 'tu' 
                model_2_label_full = 'TransUNet'
            if model_2 == 'cascade_rcnn_r101': 
                model_2_label = '11' 
                model_2_label_full = 'cascade_rcnn_r101'
            if model_2 == 'deformable_detr': 
                model_2_label = '12' 
                model_2_label_full = 'deformable_detr'
            if model_2 == 'faster_rcnn_nauplus_101': 
                model_2_label = '13' 
                model_2_label_full = 'faster_rcnn_nauplus_101'
            if model_2 == 'faster_rcnn_nauplus': 
                model_2_label = '14' 
                model_2_label_full = 'faster_rcnn_nauplus'
            if model_2 == 'fast_retinanet_r101_fpn_2x_coco': 
                model_2_label = '15' 
                model_2_label_full = 'fast_retinanet_r101_fpn_2x_coco'
            if model_2 == 'fast_retinanet_r50_2X': 
                model_2_label = '16' 
                model_2_label_full = 'fast_retinanet_r50_2X'
            if model_2 == 'fast_sparse-rcnn_r101_fpn_ms-480-800-3x_coco': 
                model_2_label = '17' 
                model_2_label_full = 'fast_sparse-rcnn_r101_fpn_ms-480-800-3x_coco'
            if model_2 == 'fast_sparse-rcnn_r50_fpn_ms-480-800-3x_coco': 
                model_2_label = '18' 
                model_2_label_full = 'fast_sparse-rcnn_r50_fpn_ms-480-800-3x_coco'
            if model_2 == 'fast_ssd300': 
                model_2_label = '19' 
                model_2_label_full = 'fast_ssd300'
            if model_2 == 'fast_ssd512': 
                model_2_label = '20' 
                model_2_label_full = 'fast_ssd512'
            if model_2 == 'fast_yolov3': 
                model_2_label = '21' 
                model_2_label_full = 'fast_yolov3'
            if model_2 == 'fast_yolox_s_8xb8-300e_coco': 
                model_2_label = '22' 
                model_2_label_full = 'fast_yolox_s_8xb8-300e_coco'
            if model_2 == 'yolov5su': 
                model_2_label = '23' 
                model_2_label_full = 'yolov5su'
            if model_2 == 'yolov7': 
                model_2_label = '24' 
                model_2_label_full = 'yolov7'
            if model_2 == 'yolov8n': 
                model_2_label = '25' 
                model_2_label_full = 'yolov8n'                

            labels_text.append(model_1_label + '-' + model_2_label)
            labels_text_full.append(model_1_label_full + '-' + model_2_label_full)

        print(f'labels_text: {labels_text}')

        # returing labels text 
        return labels_text, labels_text_full

    def get_dataset_names(self, dataset_names_list):

        # creating list of labels text 
        dataset_names = []

        # looping rows of model 1 and model 2 to create labels text 
        for dataset_name in dataset_names_list:            
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)

        # returing dataset names 
        return dataset_names

    # # Method 1 for selecting object dectection models 
    # # Description: Selection of object detection models based on all diversity measures together and no tiebreak criteria
    # def object_detection_models_selection_method_1(self, method_short_name, method_description):

    #     logging_info(f'Processing object detection model selection {method_short_name}: {method_description}')

    #     # returning results 
    #     return None 
    

    # # Method 2 for selecting object dectection models 
    # # Description: Selection of detection models based on single diversity measure and tiebreak criteria
    # def object_detection_models_selection_method_2(self, method_short_name, method_description):

    #     logging_info(f'Processing object detection model selection {method_short_name}: {method_description}')

    #     # returning results 
    #     return None 
    
    # # Method 3 for selecting object dectection models 
    # # Description: Selection of object detection models based on single diversity measure, no tiebreak criteria, and effectiveness-driven detector filter
    # def object_detection_models_selection_method_3(self, method_short_name, method_description):

    #     logging_info(f'Processing object detection model selection {method_short_name}: {method_description}')

    #     # returning results 
    #     return None     
    

    # def rank_object_detection_models(self, matrices_folder, models_dic):

    #     logging_info(f'Creating rank of Object Detectin Models')
    #     logging_info(f'')
        
    #     # Set display option to show all columns
    #     pd.set_option('display.max_columns', None)

    #     # creating diversity measure lists 
    #     cor_df = self.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'correlation_coefficient']]
    #     dfm_df = self.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'double_fault_measure']]
    #     dm_df = self.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'disagreement_measure']]
    #     ia_df = self.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'interrater_agreement']]
    #     qstat_df = self.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'q_statistic']]

    #     # print(cor_df.head())
    #     # print(dfm_df.head())
    #     # print(dm_df.head())
    #     # print(ia_df.head())
    #     # print(qstat_df.head())

    #     # sorting measures list to get the higher diversity among the detectors

    #     # Correlation coefficient - Lower value >> better diversity >> Ascending
    #     # Double Fault Measure - Lower value >> better diversity >>Ascending
    #     # Disagreement Measure - Higher value >> better diversity >> Descending
    #     # Interrater Agreement (κ) - Lower value >> better diversity >> Ascending
    #     # Q-statistic - Lower value >> better diversity >> Ascending 
    #     cor_df_sorted = cor_df.sort_values(by=['correlation_coefficient'], ascending=True)
    #     dfm_df_sorted = dfm_df.sort_values(by=['double_fault_measure'], ascending=True)
    #     dm_df_sorted = dm_df.sort_values(by=['disagreement_measure'], ascending=False)
    #     ia_df_sorted = ia_df.sort_values(by=['interrater_agreement'], ascending=True)
    #     qstat_df_sorted = qstat_df.sort_values(by=['q_statistic'], ascending=True)

    #     logging_info(f'Diversity Measures after sorting')
    #     logging_info(cor_df_sorted.head())
    #     logging_info(dfm_df_sorted.head())
    #     logging_info(dm_df_sorted.head())
    #     logging_info(ia_df_sorted.head())
    #     logging_info(qstat_df_sorted.head())

    #     print(f'Diversity Measures after sorting')
    #     print(cor_df_sorted.head())
    #     print(f'')
    #     print(dfm_df_sorted.head())
    #     print(f'')
    #     print(dm_df_sorted.head())
    #     print(f'')
    #     print(ia_df_sorted.head())
    #     print(f'')
    #     print(qstat_df_sorted.head())
    #     print(f'')

    #     # built occurences matrix to select the t top object detection models
    #     logging_info(f'')
    #     logging_info(f'Builting occurences matrix for the object detection models')
    #     logging_info(f'')

    #     print(f'')
    #     print(f'Builting occurences matrix for the object detection models')
    #     print(f'')

    #     models_occurences_results_list = []
    #     row_labels = []
    #     for model_key, model_value in models_dic.items():
    #         if model_value[1]:
    #             row_labels.append(model_key)
    #             item = [0, 0, 0, 0, 0, 0]
    #             models_occurences_results_list.append(item)

    #     column_labels = ['correlation_coefficient', 'double_fault_measure', 'disagreement_measure',
    #                     'interrater_agreement', 'q_statistic', 'total']

    #     # print(f'row_labels: {row_labels}')
    #     # print(f'column_labels: {column_labels}')
    #     # print(f'models_occurences_results_list: {models_occurences_results_list}')
        
    #     # creating dataframe from list 
    #     self.models_occurence_results_df = pd.DataFrame(
    #             models_occurences_results_list, index=row_labels, columns=column_labels)

    #     # print(self.models_occurence_results_df.head())  
    #     # print(f'--------------------------------------')              
    
    #     # counting the ranked detectors for votation scheme 
    #     logging_info(f'Counting ranked detectors for votation scheme - top-t: {self.top_t}')
    #     logging_info(f'')

    #     print(f'Counting ranked detectors for votation scheme - top-t: {self.top_t}')
    #     print(f'')

    #     # counting correlation coefficient 
    #     count_t = 0
    #     for index, row in cor_df_sorted.iterrows():
    #         if count_t >= self.top_t:
    #             print(f'stop at {count_t}')
    #             break

    #         count_t += 1
    #         dataset_name = row['dataset_name']
    #         model_1 = row['model_1']
    #         model_2 = row['model_2']
    #         correlation_coefficient = row['correlation_coefficient']          

    #         number_of = self.models_occurence_results_df.loc[model_1].loc['correlation_coefficient']
    #         self.models_occurence_results_df.loc[model_1].loc['correlation_coefficient'] = number_of + 1
    #         number_of = self.models_occurence_results_df.loc[model_2].loc['correlation_coefficient']
    #         self.models_occurence_results_df.loc[model_2].loc['correlation_coefficient'] = number_of + 1

    #     # counting double fault measure 
    #     count_t = 0
    #     for index, row in dfm_df_sorted.iterrows():
    #         if count_t >= self.top_t:
    #             print(f'stop at {count_t}')
    #             break

    #         count_t += 1
    #         dataset_name = row['dataset_name']
    #         model_1 = row['model_1']
    #         model_2 = row['model_2']
    #         double_fault_measure = row['double_fault_measure']          

    #         number_of = self.models_occurence_results_df.loc[model_1].loc['double_fault_measure']
    #         self.models_occurence_results_df.loc[model_1].loc['double_fault_measure'] = number_of + 1
    #         number_of = self.models_occurence_results_df.loc[model_2].loc['double_fault_measure']
    #         self.models_occurence_results_df.loc[model_2].loc['double_fault_measure'] = number_of + 1

    #     # counting disagreement measure
    #     count_t = 0
    #     for index, row in dm_df_sorted.iterrows():
    #         if count_t >= self.top_t:
    #             print(f'stop at {count_t}')
    #             break

    #         count_t += 1
    #         dataset_name = row['dataset_name']
    #         model_1 = row['model_1']
    #         model_2 = row['model_2']
    #         disagreement_measure = row['disagreement_measure']          

    #         number_of = self.models_occurence_results_df.loc[model_1].loc['disagreement_measure']
    #         self.models_occurence_results_df.loc[model_1].loc['disagreement_measure'] = number_of + 1
    #         number_of = self.models_occurence_results_df.loc[model_2].loc['disagreement_measure']
    #         self.models_occurence_results_df.loc[model_2].loc['disagreement_measure'] = number_of + 1

    #     # counting interrater agreement
    #     count_t = 0
    #     for index, row in ia_df_sorted.iterrows():
    #         if count_t >= self.top_t:
    #             print(f'stop at {count_t}')
    #             break

    #         count_t += 1
    #         dataset_name = row['dataset_name']
    #         model_1 = row['model_1']
    #         model_2 = row['model_2']
    #         interrater_agreement = row['interrater_agreement']          

    #         number_of = self.models_occurence_results_df.loc[model_1].loc['interrater_agreement']
    #         self.models_occurence_results_df.loc[model_1].loc['interrater_agreement'] = number_of + 1
    #         number_of = self.models_occurence_results_df.loc[model_2].loc['interrater_agreement']
    #         self.models_occurence_results_df.loc[model_2].loc['interrater_agreement'] = number_of + 1

    #     # counting q statistic
    #     count_t = 0
    #     for index, row in qstat_df_sorted.iterrows():
    #         if count_t >= self.top_t:
    #             print(f'stop at {count_t}')
    #             break

    #         count_t += 1
    #         dataset_name = row['dataset_name']
    #         model_1 = row['model_1']
    #         model_2 = row['model_2']
    #         q_statistic = row['q_statistic']      

    #         number_of = self.models_occurence_results_df.loc[model_1].loc['q_statistic']
    #         self.models_occurence_results_df.loc[model_1].loc['q_statistic'] = number_of + 1
    #         number_of = self.models_occurence_results_df.loc[model_2].loc['q_statistic']
    #         self.models_occurence_results_df.loc[model_2].loc['q_statistic'] = number_of + 1

    #     logging_info(f'After counting votation scheme')
    #     logging_info(self.models_occurence_results_df)

    #     print(f'After counting votation scheme')
    #     print(self.models_occurence_results_df)

    #     # summarizing counting 
    #     for model_key, model_value in models_dic.items():
    #         if model_value[1]:
    #             cor   = self.models_occurence_results_df.loc[model_key].loc['correlation_coefficient']
    #             dfm   = self.models_occurence_results_df.loc[model_key].loc['double_fault_measure'] 
    #             dm    = self.models_occurence_results_df.loc[model_key].loc['disagreement_measure'] 
    #             ia    = self.models_occurence_results_df.loc[model_key].loc['interrater_agreement'] 
    #             qstat = self.models_occurence_results_df.loc[model_key].loc['q_statistic'] 
    #             number_of = cor + dfm + dm + ia + qstat
    #             self.models_occurence_results_df.loc[model_key].loc['total'] = number_of

    #     # setting filename 
    #     path_and_filename = os.path.join(
    #         matrices_folder,
    #         "models-votation-results.xlsx"
    #     )

    #     # writing excel file from dataframe
    #     self.models_occurence_results_df.to_excel(
    #         path_and_filename, sheet_name='models_occurences_results', index=True)
        
    #     # sorting final 
    #     self.models_occurence_results_df_sorted = self.models_occurence_results_df.sort_values(by=['total'], ascending=False)

    #     # setting filename 
    #     path_and_filename = os.path.join(
    #         matrices_folder,
    #         "models-votation-results-sorted.xlsx"
    #     )

    #     # writing excel file from dataframe
    #     self.models_occurence_results_df_sorted.to_excel(
    #         path_and_filename, sheet_name='models_occurences_results', index=True)

    #     # select the top T object detection models for the next step of model fusion
    #     # self.selected_models = self.models_occurence_results_df_sorted["models"]
    #     self.selected_models = []
    #     cont = 0
    #     for model_index, _ in self.models_occurence_results_df_sorted.iterrows():
    #         if cont >= self.top_t:
    #             break
    #         print(f'model_index: {model_index}')
    #         self.selected_models.append(model_index)
    #         cont += 1
