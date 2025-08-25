"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the computing of the models diversity measures.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
    Prof. Dr. Ricardo da Silva Torres - supervisor at AIN of Wageningen University & Research
Date: 31/05/2025
Version: 1.0
"""

import os

# Importing python modules
from common.manage_log import *
from common.utils import * 

from model_diversity.entity.ModelPerformanceMetrics import * 
from model_diversity.entity.ModelsDiversityMeasures import * 
from model_diversity.entity.ModelsPairRelationshipMatrix import * 
from model_diversity.service.ModelsSelectionMethAllDivMeasures import * 
from model_diversity.service.ModelsSelectionMethSingleDivMeasures import * 
from model_diversity.service.ModelsSelectionMethSingleDivMeasuresFilter import * 

from model_diversity.process_predictions import * 


def process_model_diversity_for_models_and_datasets(parameters):

    # processing each model pair for computing the relationship matrix for all datasets 

    # creating new instance of objects 
    models_diversity_measures = ModelsDiversityMeasures()
    models_selection_methods = {}

    # getting all object detection models for processing     
    od_models = []
    for model_key, model_value in parameters["input"]["object_detection_models"].items():
        if model_value[1]:
            od_models.append(model_key)

    logging_info(f'Processing Object Detection Models: {od_models}')
    print(f'Processing Object Detection Models: {od_models}')

    # processing images from dataset name
    dataset_name = parameters["input"]["dataset_name"]

    # setting models diversity measures object 
    models_diversity_measures.dataset_relationship_matrix_list = {
        dataset_name: {}
    }
    print(f'models_diversity_measures: {models_diversity_measures}')
    print(f'models_diversity_measures.dataset_relationship_matrix_list: {models_diversity_measures.dataset_relationship_matrix_list}')

    # computing performance metrics of the models used to get diversity 
    model_performance_metrics_dic = {
        'valid': {},
        'test': {}
    }
    input_dataset_type = 'valid'
    valid_performance_metrics = compute_models_performance_metrics(parameters, dataset_name, od_models, input_dataset_type)
    model_performance_metrics_dic['valid'] = valid_performance_metrics
    models_diversity_measures.dataset_model_performance_metrics_dic = valid_performance_metrics
    
    test_performance_metrics = compute_models_performance_metrics(parameters, dataset_name, od_models, 'test')
    model_performance_metrics_dic['test'] = test_performance_metrics

    logging_info(f'List of models performance metrics')
    logging_info(f'{models_diversity_measures.dataset_model_performance_metrics_dic}')
    print(f'List of models performance metrics')
    print(f'{models_diversity_measures.dataset_model_performance_metrics_dic}')
    for model in od_models:
        dataset_model = dataset_name + '-model_' + model
        logging_info(f'')
        logging_info(f'dataset + model: {dataset_model}')
        logging_info(f'precision: {models_diversity_measures.dataset_model_performance_metrics_dic[dataset_model].get_model_precision()}')
        logging_info(f'recall: {models_diversity_measures.dataset_model_performance_metrics_dic[dataset_model].get_model_recall()}')
        logging_info(f'f1-score: {models_diversity_measures.dataset_model_performance_metrics_dic[dataset_model].get_model_f1_score()}')                


    # processing object detection models by pairs 
    for i in range(0, len(od_models)):
        for j in range (i+1, len(od_models)):

            # setting models pair 
            model_1 = od_models[i]
            model_2 = od_models[j]
            logging_info(f'')
            logging_info(f'Processing models pair: {model_1} and {model_2}')

            # setting full path to the predicitions folder
            predictions_base_folder = os.path.join(
                parameters["processing"]["research_root_folder"], 
                parameters["input"]["predictions_folder_path"],
                dataset_name, 
                "valid",
                "predictions",
            )

            model_1_name = "model_" + model_1 
            predictions_model_1_folder = os.path.join(
                predictions_base_folder,
                parameters[model_1_name]["input"]["predictions_json"]["valid"]
            )
            logging_info(f'Loading predictions of the model {model_1}: {predictions_model_1_folder}')
                    
            model_2_name = "model_" + model_2
            predictions_model_2_folder = os.path.join(
                predictions_base_folder,
                parameters[model_2_name]["input"]["predictions_json"]["valid"]
            )
            logging_info(f'Loading predictions of the model {model_2}: {predictions_model_2_folder}')                       

            # getting preprocessed predictions                    
            model_1_predictions = get_model_predictions(predictions_model_1_folder)
            model_2_predictions = get_model_predictions(predictions_model_2_folder)

            # creating folders for model results 
            models_pair_name_folder = create_model_results_folder(parameters, dataset_name, model_1, model_2)

            # computing relationship matrix for the object detection models pair
            models_pair_relationship_matrix = compute_relationship_matrix(dataset_name, 
                                model_1_predictions,
                                model_2_predictions, 
                                parameters["input"]["diversity_measures_computing"]["iou_threshold"],
                                models_pair_name_folder
                                )

            # saving model pair in json format 
            print(f'rubens - before saving model pair')
            models_pair_relationship_matrix.save_json()

            # printing diversity measures
            logging_info(f'Diversity Measures for the two models: {model_1} and {model_2}')
            logging_info(f'')
            models_pair_relationship_matrix.print()
            logging_info(f'Correlation Coefficient (COR): {models_pair_relationship_matrix.get_correlation_coefficient()}')
            logging_info(f'Double Fault Measure    (DFM): {models_pair_relationship_matrix.get_double_fault_measure()}')
            logging_info(f'Disagreement Measure     (DM): {models_pair_relationship_matrix.get_disagreement_measure()}')
            logging_info(f'Interrater agreement     (IA): {models_pair_relationship_matrix.get_interrater_agreement()}')
            logging_info(f'Q-Statistic           (QSTAT): {models_pair_relationship_matrix.get_q_statistic()}')

            # adding models pair relationship matrix into the model diversity measures 
            models_diversity_measures.add_models_pair_relationship_matrix(dataset_name, models_pair_relationship_matrix)

    # saving summary of model diversity for one dataset 
    matrices_folder = os.path.join(parameters['results']['matrices_folder'])
    models_diversity_measures.save_json(matrices_folder)          

    # compute the diversity measures for each object detection model
    models_diversity_measures.create_diversity_measures()

    # create a summary of diversity measures for all object detection models 
    models_diversity_measures.create_summary_diversity_measures(matrices_folder)

    # create graphs for each diversity model and the models pairs 
    models_diversity_measures.create_graphs_diversity_measures(matrices_folder)

    # process methods for selecting object detection models based on diversity measures 

    # Method 01
    if parameters['input']['object_detection_models_selection_methods']['all_diversity_measures_with_no_tiebreak']['apply']:
        # setting attributes 
        method_short_name = parameters['input']['object_detection_models_selection_methods']['all_diversity_measures_with_no_tiebreak']['short_name']
        method_description = parameters['input']['object_detection_models_selection_methods']['all_diversity_measures_with_no_tiebreak']['description']
        models_selection_meth_all_div = ModelsSelectionMethAllDivMeasures()
        models_selection_meth_all_div.method_short_name = method_short_name
        models_selection_meth_all_div.method_description = method_description
        models_selection_meth_all_div.dataset_name = dataset_name
        models_selection_meth_all_div.models_diversity_measures = models_diversity_measures
        models_selection_meth_all_div.method_results_folder = parameters['results']['m01_all_div_folder'] 
        models_selection_meth_all_div.models_dic = parameters['input']['object_detection_models']
        models_selection_meth_all_div.top_t = parameters["input"]['object_detection_models_selection_methods']['all_diversity_measures_with_no_tiebreak']["top_t"]

        # executing the method for selecting object detection models
        models_selection_meth_all_div.execute()

        # adding selected detectors according to selection method 01
        models_selection_methods[method_short_name] = {}
        models_selection_methods[method_short_name] = {
            "models_selection_method" : models_selection_meth_all_div
        }

        logging_info(f'===============================================')
        logging_info(f'Method 01: {method_short_name} - {method_description}')
        logging_info(f'')
        logging_info(f'top t: {models_selection_meth_all_div.top_t}')
        logging_info(f'Selected models; {models_selection_meth_all_div.selected_models}')
        logging_info(f'===============================================')

        print(f'===============================================')
        print(f'{method_short_name} - {method_description}')
        print(f'')
        print(f'top t: {models_selection_meth_all_div.top_t}')
        print(f'Selected models; {models_selection_meth_all_div.selected_models}')
        print(f'===============================================')
        
    # Method 02
    if parameters['input']['object_detection_models_selection_methods']['single_diversity_measure_with_tiebreak']['apply']:

        # setting attributes 
        method_short_name = parameters['input']['object_detection_models_selection_methods']['single_diversity_measure_with_tiebreak']['short_name']
        method_description = parameters['input']['object_detection_models_selection_methods']['single_diversity_measure_with_tiebreak']['description']
        models_selection_meth_single_div = ModelsSelectionMethSingleDivMeasures()
        models_selection_meth_single_div.method_short_name = method_short_name
        models_selection_meth_single_div.method_description = method_description
        models_selection_meth_single_div.dataset_name = dataset_name
        models_selection_meth_single_div.models_diversity_measures = models_diversity_measures
        models_selection_meth_single_div.method_results_folder = parameters['results']['m02_single_div_folder'] 
        models_selection_meth_single_div.models_dic = parameters['input']['object_detection_models']
        models_selection_meth_single_div.top_t = parameters["input"]['object_detection_models_selection_methods']['single_diversity_measure_with_tiebreak']["top_t"]

        # executing the method for selecting object detection models
        models_selection_meth_single_div.execute()

        # adding selected detectors according to selection method 02
        models_selection_methods[method_short_name] = {}
        models_selection_methods[method_short_name] = {
            "models_selection_method" : models_selection_meth_single_div
        }

        logging_info(f'===============================================')
        logging_info(f'{method_short_name} - {method_description}')
        logging_info(f'')
        logging_info(f'top t: {models_selection_meth_single_div.top_t}')
        logging_info(f'Object detection models selected per diversity measure:')
        logging_info(f'COR: {models_selection_meth_single_div.cor_selected_models}')
        logging_info(f'DFM: {models_selection_meth_single_div.dfm_selected_models}')
        logging_info(f'DM: {models_selection_meth_single_div.dm_selected_models}')
        logging_info(f'IA: {models_selection_meth_single_div.ia_selected_models}')
        logging_info(f'QSTAT: {models_selection_meth_single_div.qstat_selected_models}')
        logging_info(f'===============================================')

        print(f'===============================================')
        print(f'{method_short_name} - {method_description}')
        print(f'')
        print(f'top t: {models_selection_meth_single_div.top_t}')
        print(f'Object detection models selected per diversity measure:')   
        print(f'COR: {models_selection_meth_single_div.cor_selected_models}')
        print(f'DFM: {models_selection_meth_single_div.dfm_selected_models}')
        print(f'DM: {models_selection_meth_single_div.dm_selected_models}')
        print(f'IA: {models_selection_meth_single_div.ia_selected_models}')
        print(f'QSTAT: {models_selection_meth_single_div.qstat_selected_models}')
        print(f'===============================================')

    # Method 03
    if parameters['input']['object_detection_models_selection_methods']['single_diversity_measure_with_tiebreak_and_effectivenes_filter']['apply']:

        # setting attributes 
        method_short_name = parameters['input']['object_detection_models_selection_methods']['single_diversity_measure_with_tiebreak_and_effectivenes_filter']['short_name']
        method_description = parameters['input']['object_detection_models_selection_methods']['single_diversity_measure_with_tiebreak_and_effectivenes_filter']['description']
        models_selection_meth_single_div_filter = ModelsSelectionMethSingleDivMeasuresFilter()
        models_selection_meth_single_div_filter.method_short_name = method_short_name
        models_selection_meth_single_div_filter.method_description = method_description
        models_selection_meth_single_div_filter.dataset_name = dataset_name
        models_selection_meth_single_div_filter.models_diversity_measures = models_diversity_measures
        models_selection_meth_single_div_filter.method_results_folder = parameters['results']['m03_single_div_filter_folder'] 
        models_selection_meth_single_div_filter.models_dic = parameters['input']['object_detection_models']
        models_selection_meth_single_div_filter.top_t = parameters["input"]['object_detection_models_selection_methods']['single_diversity_measure_with_tiebreak_and_effectivenes_filter']["top_t"]
        models_selection_meth_single_div_filter.f1_score_threshold = \
                    parameters["input"]["object_detection_models_selection_methods"]['single_diversity_measure_with_tiebreak_and_effectivenes_filter']['f1_score_threshold_for_valid_dataset']

        # executing the method for selecting object detection models
        models_selection_meth_single_div_filter.execute()

        # adding selected detectors according to selection method 03
        models_selection_methods[method_short_name] = {}
        models_selection_methods[method_short_name] = {
            "models_selection_method" : models_selection_meth_single_div_filter
        }

        logging_info(f'===============================================')
        logging_info(f'{method_short_name} - {method_description}')
        logging_info(f'')
        logging_info(f'top t: {models_selection_meth_single_div_filter.top_t}')
        logging_info(f'Object detection models selected per diversity measure:')
        logging_info(f'COR: {models_selection_meth_single_div_filter.cor_selected_models}')
        logging_info(f'DFM: {models_selection_meth_single_div_filter.dfm_selected_models}')
        logging_info(f'DM: {models_selection_meth_single_div_filter.dm_selected_models}')
        logging_info(f'IA: {models_selection_meth_single_div_filter.ia_selected_models}')
        logging_info(f'QSTAT: {models_selection_meth_single_div_filter.qstat_selected_models}')
        logging_info(f'===============================================')

        print(f'===============================================')
        print(f'{method_short_name} - {method_description}')
        print(f'')
        print(f'top t: {models_selection_meth_single_div_filter.top_t}')
        print(f'Object detection models selected per diversity measure:')   
        print(f'COR: {models_selection_meth_single_div_filter.cor_selected_models}')
        print(f'DFM: {models_selection_meth_single_div_filter.dfm_selected_models}')
        print(f'DM: {models_selection_meth_single_div_filter.dm_selected_models}')
        print(f'IA: {models_selection_meth_single_div_filter.ia_selected_models}')
        print(f'QSTAT: {models_selection_meth_single_div_filter.qstat_selected_models}')
        print(f'===============================================')

    # returning list of selected detectors according selectin methods 
    return models_selection_methods, model_performance_metrics_dic


# computing performance metrics of the models used to get diversity 
def compute_models_performance_metrics_old(parameters, dataset_name, models, input_dataset_type):

    # dictionary of model performance metrics 
    dataset_model_performance_metrics_dic = {}

    # compute prformance metrics for each model
    for model in models:

        # setting full path to the predicitions folder
        predictions_base_folder = os.path.join(
            parameters["processing"]["research_root_folder"], 
            parameters["input"]["predictions_folder_path"],
            dataset_name, 
            input_dataset_type,
            "predictions",
        )

        model_name = "model_" + model
        predictions_model_filename = os.path.join(
            predictions_base_folder,
            parameters[model_name]["input"]["predictions_json"][input_dataset_type]
        )
        logging_info(f'Loading predictions of the model {model}: {predictions_model_filename}')
                
        # getting preprocessed predictions                    
        model_predictions = get_model_predictions(predictions_model_filename)
        logging_info(f'Predictions: {model_predictions}')

        # creating model performance metric object 
        model_performance_metrics = ModelPerformanceMetrics()
        model_performance_metrics.dataset_name = dataset_name
        model_performance_metrics.model_name = model
        model_performance_metrics.classes = parameters['neural_network_model']['classes']
        model_performance_metrics.number_of_classes = parameters['neural_network_model']['number_of_classes']
        model_performance_metrics.iou_threshold = parameters[model_name]["neural_network_model"]["iou_threshold"]
        model_performance_metrics.predictions = model_predictions

        # computing performance metrics to model 
        model_performance_metrics.compute_metrics()

        # adding into the list of models performance list
        dataset_model_performance_metrics_dic[dataset_name + '-' + model_name] = model_performance_metrics

    # returning list of model performance metrics 
    return dataset_model_performance_metrics_dic

def create_model_results_folder(parameters, dataset_name, model_1_name, model_2_name):

    # creating results folders 
    matrices_folder = parameters['results']['matrices_folder']
    dataset_name_folder = os.path.join(
        matrices_folder,
        dataset_name,
    )
    Utils.create_directory(dataset_name_folder)

    models_pair_name_folder = os.path.join(
        dataset_name_folder,
        (model_1_name + "-x-" + model_2_name)
    )
    Utils.create_directory(models_pair_name_folder)

    # returning the models pair name folder 
    return models_pair_name_folder


def compute_relationship_matrix(dataset_name, model_1_predictions, model_2_predictions, 
                                iou_threshold_for_divertsity_measure, models_pair_name_folder):

    # creating new instance of objects 
    models_pair_relationship_matrix = ModelsPairRelationshipMatrix()

    # setting fields
    models_pair_relationship_matrix.dataset_name = dataset_name
    models_pair_relationship_matrix.model_1_name = model_1_predictions['model_name']
    models_pair_relationship_matrix.model_1_predictions = model_1_predictions
    models_pair_relationship_matrix.model_2_name = model_2_predictions['model_name']
    models_pair_relationship_matrix.model_2_predictions = model_2_predictions
    models_pair_relationship_matrix.iou_threshold_for_divertsity_measure = iou_threshold_for_divertsity_measure
    models_pair_relationship_matrix.a = 0
    models_pair_relationship_matrix.b = 0
    models_pair_relationship_matrix.c = 0
    models_pair_relationship_matrix.d = 0
    models_pair_relationship_matrix.image_relationship_matrix_list = []
    models_pair_relationship_matrix.models_pair_name_folder = models_pair_name_folder

    # computing relationship matrix for the models pair
    models_pair_relationship_matrix.compute()

    # returning relationship matrix for the models pair 
    return models_pair_relationship_matrix
    

# computing performance metrics of the models used to get diversity 
def compute_models_performance_metrics(parameters, dataset_name, models, input_dataset_type):

    # dictionary of model performance metrics 
    dataset_model_performance_metrics_dic = {}

    # summary of model performance metrics
    model_performance_metric_summary = []

    # compute prformance metrics for each model
    for model in models:

        # setting full path to the predicitions folder
        predictions_base_folder = os.path.join(
            parameters["processing"]["research_root_folder"], 
            parameters["input"]["predictions_folder_path"],
            dataset_name, 
            input_dataset_type,
            "predictions",
        )

        model_name = "model_" + model
        predictions_model_filename = os.path.join(
            predictions_base_folder,
            parameters[model_name]["input"]["predictions_json"][input_dataset_type]
        )
        logging_info(f'Loading predictions of the model {model}: {predictions_model_filename}')
                
        # getting preprocessed predictions                    
        model_predictions = get_model_predictions(predictions_model_filename)
        logging_info(f'Predictions: {model_predictions}')

        # creating model performance metric object 
        model_performance_metrics = ModelPerformanceMetrics()
        model_performance_metrics.dataset_name = dataset_name
        model_performance_metrics.model_name = model
        model_performance_metrics.classes = parameters['neural_network_model']['classes']
        model_performance_metrics.number_of_classes = parameters['neural_network_model']['number_of_classes']
        model_performance_metrics.iou_threshold = parameters[model_name]["neural_network_model"]["iou_threshold"]
        model_performance_metrics.predictions = model_predictions
        model_performance_metrics.input_dataset_type = input_dataset_type

        # computing performance metrics to model 
        model_performance_metrics.compute_metrics()

        # adding into the list of models performance list
        dataset_model_performance_metrics_dic[dataset_name + '-' + model_name] = model_performance_metrics

        # saving performance metrics for the model 
        performance_metrics_folder = os.path.join(parameters['results']['performance_metrics_folder'], input_dataset_type)
        Utils.create_directory(performance_metrics_folder)
        title = 'Confusion Matrix' + \
                ' - Models Selection Method: ' + parameters['input']['object_detection_models_selection_methods']['hierarchical_clustering_based_on_diversity_measure']['short_name'] + \
                ' - Dataset: ' + dataset_name + \
                ' - Type: ' + input_dataset_type + \
                ' - Model name: ' + model_name
        title += LINE_FEED + \
                'Confidence threshold: ' + \
                '   IoU threshold: ' + f"{model_performance_metrics.iou_threshold:.1f}"
        filename = model_name + '-' + \
                   parameters['input']['object_detection_models_selection_methods']['hierarchical_clustering_based_on_diversity_measure']['short_name'] + '-' + \
                   parameters['processing']['running_id_text']                   
        classes = parameters['neural_network_model']['classes']
        number_of_classes = parameters['neural_network_model']['number_of_classes']
        cm_classes = classes[0:(number_of_classes+1)]
        model_performance_metrics.save_confusion_matrix(performance_metrics_folder, filename, title, cm_classes)
        
        # saving performance metrics 
        cm_classes_for_metrics = classes[1:(number_of_classes+1)]
        model_performance_metrics.save_performance_metrics(
            performance_metrics_folder, filename, title, cm_classes_for_metrics, 
            0 ,  model_performance_metrics.iou_threshold, 0, model_name)

        # saving bounding boxes used in the fusion 
        model_performance_metrics.save_inferenced_images(performance_metrics_folder, filename)

        # adding model performance metrics into the summary
        item = [dataset_name, input_dataset_type, model_name, model_performance_metrics.get_model_precision(),
                model_performance_metrics.get_model_recall(), model_performance_metrics.get_model_f1_score()]
        model_performance_metric_summary.append(item)

    # saving model performance metrics summary in excel sheet
    running_id_text = parameters['processing']['running_id_text']
    save_summary_metrics(model_performance_metric_summary, performance_metrics_folder, input_dataset_type, running_id_text)

    # returning list of model performance metrics 
    return dataset_model_performance_metrics_dic


def save_summary_metrics(model_performance_metric_summary, folder, input_dataset_type, running_id_text):
    """
    Save the summary of model performance metrics to an Excel file.
    """

    # setting path and filename 
    model_performance_metrics_summary_filename = os.path.join(
        folder, 'summary-' + input_dataset_type + '-' + running_id_text +'-model-performance-metrics.xlsx')
    print(f'rubens model performance metrics summary to {model_performance_metrics_summary_filename}')

    # preparing columns name to list        
    column_names = [
        "dataset_name",
        "input_dataset_type",
        "model_name",
        "precision",
        "recall",
        "f1_score"        
    ]

    # creating dataframe from list 
    df = pd.DataFrame(model_performance_metric_summary, columns=column_names)

    # writing excel file from dataframe
    df.to_excel(model_performance_metrics_summary_filename, sheet_name='model_performance_metrics', index=False)
