"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the methods of bounding boxes fusion based on detectors diversity.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
    Prof. Dr. Ricardo da Silva Torres - supervisor at AIN of Wageningen University & Research
Date: 20/06/2025
Version: 1.0
"""

import os

# Importing python modules
from common.manage_log import *
from model_diversity.process_predictions import * 
from model_diversity.entity.VotingBoxesFusion import * 
# from model_diversity.entity.WeightedBoxesFusion import * 

def perform_bounding_boxes_fusion(parameters, models_selection_methods, input_dataset_type, dataset_model_performance_metrics_dic):

    # evaluating number of selected models as it need to be two or greater 
    # for models_selection_method_key, models_selection_method_value in models_selection_methods.items():
    #     if len(models_selection_method_value['models_selection_method'].selected_models) < 2:
    #         logging_warning(f'Not enough selected models for {models_selection_method_key}. Skipping fusion.')
    #         print(f'')
    #         continue

    # performing bounding boxes fusion for all selection methods and all bounding boxes fusion methods

    # dictionary of bounding boxes fusion methods results  
    bboxes_fusion_method_results_dic = {}   

    # getting all bounding boxes fusion methods for processing
    bboxes_fusion_methods = []
    for fusion_method_key, fusion_method_value in parameters["input"]["bounding_boxes_fusion_methods"].items():
        if fusion_method_value["apply"]:
            bboxes_fusion_methods.append(fusion_method_key)

    # setting the original fusion method folder
    original_fusion_method_folder = parameters['results']['bboxes_fusion_folder']

    # performing bounding boxes fusion for each method 
    for bboxes_fusion_method in bboxes_fusion_methods:

        # logging_info(f'bboxes_fusion_method: {bboxes_fusion_method}')        
        # creating folder for fusion method results
        fusion_method_folder = os.path.join(original_fusion_method_folder, bboxes_fusion_method)
        parameters['results']['bboxes_fusion_folder'] = fusion_method_folder
        Utils.create_directory(fusion_method_folder)

        # performing bounding boxes fusion using the voting scheme
        # if bboxes_fusion_method == "voting_scheme":

        # performing fusion for each selection method
        for models_selection_method_key, models_selection_method_value in models_selection_methods.items():

            # logging_info(f'')
            # print(f"models_selection_method_value['models_selection_method']: {models_selection_method_value['models_selection_method']}")
            # print(f"models_selection_method_value['models_selection_method'].selected_models: {models_selection_method_value['models_selection_method'].selected_models}")
            # logging_info(f"models_selection_method_value['models_selection_method'].selected_models: {models_selection_method_value['models_selection_method'].selected_models}")
            # logging_info(f"len(models_selection_method_value['models_selection_method'].selected_models): {len(models_selection_method_value['models_selection_method'].selected_models)}")
            # print(f'')
            # print(f"models_selection_method_value['models_selection_method'].selected_models: {models_selection_method_value['models_selection_method'].selected_models}")
            # print(f"len(models_selection_method_value['models_selection_method'].selected_models): {len(models_selection_method_value['models_selection_method'].selected_models)}")
            # if len(models_selection_method_value['models_selection_method'].selected_models) < 2:
            #     logging_info(f'')
            #     logging_info(f'The model selection method {models_selection_method_key} has less than 2 selected models. Skipping fusion.')
            #     logging_info(f'The only selected model is: {models_selection_method_value["models_selection_method"].selected_models[0]}')
            #     print(f'')
            #     print(f'The model selection method {models_selection_method_key} has less than 2 selected models. Skipping fusion.')
            #     print(f'The only selected model is: {models_selection_method_value["models_selection_method"].selected_models[0]}')
            #     continue

            # logging_info(f'models_selection_method_key: {models_selection_method_key}')
            # logging_info(f'models_selection_method_value: {models_selection_method_value}')
            # logging_info(f'type(models_selection_method_value): {type(models_selection_method_value)}')
            # logging_info(f'models_selection_method_value.keys(): {models_selection_method_value.keys()}')
            # logging_info(f'models_selection_method_key: {models_selection_method_key}')
            # logging_info(f'models_selection_method_value: {models_selection_method_value["models_selection_method"]}')
            # logging_info(f'models_selection_method_value[models_selection_method].dataset_name: {models_selection_method_value["models_selection_method"].dataset_name}')
            # logging_info(f'')

            # setting parameters 
            dataset_name = models_selection_method_value['models_selection_method'].dataset_name

            # performing bounding boxes fusion using the voting scheme
            if bboxes_fusion_method == "voting_scheme":
                # performing bboxes fusion for each model selection method
                process_fusion_voting_scheme_per_selection_model_methods(
                        parameters, models_selection_method_key, models_selection_method_value, input_dataset_type, 
                        dataset_name, bboxes_fusion_method, bboxes_fusion_method_results_dic)

            if bboxes_fusion_method == "weighted_boxes_fusion":
                # performing bboxes fusion for each model selection method
                process_fusion_weighted_boxes_fusion_per_selection_model_methods(
                        parameters, models_selection_method_key, models_selection_method_value, input_dataset_type, 
                        dataset_name, bboxes_fusion_method, bboxes_fusion_method_results_dic)

            # if models_selection_method_key == "m01-all-div":
            #     models_selection_method_name = models_selection_method_key + "-all"
            #     selected_models = models_selection_method_value['models_selection_method'].selected_models
            #     if bboxes_fusion_method == "voting_scheme":
            #         all_div_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(
            #             parameters, input_dataset_type, models_selection_method_name, dataset_name, selected_models)
            #     elif bboxes_fusion_method == "weighted_boxes_fusion":
            #         pass
            #     all_div_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
            #     all_div_boxes_fusion.f1_score_threshold = ''
            #     all_div_boxes_fusion.minimum_number_of_models = ''
            #     # adding results of bounding boxes fusion method
            #     bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'all'
            #     bboxes_fusion_method_results_dic[bboxes_fusion_name] = all_div_boxes_fusion
            
            # if models_selection_method_key in ["m02-single-div", "m03-single-div-filter"]:
            #     models_selection_method_name = models_selection_method_key + "-cor"
            #     cor_selected_models = models_selection_method_value['models_selection_method'].cor_selected_models
            #     if bboxes_fusion_method == "voting_scheme":
            #         cor_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(
            #             parameters, input_dataset_type, models_selection_method_name, dataset_name, cor_selected_models)
            #     elif bboxes_fusion_method == "weighted_boxes_fusion":
            #         pass
            #     cor_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
            #     cor_boxes_fusion.f1_score_threshold = ''
            #     if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            #         cor_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
            #     elif bboxes_fusion_method == "weighted_boxes_fusion":
            #         pass
            #     cor_boxes_fusion.minimum_number_of_models = ''
            #     # adding results of bounding boxes fusion method
            #     bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'cor'
            #     bboxes_fusion_method_results_dic[bboxes_fusion_name] = cor_boxes_fusion                

            #     models_selection_method_name = models_selection_method_key + "-dfm"
            #     dfm_selected_models = models_selection_method_value['models_selection_method'].dfm_selected_models
            #     if bboxes_fusion_method == "voting_scheme":
            #         dfm_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(
            #             parameters, input_dataset_type, models_selection_method_name, dataset_name, dfm_selected_models)
            #     elif bboxes_fusion_method == "weighted_boxes_fusion":
            #         pass
            #     dfm_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
            #     dfm_boxes_fusion.f1_score_threshold = ''
            #     if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            #         dfm_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
            #     dfm_boxes_fusion.minimum_number_of_models = ''
            #     # adding results of bounding boxes fusion method
            #     bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'dfm'
            #     bboxes_fusion_method_results_dic[bboxes_fusion_name] = dfm_boxes_fusion                

            #     models_selection_method_name = models_selection_method_key + "-dm"
            #     dm_selected_models = models_selection_method_value['models_selection_method'].dm_selected_models
            #     if bboxes_fusion_method == "voting_scheme":
            #         dm_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(
            #             parameters, input_dataset_type, models_selection_method_name, dataset_name, dm_selected_models)
            #     elif bboxes_fusion_method == "weighted_boxes_fusion":
            #         pass
            #     dm_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
            #     dm_boxes_fusion.f1_score_threshold = ''
            #     if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            #         dm_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
            #     dm_boxes_fusion.minimum_number_of_models = ''
            #     # adding results of bounding boxes fusion method
            #     bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'dm'
            #     bboxes_fusion_method_results_dic[bboxes_fusion_name] = dm_boxes_fusion                

            #     models_selection_method_name = models_selection_method_key + "-ia"
            #     ia_selected_models = models_selection_method_value['models_selection_method'].ia_selected_models
            #     if bboxes_fusion_method == "voting_scheme":
            #         ia_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(
            #             parameters, input_dataset_type, models_selection_method_name, dataset_name, ia_selected_models)
            #     elif bboxes_fusion_method == "weighted_boxes_fusion":
            #         pass
            #     ia_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
            #     ia_boxes_fusion.f1_score_threshold = ''
            #     if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            #         ia_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
            #     ia_boxes_fusion.minimum_number_of_models = ''
            #     # adding results of bounding boxes fusion method
            #     bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'ia'
            #     bboxes_fusion_method_results_dic[bboxes_fusion_name] = ia_boxes_fusion                

            #     models_selection_method_name = models_selection_method_key + "-qstat"
            #     qstat_selected_models = models_selection_method_value['models_selection_method'].qstat_selected_models
            #     if bboxes_fusion_method == "voting_scheme":
            #         qstat_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(
            #             parameters, input_dataset_type, models_selection_method_name, dataset_name, qstat_selected_models)
            #     elif bboxes_fusion_method == "weighted_boxes_fusion":
            #         pass
            #     qstat_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
            #     qstat_boxes_fusion.f1_score_threshold = ''
            #     if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            #         qstat_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
            #     qstat_boxes_fusion.minimum_number_of_models = ''
            #     # adding results of bounding boxes fusion method
            #     bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'qstat'
            #     bboxes_fusion_method_results_dic[bboxes_fusion_name] = qstat_boxes_fusion                

            # # performing bboxes fusion for each model selection method
            # if models_selection_method_key == "m04-clustering" or models_selection_method_key == "m05-clustering-filtered":
            #     # performing bounding boxes fusion by clustering and diversity measures
            #     # bboxes_fusion_method_results_dic = perform_model_fusion_by_clustering_and_diversity(
            #     #                                         parameters, models_selection_methods, input_dataset_type, dataset_name,
            #     #                                         bboxes_fusion_method, models_selection_method_key)   
            #     models_selection_method_name = models_selection_method_key 
            #     selected_models_all_detectors = models_selection_method_value['models_selection_method'].selected_models
            #     print(f'selected_models_all_detectors: {selected_models_all_detectors}')
            #     if bboxes_fusion_method == "voting_scheme":
            #         all_div_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(
            #             parameters, input_dataset_type, models_selection_method_name, dataset_name, selected_models_all_detectors)
            #     elif bboxes_fusion_method == "weighted_boxes_fusion":
            #         pass
            #     all_div_boxes_fusion.top_t = ''
            #     all_div_boxes_fusion.f1_score_threshold = ''
            #     if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold_for_valid_dataset'):
            #         all_div_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold_for_valid_dataset
            #     all_div_boxes_fusion.minimum_number_of_models = models_selection_method_value['models_selection_method'].minimum_models_in_cluster
            #     # adding results of bounding boxes fusion method
            #     bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key
            #     bboxes_fusion_method_results_dic[bboxes_fusion_name] = all_div_boxes_fusion                

            # if models_selection_method_key == "m03-single-div-filter":
            #     print(f'models_selection_method_key: {models_selection_method_key}')
                

        # # performing bounding boxes fusion using the weighted boxes fusion
        # if bboxes_fusion_method == "weighted_boxes_fusion":
        #     print(f'bboxes_fusion_method: {bboxes_fusion_method}')
        #     # execute_weighted_boxes_fusion()
        #     # summarizing bounding boxes fusion results
        #     bboxes_fusion_folder = parameters["results"]["bboxes_fusion_folder"]
        #     summary_bounding_boxes_fusion(bboxes_fusion_method_results_dic, bboxes_fusion_folder, 
        #                                   input_dataset_type, dataset_model_performance_metrics_dic, 
        #                                   parameters["processing"]["running_id_text"], bboxes_fusion_method)

    # print(f'Processing weighted boxes fusion for {models_selection_method_key} with {len(models_selection_method_value["models_selection_method"].selected_models)} selected models.')
    # print(f'')
    print(f'bboxes_fusion_method: {bboxes_fusion_method}')
    print(f'bboxes_fusion_method_results_dic.keys: {bboxes_fusion_method_results_dic.keys()}')
    print(f'bboxes_fusion_method_results_dic.values: {bboxes_fusion_method_results_dic.values()}')

   
    # summarizing bounding boxes fusion results
    print(f'summary_bounding_boxes_fusion 01')
    summary_bounding_boxes_fusion(bboxes_fusion_method_results_dic, original_fusion_method_folder, 
                                    input_dataset_type, dataset_model_performance_metrics_dic, 
                                    parameters["processing"]["running_id_text"])
    print(f'summary_bounding_boxes_fusion 02')

    # # summarizing bounding boxes fusion results
    # bboxes_fusion_folder = parameters["results"]["bboxes_fusion_folder"]
    # summary_bounding_boxes_fusion(bboxes_fusion_method_results_dic, bboxes_fusion_folder, 
    #                               input_dataset_type, dataset_model_performance_metrics_dic, parameters["processing"]["running_id_text"])

    # returning results of detector diversity with bounding boxes fusion methods 
    return bboxes_fusion_method_results_dic


def process_fusion_voting_scheme_per_selection_model_methods(
        parameters, models_selection_method_key, models_selection_method_value, input_dataset_type, 
        dataset_name, bboxes_fusion_method, bboxes_fusion_method_results_dic):

    # performing bboxes fusion for each model selection method
    if models_selection_method_key == "m01-all-div":
        models_selection_method_name = models_selection_method_key + "-all"
        selected_models = models_selection_method_value['models_selection_method'].selected_models
        voting_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(parameters, input_dataset_type, 
                                        models_selection_method_name, dataset_name, selected_models)
        voting_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        voting_boxes_fusion.f1_score_threshold = ''
        voting_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'all'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = voting_boxes_fusion
    
    if models_selection_method_key in ["m02-single-div", "m03-single-div-filter"]:
        models_selection_method_name = models_selection_method_key + "-cor"
        cor_selected_models = models_selection_method_value['models_selection_method'].cor_selected_models
        cor_voting_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, cor_selected_models)
        cor_voting_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        cor_voting_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            cor_voting_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        cor_voting_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'cor'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = cor_voting_boxes_fusion                

        models_selection_method_name = models_selection_method_key + "-dfm"
        dfm_selected_models = models_selection_method_value['models_selection_method'].dfm_selected_models
        dfm_voting_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, dfm_selected_models)
        dfm_voting_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        dfm_voting_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            dfm_voting_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        dfm_voting_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'dfm'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = dfm_voting_boxes_fusion                

        models_selection_method_name = models_selection_method_key + "-dm"
        dm_selected_models = models_selection_method_value['models_selection_method'].dm_selected_models
        dm_voting_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, dm_selected_models)
        dm_voting_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        dm_voting_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            dm_voting_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        dm_voting_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'dm'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = dm_voting_boxes_fusion                

        models_selection_method_name = models_selection_method_key + "-ia"
        ia_selected_models = models_selection_method_value['models_selection_method'].ia_selected_models
        ia_voting_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, ia_selected_models)
        ia_voting_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        ia_voting_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            ia_voting_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        ia_voting_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'ia'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = ia_voting_boxes_fusion                

        models_selection_method_name = models_selection_method_key + "-qstat"
        qstat_selected_models = models_selection_method_value['models_selection_method'].qstat_selected_models
        qstat_voting_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, qstat_selected_models)
        qstat_voting_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        qstat_voting_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            qstat_voting_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        qstat_voting_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'qstat'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = qstat_voting_boxes_fusion                

    # performing bboxes fusion for each model selection method
    if models_selection_method_key == "m04-clustering" or models_selection_method_key == "m05-clustering-filtered":
        # performing bounding boxes fusion by clustering and diversity measures
        # bboxes_fusion_method_results_dic = perform_model_fusion_by_clustering_and_diversity(
        #                                         parameters, models_selection_methods, input_dataset_type, dataset_name,
        #                                         bboxes_fusion_method, models_selection_method_key)   
        models_selection_method_name = models_selection_method_key 
        selected_models_all_detectors = models_selection_method_value['models_selection_method'].selected_models
        print(f'selected_models_all_detectors: {selected_models_all_detectors}')
        voting_boxes_fusion = perform_bounding_boxes_fusion_voting_scheme(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, selected_models_all_detectors)
        voting_boxes_fusion.top_t = ''
        voting_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold_for_valid_dataset'):
            voting_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold_for_valid_dataset
        voting_boxes_fusion.minimum_number_of_models = models_selection_method_value['models_selection_method'].minimum_models_in_cluster
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = voting_boxes_fusion                


def perform_bounding_boxes_fusion_voting_scheme(parameters, input_dataset_type, 
                                                models_selection_method_name, 
                                                dataset_name, selected_models):

    # perform bounding boxes fusion for each diversity measure result
    voting_boxes_fusion = VotingBoxesFusion()
    voting_boxes_fusion.models_selection_method_name = models_selection_method_name
    voting_boxes_fusion.dataset_name = dataset_name
    voting_boxes_fusion.selected_models = selected_models
    voting_boxes_fusion.all_predictions_list = get_all_predictions(parameters, input_dataset_type, dataset_name, selected_models)
    voting_boxes_fusion.iou_threshold_for_grouping = parameters["input"]["bounding_boxes_fusion_methods"]["voting_scheme"]["iou_threshold_for_grouping"]
    voting_boxes_fusion.iou_threshold_for_inference = parameters["input"]["bounding_boxes_fusion_methods"]["voting_scheme"]["iou_threshold_for_inference"]
    voting_boxes_fusion.non_maximum_suppression = parameters["input"]["bounding_boxes_fusion_methods"]["voting_scheme"]["non_maximum_suppression"]
    voting_boxes_fusion.number_of_classes = parameters["neural_network_model"]["number_of_classes"]
    voting_boxes_fusion.classes = parameters["neural_network_model"]["classes"]
    voting_boxes_fusion.input_dataset_type = input_dataset_type
    voting_boxes_fusion.execute()
    voting_boxes_fusion.save_results(parameters["results"]["bboxes_fusion_folder"], parameters["processing"]["running_id_text"])

    # returning voting boxes fusion results 
    return voting_boxes_fusion


def process_fusion_weighted_boxes_fusion_per_selection_model_methods(
        parameters, models_selection_method_key, models_selection_method_value, input_dataset_type, 
        dataset_name, bboxes_fusion_method, bboxes_fusion_method_results_dic):

    # performing bboxes fusion for each model selection method
    if models_selection_method_key == "m01-all-div":
        models_selection_method_name = models_selection_method_key + "-all"
        selected_models = models_selection_method_value['models_selection_method'].selected_models
        wbf_boxes_fusion = perform_weighted_boxes_fusion(parameters, input_dataset_type, 
                                        models_selection_method_name, dataset_name, selected_models)
        wbf_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        wbf_boxes_fusion.f1_score_threshold = ''
        wbf_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'all'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = wbf_boxes_fusion
    
    if models_selection_method_key in ["m02-single-div", "m03-single-div-filter"]:
        models_selection_method_name = models_selection_method_key + "-cor"
        cor_selected_models = models_selection_method_value['models_selection_method'].cor_selected_models
        cor_wbf_boxes_fusion = perform_weighted_boxes_fusion(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, cor_selected_models)
        cor_wbf_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        cor_wbf_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            cor_wbf_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        cor_wbf_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'cor'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = cor_wbf_boxes_fusion                

        models_selection_method_name = models_selection_method_key + "-dfm"
        dfm_selected_models = models_selection_method_value['models_selection_method'].dfm_selected_models
        dfm_wbf_boxes_fusion = perform_weighted_boxes_fusion(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, dfm_selected_models)
        dfm_wbf_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        dfm_wbf_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            dfm_wbf_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        dfm_wbf_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'dfm'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = dfm_wbf_boxes_fusion                

        models_selection_method_name = models_selection_method_key + "-dm"
        dm_selected_models = models_selection_method_value['models_selection_method'].dm_selected_models
        dm_wbf_boxes_fusion = perform_weighted_boxes_fusion(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, dm_selected_models)
        dm_wbf_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        dm_wbf_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            dm_wbf_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        dm_wbf_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'dm'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = dm_wbf_boxes_fusion                

        models_selection_method_name = models_selection_method_key + "-ia"
        ia_selected_models = models_selection_method_value['models_selection_method'].ia_selected_models
        ia_wbf_boxes_fusion = perform_weighted_boxes_fusion(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, ia_selected_models)
        ia_wbf_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        ia_wbf_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            ia_wbf_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        ia_wbf_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'ia'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = ia_wbf_boxes_fusion                

        models_selection_method_name = models_selection_method_key + "-qstat"
        qstat_selected_models = models_selection_method_value['models_selection_method'].qstat_selected_models
        qstat_wbf_boxes_fusion = perform_weighted_boxes_fusion(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, qstat_selected_models)
        qstat_wbf_boxes_fusion.top_t = models_selection_method_value['models_selection_method'].top_t
        qstat_wbf_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold'):
            qstat_wbf_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold
        qstat_wbf_boxes_fusion.minimum_number_of_models = ''
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key + '-' + 'qstat'
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = qstat_wbf_boxes_fusion                

    # performing bboxes fusion for each model selection method
    if models_selection_method_key == "m04-clustering" or models_selection_method_key == "m05-clustering-filtered":
        models_selection_method_name = models_selection_method_key 
        selected_models_all_detectors = models_selection_method_value['models_selection_method'].selected_models
        print(f'selected_models_all_detectors: {selected_models_all_detectors}')
        wbf_boxes_fusion = perform_weighted_boxes_fusion(parameters, input_dataset_type,
                                        models_selection_method_name, dataset_name, selected_models_all_detectors)
        wbf_boxes_fusion.top_t = ''
        wbf_boxes_fusion.f1_score_threshold = ''
        if hasattr(models_selection_method_value['models_selection_method'], 'f1_score_threshold_for_valid_dataset'):
            wbf_boxes_fusion.f1_score_threshold = models_selection_method_value['models_selection_method'].f1_score_threshold_for_valid_dataset
        wbf_boxes_fusion.minimum_number_of_models = models_selection_method_value['models_selection_method'].minimum_models_in_cluster
        # adding results of bounding boxes fusion method
        bboxes_fusion_name = bboxes_fusion_method + '-' + models_selection_method_key
        bboxes_fusion_method_results_dic[bboxes_fusion_name] = wbf_boxes_fusion                


def perform_weighted_boxes_fusion(parameters, input_dataset_type, models_selection_method_name, dataset_name, selected_models):

    # perform bounding boxes fusion for each diversity measure result
    weighted_boxes_fusion = WeightedBoxesFusion()
    weighted_boxes_fusion.models_selection_method_name = models_selection_method_name
    weighted_boxes_fusion.dataset_name = dataset_name
    weighted_boxes_fusion.selected_models = selected_models
    weighted_boxes_fusion.all_predictions_list = get_all_predictions(parameters, input_dataset_type, dataset_name, selected_models)
    weighted_boxes_fusion.iou_threshold_for_matched_boxes = parameters["input"]["bounding_boxes_fusion_methods"]["weighted_boxes_fusion"]["iou_threshold_for_matched_boxes"]
    weighted_boxes_fusion.iou_threshold_for_inference = parameters["input"]["bounding_boxes_fusion_methods"]["weighted_boxes_fusion"]["iou_threshold_for_inference"]
    weighted_boxes_fusion.number_of_classes = parameters["neural_network_model"]["number_of_classes"]
    weighted_boxes_fusion.classes = parameters["neural_network_model"]["classes"]
    weighted_boxes_fusion.input_dataset_type = input_dataset_type
    weighted_boxes_fusion.minimum_fused_bounding_boxes = parameters["input"]["bounding_boxes_fusion_methods"]["weighted_boxes_fusion"]["minimum_fused_bounding_boxes"]
    weighted_boxes_fusion.execute()
    weighted_boxes_fusion.save_results(parameters["results"]["bboxes_fusion_folder"], 
                                       parameters["processing"]["running_id_text"], 
                                       parameters["input"]["bounding_boxes_fusion_methods"]["weighted_boxes_fusion"]["short_name"]
                                       )

    # returning weighted boxes fusion results
    return weighted_boxes_fusion

def get_all_predictions(parameters, input_dataset_type, dataset_name, selected_models):

    # logging_info(f'get_all_predictions')
    # logging_info(f'input_dataset_type: {input_dataset_type}')
    # logging_info(f'dataset_name: {dataset_name}')
    # logging_info(f'selected_models: {selected_models}')

    # initializing all predictions list 
    all_predictions_list = []

    # setting full path to the predicitions folder
    predictions_base_folder = os.path.join(
        parameters["processing"]["research_root_folder"], 
        parameters["input"]["predictions_folder_path"],
        dataset_name, input_dataset_type, "predictions",
    )

    # getting predictions for each model 
    for selected_model in selected_models:

        model_name = "model_" + selected_model
        predictions_model_folder = os.path.join(
            predictions_base_folder,
            parameters[model_name]["input"]["predictions_json"][input_dataset_type]
        )
        # logging_info(f'Loading predictions of the model {selected_model}: {predictions_model_folder}')
            
        # getting preprocessed predictions                    
        model_predictions = get_model_predictions(predictions_model_folder)

        # logging_info(f'model_predictions: {model_predictions}')

        # adding to all predictions list 
        all_predictions_list.append(model_predictions)

    # returning all predictions 
    return all_predictions_list


def summary_bounding_boxes_fusion(bboxes_fusion_method_results_dic, bboxes_fusion_folder, input_dataset_type, 
                                  dataset_model_performance_metrics_dic, running_id_text):

    # summary of bounding boxes fusion results
    single_models_bboxes_fusion_summary = []
    all_methods_bboxes_fusion_summary = []
    wbf_bboxes_fusion_summary = []
    voting_bboxes_fusion_summary = []

    running_id = running_id_text.replace('running-', '')

    # adding the performance metrics of the single models in the test dataset
    for model_name, model_performance_metrics in dataset_model_performance_metrics_dic['test'].items():
        # setting common columns for all methods
        dataset_name = model_performance_metrics.dataset_name
        method_input_dataset_type = input_dataset_type
        model_selection_method = 'single-model'
        top_t = ''
        f1_score_threshold = ''
        minimum_models_in_cluster = ''
        selected_models = model_name
        bboxes_fusion_method = 'no fusion'
        fusion_method = ''

        # setting model performance metrics
        fusion_complement = ''
        precision = model_performance_metrics.get_model_precision()
        recall = model_performance_metrics.get_model_recall()
        f1_score = model_performance_metrics.get_model_f1_score()

        # adding model performance metrics into the summary
        item = [dataset_name, method_input_dataset_type, model_selection_method, top_t, f1_score_threshold,
                minimum_models_in_cluster, selected_models, fusion_method, fusion_complement, 
                precision, recall, f1_score, running_id]
        single_models_bboxes_fusion_summary.append(item)

    for bboxes_fusion_method, bboxes_fusion_results in bboxes_fusion_method_results_dic.items():

        if isinstance(bboxes_fusion_results, VotingBoxesFusion):
            voting_bboxes_fusion_summary.extend(set_voting_results(running_id, bboxes_fusion_results, bboxes_fusion_method))

            # # setting common columns for all methods
            # dataset_name = bboxes_fusion_results.dataset_name
            # method_input_dataset_type = bboxes_fusion_results.input_dataset_type
            # model_selection_method = bboxes_fusion_results.models_selection_method_name
            # top_t = bboxes_fusion_results.top_t
            # f1_score_threshold = bboxes_fusion_results.f1_score_threshold
            # minimum_models_in_cluster = bboxes_fusion_results.minimum_number_of_models
            # selected_models = bboxes_fusion_results.selected_models
            # fusion_method = bboxes_fusion_method

            # # setting affirmative fusion results
            # fusion_complement = 'affirmative'
            # precision = bboxes_fusion_results.affirmative_performance_metrics.get_model_precision()
            # recall = bboxes_fusion_results.affirmative_performance_metrics.get_model_recall()
            # f1_score = bboxes_fusion_results.affirmative_performance_metrics.get_model_f1_score()

            # # adding model performance metrics into the summary
            # item = [dataset_name, method_input_dataset_type, model_selection_method, top_t, f1_score_threshold,
            #         minimum_models_in_cluster, selected_models, fusion_method, fusion_complement, 
            #         precision, recall, f1_score, running_id]
            # bboxes_fusion_summary.append(item)

            # # setting consensus fusion results
            # fusion_complement = 'consensus'
            # precision = bboxes_fusion_results.consensus_performance_metrics.get_model_precision()
            # recall = bboxes_fusion_results.consensus_performance_metrics.get_model_recall()
            # f1_score = bboxes_fusion_results.consensus_performance_metrics.get_model_f1_score()

            # # adding model performance metrics into the summary
            # item = [dataset_name, method_input_dataset_type, model_selection_method, top_t, f1_score_threshold,
            #         minimum_models_in_cluster, selected_models, fusion_method, fusion_complement, 
            #         precision, recall, f1_score, running_id]
            # bboxes_fusion_summary.append(item)

            # # setting unanimous fusion results
            # fusion_complement = 'unanimous'
            # precision = bboxes_fusion_results.unanimous_performance_metrics.get_model_precision()
            # recall = bboxes_fusion_results.unanimous_performance_metrics.get_model_recall()
            # f1_score = bboxes_fusion_results.unanimous_performance_metrics.get_model_f1_score()

            # # adding model performance metrics into the summary
            # item = [dataset_name, method_input_dataset_type, model_selection_method, top_t, f1_score_threshold,
            #         minimum_models_in_cluster, selected_models, fusion_method, fusion_complement, 
            #         precision, recall, f1_score, running_id]
            # bboxes_fusion_summary.append(item)

        if isinstance(bboxes_fusion_results, WeightedBoxesFusion):
            # computing weighted boxes fusion results
            wbf_bboxes_fusion_summary.extend(compute_wbf_results(running_id, bboxes_fusion_results, bboxes_fusion_method))

            # adding wbf results summary into the all summary 
            # bboxes_fusion_summary.extend(wbf_bboxes_fusion_summary)


    # print(f'wbf_bboxes_fusion_summary: {wbf_bboxes_fusion_summary}')
    # print(f'voting_bboxes_fusion_summary: {voting_bboxes_fusion_summary}')

    # saving model performance metrics summary in excel sheet
    if len(voting_bboxes_fusion_summary) > 0:
        # adding wbf results summary into the all summary 
        all_methods_bboxes_fusion_summary.extend(voting_bboxes_fusion_summary)

        # adding single models results summary into the voting summary
        voting_bboxes_fusion_summary.extend(single_models_bboxes_fusion_summary)

        # saving wbf fusion results 
        voting_bboxes_fusion_summary_filename = 'summary-' + input_dataset_type + '-' + running_id_text + '-' + \
                                                'voting-bboxes-fusion.xlsx'
        save_summary_metrics(voting_bboxes_fusion_summary, bboxes_fusion_folder, voting_bboxes_fusion_summary_filename)

    # saving model performance metrics summary in excel sheet
    if len(wbf_bboxes_fusion_summary) > 0:
        # adding wbf results summary into the all summary 
        all_methods_bboxes_fusion_summary.extend(wbf_bboxes_fusion_summary)

        # adding single models results summary into the voting summary
        wbf_bboxes_fusion_summary.extend(single_models_bboxes_fusion_summary)

        # saving wbf fusion results 
        wbf_bboxes_fusion_summary_filename = 'summary-' + input_dataset_type + '-' + running_id_text + '-' + \
                                                'wbf-bboxes-fusion.xlsx'
        save_summary_metrics(wbf_bboxes_fusion_summary, bboxes_fusion_folder, wbf_bboxes_fusion_summary_filename)

    # adding single models results summary into the voting summary
    all_methods_bboxes_fusion_summary.extend(single_models_bboxes_fusion_summary)

    # saving model performance metrics summary in excel sheet
    bboxes_fusion_summary_filename = 'summary-' + input_dataset_type + '-' + running_id_text + '-' + \
                                     'all-methods-bboxes-fusion.xlsx'
    save_summary_metrics(all_methods_bboxes_fusion_summary, bboxes_fusion_folder, bboxes_fusion_summary_filename)


def set_voting_results(running_id, bboxes_fusion_results, bboxes_fusion_method):
    # initializing objects 
    bboxes_fusion_summary = []

    # setting common columns for all methods
    dataset_name = bboxes_fusion_results.dataset_name
    method_input_dataset_type = bboxes_fusion_results.input_dataset_type
    model_selection_method = bboxes_fusion_results.models_selection_method_name
    top_t = bboxes_fusion_results.top_t
    f1_score_threshold = bboxes_fusion_results.f1_score_threshold
    minimum_models_in_cluster = bboxes_fusion_results.minimum_number_of_models
    selected_models = bboxes_fusion_results.selected_models
    fusion_method = bboxes_fusion_method

    # setting affirmative fusion results
    fusion_complement = 'affirmative'
    precision = bboxes_fusion_results.affirmative_performance_metrics.get_model_precision()
    recall = bboxes_fusion_results.affirmative_performance_metrics.get_model_recall()
    f1_score = bboxes_fusion_results.affirmative_performance_metrics.get_model_f1_score()

    # adding model performance metrics into the summary
    item = [dataset_name, method_input_dataset_type, model_selection_method, top_t, f1_score_threshold,
            minimum_models_in_cluster, selected_models, fusion_method, fusion_complement, 
            precision, recall, f1_score, running_id]
    bboxes_fusion_summary.append(item)

    # setting consensus fusion results
    fusion_complement = 'consensus'
    precision = bboxes_fusion_results.consensus_performance_metrics.get_model_precision()
    recall = bboxes_fusion_results.consensus_performance_metrics.get_model_recall()
    f1_score = bboxes_fusion_results.consensus_performance_metrics.get_model_f1_score()

    # adding model performance metrics into the summary
    item = [dataset_name, method_input_dataset_type, model_selection_method, top_t, f1_score_threshold,
            minimum_models_in_cluster, selected_models, fusion_method, fusion_complement, 
            precision, recall, f1_score, running_id]
    bboxes_fusion_summary.append(item)

    # setting unanimous fusion results
    fusion_complement = 'unanimous'
    precision = bboxes_fusion_results.unanimous_performance_metrics.get_model_precision()
    recall = bboxes_fusion_results.unanimous_performance_metrics.get_model_recall()
    f1_score = bboxes_fusion_results.unanimous_performance_metrics.get_model_f1_score()

    # adding model performance metrics into the summary
    item = [dataset_name, method_input_dataset_type, model_selection_method, top_t, f1_score_threshold,
            minimum_models_in_cluster, selected_models, fusion_method, fusion_complement, 
            precision, recall, f1_score, running_id]
    bboxes_fusion_summary.append(item)

    # returning summary
    return bboxes_fusion_summary

def compute_wbf_results(running_id, bboxes_fusion_results, bboxes_fusion_method):
    # initializing objects 
    bboxes_fusion_summary = []

    # setting common columns for all methods
    dataset_name = bboxes_fusion_results.dataset_name
    method_input_dataset_type = bboxes_fusion_results.input_dataset_type
    model_selection_method = bboxes_fusion_results.models_selection_method_name
    # top_t = bboxes_fusion_results.top_t
    f1_score_threshold = bboxes_fusion_results.f1_score_threshold
    # minimum_models_in_cluster = bboxes_fusion_results.minimum_number_of_models
    selected_models = bboxes_fusion_results.selected_models
    fusion_method = bboxes_fusion_method

    # setting wbf fusion results
    fusion_complement = 'wbf'
    precision = bboxes_fusion_results.wbf_metrics.get_model_precision()
    recall = bboxes_fusion_results.wbf_metrics.get_model_recall()
    f1_score = bboxes_fusion_results.wbf_metrics.get_model_f1_score()

    # adding model performance metrics into the summary
    item = [dataset_name, method_input_dataset_type, model_selection_method, 0, 
            f1_score_threshold, 0, selected_models, fusion_method, fusion_complement, 
            precision, recall, f1_score, running_id]
    bboxes_fusion_summary.append(item)

    # returning summary
    return bboxes_fusion_summary


def save_summary_metrics(bboxes_fusion_summary, folder, summary_filename):
    """
    Save the summary of model performance metrics to an Excel file.
    """

    # setting path and filename 
    bboxes_fusion_summary_filename = os.path.join(folder, summary_filename)

    # preparing columns name to list        
    column_names = [
        "dataset_name",
        "input_dataset_type",
        "model_selection_method",
        "top_t",
        "f1_score_threshold",
        "minimum_models_in_cluster",
        "selected_models",
        "fusion_method",
        "complement",
        "precision",
        "recall",
        "f1_score",
        "running_id"
    ]

    # creating dataframe from list 
    df = pd.DataFrame(bboxes_fusion_summary, columns=column_names)

    # sorting dataframe by f1-score, precision, and recall descending values
    df = df.sort_values(by=['f1_score', 'precision', 'recall'], ascending=False)

    # writing excel file from dataframe
    df.to_excel(bboxes_fusion_summary_filename, sheet_name='bboxes_fusion_summary', index=False)    