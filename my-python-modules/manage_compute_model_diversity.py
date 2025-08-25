"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the management to compute models diversity.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
    Prof. Dr. Ricardo da Silva Torres - supervisor at AIN of Wageningen University & Research
Date: 30/05/2025
Version: 1.0
Command line:
- python .\my-python-modules\manage_compute_model_diversity.py --os windows --parameter compute_models_diversity_parameters_windows_swm_dataset.json
- python .\my-python-modules\manage_compute_model_diversity.py --os windows --parameter compute_models_diversity_parameters_windows_nc_dataset.json
"""

# Basic python and ML Libraries
import os
from datetime import datetime
import argparse

# Importing python modules
from common.manage_log import *
from common.tasks import Tasks

# from model_diversity.compute_models_diversity import * 
from model_diversity.compute_model_diversity import * 
from model_diversity.perform_bounding_boxes_fusion import * 

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'
NEW_FILE = True

# ###########################################
# Application Methods
# ###########################################

# ###########################################
# Methods of Level 1
# ###########################################

def main():
    """
    Main method to compute models diversity.    
    """
 
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("--os", type=str, help="Operational System")
    parser.add_argument("--parameter", type=str, help="Parameter File")
    args = parser.parse_args()
    print("Operational System:", args.os)
    print("Parameter File:", args.parameter)
    if args.os not in ['windows', 'linux']:
        print(f'')
        print(f'Attention: missing parameters in the command line: "linux" or "windows"')
        print(f'')
        exit()
    if args.parameter is None:
        print(f'')
        print(f'Attention: missing parameters in the command line: "parameter"')
        print(f'')
        exit()
        
    # creating Tasks object 
    processing_tasks = Tasks()

    # setting dictionary initial parameters for processing
    if args.os == "windows":
        full_path_project = 'E:\Doctorate\White-Mold-Applications\wm-model-diversity'
    else:
        full_path_project = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-diversity'
        
    # getting application parameters 
    processing_tasks.start_task('Getting application parameters')
    parameters_filename = args.parameter
    parameters = get_parameters(full_path_project, parameters_filename)
    processing_tasks.finish_task('Getting application parameters')

    # getting last running id
    processing_tasks.start_task('Getting running id')
    running_id = get_running_id(parameters)
    processing_tasks.finish_task('Getting running id')
       
    # setting output folder results
    processing_tasks.start_task('Setting result folders')
    diversity_experiment = 'div'
    set_result_folders(parameters, diversity_experiment)
    processing_tasks.finish_task('Setting result folders')
        
    # creating log file 
    processing_tasks.start_task('Creating log file')
    logging_create_log(
        parameters['results']['log_folder'], 
        parameters['results']['log_filename']
    )
    processing_tasks.finish_task('Creating log file')

    logging_info('White Mold Research')
    logging_info('Compute Model Diversity ' + LINE_FEED)

    logging_info(f'')
    logging_info(f'>> Set input image folders')
    logging_info(f'')
    logging_info(f'>> Get running id')
    logging_info(f'running id: {str(running_id)}')   
    logging_info(f'')
    logging_info(f'>> Set result folders') 
  
    # creating new instance of parameters file related to current running
    processing_tasks.start_task('Saving processing parameters')
    save_processing_parameters(parameters_filename, parameters)
    processing_tasks.finish_task('Saving processing parameters')
   
    # original_parameters = copy.deepcopy(parameters)
    
    # printing parameters of the processing    # computing the models diversity according parameters 
    processing_tasks.start_task('Computing models diversity')
    models_selection_methods, dataset_model_performance_metrics_dic = process_model_diversity_for_models_and_datasets(parameters)
    processing_tasks.finish_task('Computing models diversity')
   
    # performing bounding boxes fusion
    processing_tasks.start_task('Performing bounding boxes fusion methods')
    input_dataset_type = 'test'
    bboxes_fusion_method_results_dic = perform_bounding_boxes_fusion(parameters, models_selection_methods,
                                                                     input_dataset_type, dataset_model_performance_metrics_dic)
    logging_info(f'')
    logging_info(f'Performing bounding boxes fusion methods')

    for key, value in bboxes_fusion_method_results_dic.items():
        logging_info(f'> {key}')
        print(f'> {key}')        

        if 'voting_scheme' in key:
            logging_info(f'> affirmative_performance_metrics - precision: {value.affirmative_performance_metrics.get_model_precision()}')
            logging_info(f'> affirmative_performance_metrics - recall   : {value.affirmative_performance_metrics.get_model_recall()}')
            logging_info(f'> affirmative_performance_metrics - f1-score : {value.affirmative_performance_metrics.get_model_f1_score()}')
            logging_info('')
            logging_info(f'> consensus_performance_metrics - precision  : {value.consensus_performance_metrics.get_model_precision()}')
            logging_info(f'> consensus_performance_metrics - recall     : {value.consensus_performance_metrics.get_model_recall()}')
            logging_info(f'> consensus_performance_metrics - f1-score   : {value.consensus_performance_metrics.get_model_f1_score()}')
            logging_info('')
            logging_info(f'> unanimous_performance_metrics - precision  : {value.unanimous_performance_metrics.get_model_precision()}')
            logging_info(f'> unanimous_performance_metrics - recall     : {value.unanimous_performance_metrics.get_model_recall()}')
            logging_info(f'> unanimous_performance_metrics - f1-score   : {value.unanimous_performance_metrics.get_model_f1_score()}')
            logging_info('')

        if 'weighted_boxes_fusion' in key:
            logging_info(f'> weighted_boxes_fusion - precision  : {value.wbf_metrics.get_model_precision()}')
            logging_info(f'> weighted_boxes_fusion - recall     : {value.wbf_metrics.get_model_recall()}')
            logging_info(f'> weighted_boxes_fusion - f1-score   : {value.wbf_metrics.get_model_f1_score()}')
            logging_info('')
            print(f'> weighted_boxes_fusion - precision  : {value.wbf_metrics.get_model_precision()}')
            print(f'> weighted_boxes_fusion - recall     : {value.wbf_metrics.get_model_recall()}')
            print(f'> weighted_boxes_fusion - f1-score   : {value.wbf_metrics.get_model_f1_score()}')
            print(f'')

    processing_tasks.finish_task('Performing bounding boxes fusion methods')
   
    # finishing model training 
    logging_info('')
    logging_info('Finished the computing of Model Diversity' + LINE_FEED)
    print('Finished the computing of Model Diversity')

    # printing tasks summary 
    processing_tasks.finish_processing()
    logging_info(processing_tasks.to_string())


# ###########################################
# Methods of Level 2
# ###########################################

def get_parameters(full_path_project, parameters_filename):
    '''
    Get dictionary parameters for processing
    '''    
    # getting parameters 
    path_and_parameters_filename = os.path.join(full_path_project, parameters_filename)
    parameters = Utils.read_json_parameters(path_and_parameters_filename)

    # returning parameters 
    return parameters

def get_running_id(parameters):
    '''
    Get last running id to calculate the current id
    '''    

    # setting control filename 
    running_control_filename = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['running_control_filename'],
    )

    # getting control info 
    running_control = Utils.read_json_parameters(running_control_filename)

    # calculating the current running id 
    running_control['last_running_id'] = int(running_control['last_running_id']) + 1

    # updating running control file 
    running_id = int(running_control['last_running_id'])

    # saving file 
    Utils.save_text_file(running_control_filename, \
                         Utils.get_pretty_json(running_control), 
                         NEW_FILE)

    # updating running id in the processing parameters 
    parameters['processing']['running_id'] = running_id
    parameters['processing']['running_id_text'] = 'running-' + f'{running_id:04}'

    # returning the current running id
    return running_id

def set_result_folders(parameters, diversity_experiment):
    '''
    Set folder name of output results
    '''

    # resetting training results 
    # parameters['training_results'] = {}

    # creating results folders 
    main_folder = os.path.join(
        parameters['processing']['research_root_folder'],     
        parameters['results']['main_folder']
    )
    parameters['results']['main_folder'] = main_folder
    Utils.create_directory(main_folder)

    # setting and creating model folder 
    parameters['results']['model_folder'] = parameters['neural_network_model']['model_name']
    model_folder = os.path.join(
        main_folder,
        parameters['results']['model_folder']
    )
    parameters['results']['model_folder'] = model_folder
    Utils.create_directory(model_folder)

    # setting and creating experiment folder
    experiment_folder = os.path.join(
        model_folder,
        parameters['input']['experiment']['id'] + '-' + diversity_experiment
    )
    parameters['results']['experiment_folder'] = experiment_folder
    Utils.create_directory(experiment_folder)

    # setting and creating action folder of training
    action_folder = os.path.join(
        experiment_folder,
        parameters['results']['action_folder']
    )
    parameters['results']['action_folder'] = action_folder
    Utils.create_directory(action_folder)

    # setting and creating running folder 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    parameters['results']['running_folder'] = running_id_text + "-" + input_image_size + 'x' + input_image_size   
    running_folder = os.path.join(
        action_folder,
        parameters['results']['running_folder']
    )
    parameters['results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating others specific folders
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['results']['processing_parameters_folder']
    )
    parameters['results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    log_folder = os.path.join(
        running_folder,
        parameters['results']['log_folder']
    )
    parameters['results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

    performance_metrics_folder = os.path.join(
        running_folder,
        parameters['results']['performance_metrics_folder']
    )
    parameters['results']['performance_metrics_folder'] = performance_metrics_folder
    Utils.create_directory(performance_metrics_folder)

    matrices_folder = os.path.join(
        running_folder,
        parameters['results']['matrices_folder']
    )
    parameters['results']['matrices_folder'] = matrices_folder
    Utils.create_directory(matrices_folder)

    m01_all_div_folder = os.path.join(
        running_folder,
        parameters['results']['m01_all_div_folder']
    )
    parameters['results']['m01_all_div_folder'] = m01_all_div_folder
    Utils.create_directory(m01_all_div_folder)

    m02_single_div_folder = os.path.join(
        running_folder,
        parameters['results']['m02_single_div_folder']
    )
    parameters['results']['m02_single_div_folder'] = m02_single_div_folder
    Utils.create_directory(m02_single_div_folder)

    m03_single_div_filter_folder = os.path.join(
        running_folder,
        parameters['results']['m03_single_div_filter_folder']
    )
    parameters['results']['m03_single_div_filter_folder'] = m03_single_div_filter_folder
    Utils.create_directory(m03_single_div_filter_folder)

    bboxes_fusion_folder = os.path.join(
        running_folder,
        parameters['results']['bboxes_fusion_folder']
    )
    parameters['results']['bboxes_fusion_folder'] = bboxes_fusion_folder
    Utils.create_directory(bboxes_fusion_folder)

def save_processing_parameters(parameters_filename, parameters):
    '''
    Update parameters file of the processing
    '''

    logging_info(f'')
    logging_info(f'>> Save processing parameters of this running')

    # setting full path and log folder  to write parameters file 
    path_and_parameters_filename = os.path.join(
        parameters['results']['processing_parameters_folder'], 
        parameters_filename)

    # saving current processing parameters in the log folder 
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(parameters), 
                        NEW_FILE)

# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
