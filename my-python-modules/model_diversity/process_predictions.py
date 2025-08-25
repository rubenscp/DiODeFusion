"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the management of all predictions of the object detection models writing them as Json format.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
    Prof. Dr. Ricardo da Silva Torres - supervisor at AIN of Wageningen University & Research
Date: 28/05/2025
Version: 1.0
"""

import json  

def convert_and_save_predictions(model_name, input_dataset_type, predictions_list, output_filename, classes):

    # initializing dictionary and variables 
    predictions_dic = {
        "model_name" : model_name,
        "input_dataset_type" : input_dataset_type,
        "number_of_images" : len(predictions_list),
        "number_of_predictions" : 0,
        "image_predictions" : {}
    }
    number_of_predictions = 0

    # processing predictions list 
    for image_filename in predictions_list:
        print(f'image_filename: {image_filename}')
        predictions_dic['image_predictions'][image_filename] = {
            "ground_truths": {
                "boxes" : [],
                "labels" : [],
                "label_names" : []
            },
            "predictions": {
                "boxes" : [],
                "scores" : [],
                "labels" : [],
                "label_names" : []
            }
        }
        predictions_dic['image_predictions'][image_filename]['ground_truths']['boxes'] = predictions_list[image_filename][0][0]['boxes'].tolist()
        predictions_dic['image_predictions'][image_filename]['ground_truths']['labels'] = predictions_list[image_filename][0][0]['labels'].tolist()
        label_names = get_label_names(
                        predictions_dic['image_predictions'][image_filename]['ground_truths']['labels'],
                        classes)
        predictions_dic['image_predictions'][image_filename]['ground_truths']['label_names'] = label_names

        if len(predictions_list[image_filename][1]) > 0:
            predictions_dic['image_predictions'][image_filename]['predictions']['boxes'] = predictions_list[image_filename][1][0]['boxes'].tolist()
            predictions_dic['image_predictions'][image_filename]['predictions']['scores'] = predictions_list[image_filename][1][0]['scores'].tolist()
            predictions_dic['image_predictions'][image_filename]['predictions']['labels'] = predictions_list[image_filename][1][0]['labels'].tolist()
            label_names = get_label_names(
                            predictions_dic['image_predictions'][image_filename]['predictions']['labels'],
                            classes)
            predictions_dic['image_predictions'][image_filename]['predictions']['label_names'] = label_names

        # counting number of predictions 
        number_of_predictions += len(predictions_dic["image_predictions"][image_filename]['predictions']['labels'])

    # updating number of predictions 
    predictions_dic['number_of_predictions'] = number_of_predictions

    # Save as JSON
    with open(output_filename, "w") as out_f:
        json.dump(predictions_dic, out_f, indent=2)

    print(f'')
    print(f"JSON saved to {output_filename}")
    print(f'')


def get_label_names(labels, classes):
    # initializing list of label names 
    label_names = []

    # creating label names from label if 
    for label_id in labels:
        label_names.append(classes[label_id])

    # returning label names 
    return label_names


def get_model_predictions(path_and_filename_model_predictions):

    # initializing predictions list 
    predictions_list = []

    # getting predictions list from JSON file 
    with open(path_and_filename_model_predictions, 'r') as file:
        predictions_list = json.load(file)

    # # checking the size of the model name 
    # if len(predictions_list['model_name']) > 25: 
    #     predictions_list['model_name'] = predictions_list['model_name'][:25]

    # returning predictions list 
    return predictions_list