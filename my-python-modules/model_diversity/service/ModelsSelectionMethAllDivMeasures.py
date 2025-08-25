# ModelsSelectionMethod01 Class
# This class represents the method 1 for selecting object detection models


# Importing python libraries 
# import math
import pandas as pd
import matplotlib.pyplot as plt

# Importing python modules
from common.manage_log import *
from model_diversity.entity.ModelsPairRelationshipMatrix import * 

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class ModelsSelectionMethAllDivMeasures:
    def __init__(self, method_short_name=None, method_description=None, 
                 dataset_name=None, models_diversity_measures=None,
                 method_results_folder=None, models_dic=None, top_t=None):
        
        self.method_short_name = method_short_name
        self.method_description = method_description
        self.dataset_name = dataset_name
        self.models_diversity_measures = models_diversity_measures
        self.method_results_folder = method_results_folder
        self.models_dic = models_dic
        self.top_t = top_t

        self.selected_models = None
        # self.method_complement_text = ''

    # Method 1 for selecting object dectection models 
    # Description: Selection of object detection models based on all diversity measures together and no tiebreak criteria
    def execute(self):

        logging_info(f'Processing object detection model selection {self.method_short_name}: {self.method_description}')

        logging_info(f'Creating rank of Object Detectin Models')
        logging_info(f'')
        
        # Set display option to show all columns
        pd.set_option('display.max_columns', None)

        # setting the method complement text
        # self.method_complement_text = f'Top_t: {self.top_t:02d}'

        # creating diversity measure lists 
        cor_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'correlation_coefficient']]
        dfm_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'double_fault_measure']]
        dm_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'disagreement_measure']]
        ia_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'interrater_agreement']]
        qstat_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'q_statistic']]

        # sorting measures list to get the higher diversity among the detectors

        # Correlation coefficient - Lower value >> better diversity >> Ascending
        # Double Fault Measure - Lower value >> better diversity >>Ascending
        # Disagreement Measure - Higher value >> better diversity >> Descending
        # Interrater Agreement (κ) - Lower value >> better diversity >> Ascending
        # Q-statistic - Lower value >> better diversity >> Ascending 
        cor_df_sorted = cor_df.sort_values(by=['correlation_coefficient'], ascending=True)
        dfm_df_sorted = dfm_df.sort_values(by=['double_fault_measure'], ascending=True)
        dm_df_sorted = dm_df.sort_values(by=['disagreement_measure'], ascending=False)
        ia_df_sorted = ia_df.sort_values(by=['interrater_agreement'], ascending=True)
        qstat_df_sorted = qstat_df.sort_values(by=['q_statistic'], ascending=True)

        logging_info(f'Diversity Measures after sorting')
        logging_info(cor_df_sorted.head())
        logging_info(dfm_df_sorted.head())
        logging_info(dm_df_sorted.head())
        logging_info(ia_df_sorted.head())
        logging_info(qstat_df_sorted.head())

        print(f'Diversity Measures after sorting')
        print(cor_df_sorted.head())
        print(f'')
        print(dfm_df_sorted.head())
        print(f'')
        print(dm_df_sorted.head())
        print(f'')
        print(ia_df_sorted.head())
        print(f'')
        print(qstat_df_sorted.head())
        print(f'')

        # built occurences matrix to select the t top object detection models
        logging_info(f'')
        logging_info(f'Builting occurences matrix for the object detection models')
        logging_info(f'')

        print(f'')
        print(f'Builting occurences matrix for the object detection models')
        print(f'')

        models_occurences_results_list = []
        row_labels = []
        for model_key, model_value in self.models_dic.items():
            if model_value[1]:
                row_labels.append(model_key)
                item = [0, 0, 0, 0, 0, 0]
                models_occurences_results_list.append(item)

        column_labels = ['correlation_coefficient', 'double_fault_measure', 'disagreement_measure',
                        'interrater_agreement', 'q_statistic', 'total']
        
        # creating dataframe from list 
        self.models_occurence_results_df = pd.DataFrame(
                models_occurences_results_list, index=row_labels, columns=column_labels)
   
        # counting the ranked detectors for votation scheme 
        logging_info(f'Counting ranked detectors for votation scheme - top-t: {self.top_t}')
        logging_info(f'')

        print(f'Counting ranked detectors for votation scheme - top-t: {self.top_t}')
        print(f'')

        # counting correlation coefficient 
        count_t = 0
        for index, row in cor_df_sorted.iterrows():
            if count_t >= self.top_t:
                print(f'stop at {count_t}')
                break

            count_t += 1
            dataset_name = row['dataset_name']
            model_1 = row['model_1']
            model_2 = row['model_2']
            correlation_coefficient = row['correlation_coefficient']          

            number_of = self.models_occurence_results_df.loc[model_1].loc['correlation_coefficient']
            self.models_occurence_results_df.loc[model_1].loc['correlation_coefficient'] = number_of + 1
            number_of = self.models_occurence_results_df.loc[model_2].loc['correlation_coefficient']
            self.models_occurence_results_df.loc[model_2].loc['correlation_coefficient'] = number_of + 1

        # counting double fault measure 
        count_t = 0
        for index, row in dfm_df_sorted.iterrows():
            if count_t >= self.top_t:
                print(f'stop at {count_t}')
                break

            count_t += 1
            dataset_name = row['dataset_name']
            model_1 = row['model_1']
            model_2 = row['model_2']
            double_fault_measure = row['double_fault_measure']          

            number_of = self.models_occurence_results_df.loc[model_1].loc['double_fault_measure']
            self.models_occurence_results_df.loc[model_1].loc['double_fault_measure'] = number_of + 1
            number_of = self.models_occurence_results_df.loc[model_2].loc['double_fault_measure']
            self.models_occurence_results_df.loc[model_2].loc['double_fault_measure'] = number_of + 1

        # counting disagreement measure
        count_t = 0
        for index, row in dm_df_sorted.iterrows():
            if count_t >= self.top_t:
                print(f'stop at {count_t}')
                break

            count_t += 1
            dataset_name = row['dataset_name']
            model_1 = row['model_1']
            model_2 = row['model_2']
            disagreement_measure = row['disagreement_measure']          

            number_of = self.models_occurence_results_df.loc[model_1].loc['disagreement_measure']
            self.models_occurence_results_df.loc[model_1].loc['disagreement_measure'] = number_of + 1
            number_of = self.models_occurence_results_df.loc[model_2].loc['disagreement_measure']
            self.models_occurence_results_df.loc[model_2].loc['disagreement_measure'] = number_of + 1

        # counting interrater agreement
        count_t = 0
        for index, row in ia_df_sorted.iterrows():
            if count_t >= self.top_t:
                print(f'stop at {count_t}')
                break

            count_t += 1
            dataset_name = row['dataset_name']
            model_1 = row['model_1']
            model_2 = row['model_2']
            interrater_agreement = row['interrater_agreement']          

            number_of = self.models_occurence_results_df.loc[model_1].loc['interrater_agreement']
            self.models_occurence_results_df.loc[model_1].loc['interrater_agreement'] = number_of + 1
            number_of = self.models_occurence_results_df.loc[model_2].loc['interrater_agreement']
            self.models_occurence_results_df.loc[model_2].loc['interrater_agreement'] = number_of + 1

        # counting q statistic
        count_t = 0
        for index, row in qstat_df_sorted.iterrows():
            if count_t >= self.top_t:
                print(f'stop at {count_t}')
                break

            count_t += 1
            dataset_name = row['dataset_name']
            model_1 = row['model_1']
            model_2 = row['model_2']
            q_statistic = row['q_statistic']      

            number_of = self.models_occurence_results_df.loc[model_1].loc['q_statistic']
            self.models_occurence_results_df.loc[model_1].loc['q_statistic'] = number_of + 1
            number_of = self.models_occurence_results_df.loc[model_2].loc['q_statistic']
            self.models_occurence_results_df.loc[model_2].loc['q_statistic'] = number_of + 1

        logging_info(f'After counting votation scheme')
        logging_info(self.models_occurence_results_df)

        print(f'After counting votation scheme')
        print(self.models_occurence_results_df)

        # summarizing counting 
        for model_key, model_value in self.models_dic.items():
            if model_value[1]:
                cor   = self.models_occurence_results_df.loc[model_key].loc['correlation_coefficient']
                dfm   = self.models_occurence_results_df.loc[model_key].loc['double_fault_measure'] 
                dm    = self.models_occurence_results_df.loc[model_key].loc['disagreement_measure'] 
                ia    = self.models_occurence_results_df.loc[model_key].loc['interrater_agreement'] 
                qstat = self.models_occurence_results_df.loc[model_key].loc['q_statistic'] 
                number_of = cor + dfm + dm + ia + qstat
                self.models_occurence_results_df.loc[model_key].loc['total'] = number_of

        # setting filename 
        path_and_filename = os.path.join(
            self.method_results_folder,
            "models-votation-results.xlsx"
        )

        # writing excel file from dataframe
        self.models_occurence_results_df.to_excel(
            path_and_filename, sheet_name='models_occurences_results', index=True)
        
        # sorting final 
        self.models_occurence_results_df_sorted = self.models_occurence_results_df.sort_values(by=['total'], ascending=False)

        # setting filename 
        path_and_filename = os.path.join(
            self.method_results_folder,
            "models-votation-results-ranked.xlsx"
        )

        # writing excel file from dataframe
        self.models_occurence_results_df_sorted.to_excel(
            path_and_filename, sheet_name='models_occurences_results', index=True)

        # select the top T object detection models for the next step of model fusion
        self.selected_models = []
        cont = 0
        for model_index, _ in self.models_occurence_results_df_sorted.iterrows():
            if cont >= self.top_t:
                break
            print(f'model_index: {model_index}')
            self.selected_models.append(model_index)
            cont += 1

        # saving model selection in JSON format
        selected_models_dic = {
             "method": self.method_short_name,
             "description": self.method_description,
             "top_t": self.top_t
        }
        selected_models_dic["selected_models"] = []        

        # initializing dictionary and variables 
        for model in self.selected_models:
            selected_models_dic["selected_models"].append(model)
             
        # setting filename 
        json_filename = os.path.join(
            self.method_results_folder,
            ("m01-all-div_" + self.dataset_name + "_selected_models_" +  ".json")
        )

        # Save as JSON
        with open(json_filename, "w") as out_f:
                json.dump(selected_models_dic, out_f, indent=2)




