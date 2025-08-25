# ModelsSelectionMethSingleDivMeasuresFilter Class
# This class represents the method 3 for selecting object detection models


# Importing python libraries 
# import math
import pandas as pd
import matplotlib.pyplot as plt 

# Importing python modules
from common.manage_log import *
from model_diversity.entity.ModelsPairRelationshipMatrix import * 
from model_diversity.service.DiversityUtils import * 


# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class ModelsSelectionMethSingleDivMeasuresFilter:
    def __init__(self, method_short_name=None, method_description=None, 
                 dataset_name=None, models_diversity_measures=None,
                 method_results_folder=None, models_dic=None, top_t=None, f1_score_threshold=None):
        
        self.method_short_name = method_short_name
        self.method_description = method_description
        self.dataset_name = dataset_name
        self.models_diversity_measures = models_diversity_measures
        self.method_results_folder = method_results_folder
        self.models_dic = models_dic
        self.top_t = top_t
        self.f1_score_threshold = f1_score_threshold

        self.cor_selected_models = []
        self.dfm_selected_models = []
        self.dm_selected_models = []
        self.ia_selected_models = []
        self.qstat_selected_models = []

        self.method_complement_text = ''

    # Method 3 for selecting object dectection models 
    # Description: Selection of object detection models based on single diversity measure, effectiveness-driven detector filter and tiebreak criteria

    def execute(self):

        logging_info(f'Processing object detection model selection {self.method_short_name}: {self.method_description}')

        logging_info(f'Creating rank of Object Detectin Models')
        logging_info(f'')
        
        # Set display option to show all columns
        pd.set_option('display.max_columns', None)

        # setting the method complement text
        self.method_complement_text = f'Top_t: {self.top_t:02d}   |   F1-score threshold: {self.f1_score_threshold:.2f}'

        # creating diversity measure lists 
        cor_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'correlation_coefficient', 'model_1_f1_score', 'model_2_f1_score', 'average_f1_score']]
        dfm_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'double_fault_measure', 'model_1_f1_score', 'model_2_f1_score', 'average_f1_score']]
        dm_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'disagreement_measure', 'model_1_f1_score', 'model_2_f1_score', 'average_f1_score']]
        ia_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'interrater_agreement', 'model_1_f1_score', 'model_2_f1_score', 'average_f1_score']]
        qstat_df = self.models_diversity_measures.diversity_measures_df[['dataset_name', 'model_1', 'model_2', 'q_statistic', 'model_1_f1_score', 'model_2_f1_score', 'average_f1_score']]

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
                item = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                models_occurences_results_list.append(item)

        column_labels = ['correlation_coefficient', 'double_fault_measure', 'disagreement_measure',
                        'interrater_agreement', 'q_statistic', 'model_f1_score']
        
        # creating dataframe from list 
        self.models_occurence_results_df = pd.DataFrame(
                models_occurences_results_list, index=row_labels, columns=column_labels)
   
        # counting the ranked detectors for votation scheme 
        logging_info(f'Counting ranked detectors for votation scheme - top-t: {self.top_t}')
        logging_info(f'')

        print(f'Counting ranked detectors for votation scheme - top-t: {self.top_t}')
        print(f'')

        print(f'Creating models_occurence_results_df')
        print(self.models_occurence_results_df)
        print(f'-------------------------------------------')
        print(f'self.models_occurence_results_df types')
        print(self.models_occurence_results_df.dtypes)
        print(f'-------------------------------------------')

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
            model_1_f1_score = row['model_1_f1_score']          
            model_2_f1_score = row['model_2_f1_score']          

            number_of = self.models_occurence_results_df.loc[model_1].loc['correlation_coefficient']
            self.models_occurence_results_df.loc[model_1].loc['correlation_coefficient'] = number_of + 1
            self.models_occurence_results_df.loc[model_1].loc['model_f1_score'] = model_1_f1_score
            number_of = self.models_occurence_results_df.loc[model_2].loc['correlation_coefficient']
            self.models_occurence_results_df.loc[model_2].loc['correlation_coefficient'] = number_of + 1
            self.models_occurence_results_df.loc[model_2].loc['model_f1_score'] = model_2_f1_score

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
            model_1_f1_score = row['model_1_f1_score']          
            model_2_f1_score = row['model_2_f1_score']          

            number_of = self.models_occurence_results_df.loc[model_1].loc['double_fault_measure']
            self.models_occurence_results_df.loc[model_1].loc['double_fault_measure'] = number_of + 1
            self.models_occurence_results_df.loc[model_1].loc['model_f1_score'] = model_1_f1_score
            number_of = self.models_occurence_results_df.loc[model_2].loc['double_fault_measure']
            self.models_occurence_results_df.loc[model_2].loc['double_fault_measure'] = number_of + 1
            self.models_occurence_results_df.loc[model_2].loc['model_f1_score'] = model_2_f1_score

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
            model_1_f1_score = row['model_1_f1_score']          
            model_2_f1_score = row['model_2_f1_score']          

            number_of = self.models_occurence_results_df.loc[model_1].loc['disagreement_measure']
            self.models_occurence_results_df.loc[model_1].loc['disagreement_measure'] = number_of + 1
            self.models_occurence_results_df.loc[model_1].loc['model_f1_score'] = model_1_f1_score
            number_of = self.models_occurence_results_df.loc[model_2].loc['disagreement_measure']
            self.models_occurence_results_df.loc[model_2].loc['disagreement_measure'] = number_of + 1
            self.models_occurence_results_df.loc[model_2].loc['model_f1_score'] = model_2_f1_score

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
            model_1_f1_score = row['model_1_f1_score']          
            model_2_f1_score = row['model_2_f1_score']          

            number_of = self.models_occurence_results_df.loc[model_1].loc['interrater_agreement']
            self.models_occurence_results_df.loc[model_1].loc['interrater_agreement'] = number_of + 1
            self.models_occurence_results_df.loc[model_1].loc['model_f1_score'] = model_1_f1_score
            number_of = self.models_occurence_results_df.loc[model_2].loc['interrater_agreement']
            self.models_occurence_results_df.loc[model_2].loc['interrater_agreement'] = number_of + 1
            self.models_occurence_results_df.loc[model_2].loc['model_f1_score'] = model_2_f1_score

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
            model_1_f1_score = row['model_1_f1_score']          
            model_2_f1_score = row['model_2_f1_score']    
            # print(f'rubens f1_score')      
            print(f'model_1_f1_score: {model_1_f1_score}')
            print(f'model_2_f1_score: {model_2_f1_score}')

            number_of = self.models_occurence_results_df.loc[model_1].loc['q_statistic']
            self.models_occurence_results_df.loc[model_1].loc['q_statistic'] = number_of + 1
            self.models_occurence_results_df.loc[model_1].loc['model_f1_score'] = model_1_f1_score
            number_of = self.models_occurence_results_df.loc[model_2].loc['q_statistic']
            self.models_occurence_results_df.loc[model_2].loc['q_statistic'] = number_of + 1
            self.models_occurence_results_df.loc[model_2].loc['model_f1_score'] = model_2_f1_score

        logging_info(f'After counting votation scheme')
        logging_info(self.models_occurence_results_df)

        print(f'After counting votation scheme')
        print(self.models_occurence_results_df)

        # creating single occurrence lists for each diversity measure
        occurence_cor_df = self.models_occurence_results_df[['correlation_coefficient', 'model_f1_score']]
        occurence_dfm_df = self.models_occurence_results_df[['double_fault_measure', 'model_f1_score']]
        occurence_dm_df = self.models_occurence_results_df[['disagreement_measure', 'model_f1_score']]
        occurence_ia_df = self.models_occurence_results_df[['interrater_agreement', 'model_f1_score']]
        occurence_qstat_df = self.models_occurence_results_df[['q_statistic', 'model_f1_score']]

        filtered_cor_df = occurence_cor_df.loc[(occurence_cor_df['model_f1_score'] >= self.f1_score_threshold)]
        filtered_dfm_df = occurence_dfm_df.loc[(occurence_dfm_df['model_f1_score'] >= self.f1_score_threshold)]
        filtered_dm_df = occurence_dm_df.loc[(occurence_dm_df['model_f1_score'] >= self.f1_score_threshold)]
        filtered_ia_df = occurence_ia_df.loc[(occurence_ia_df['model_f1_score'] >= self.f1_score_threshold)]
        filtered_qstat_df = occurence_qstat_df.loc[(occurence_qstat_df['model_f1_score'] >= self.f1_score_threshold)]

        print(filtered_cor_df)
        print(filtered_dfm_df)
        print(filtered_dm_df)
        print(filtered_ia_df)
        print(filtered_qstat_df)

        # sorting by diversity measure and model F1-score 
        filtered_cor_df_ranked = filtered_cor_df.sort_values(by=['correlation_coefficient', 'model_f1_score'], ascending=[False, False])
        filtered_dfm_df_ranked = filtered_dfm_df.sort_values(by=['double_fault_measure', 'model_f1_score'], ascending=[False, False])
        filtered_dm_df_ranked = filtered_dm_df.sort_values(by=['disagreement_measure', 'model_f1_score'], ascending=[False, False])
        filtered_ia_df_ranked = filtered_ia_df.sort_values(by=['interrater_agreement', 'model_f1_score'], ascending=[False, False])
        filtered_qstat_df_ranked = filtered_qstat_df.sort_values(by=['q_statistic', 'model_f1_score'], ascending=[False, False])

        print(filtered_cor_df_ranked)
        print(f'---------------------------------')
        print(filtered_dfm_df_ranked)
        print(f'---------------------------------')
        print(filtered_dm_df_ranked)
        print(f'---------------------------------')
        print(filtered_ia_df_ranked)
        print(f'---------------------------------')
        print(filtered_qstat_df_ranked)
        print(f'---------------------------------')

        # setting filename and writing excel file from dataframe
        path_and_filename = os.path.join(
            self.method_results_folder, 
            (self.method_short_name + "-single-correlation-coeficient-results-ranked.xlsx")
        )
        filtered_cor_df_ranked.to_excel(path_and_filename, sheet_name='cor', index=True)
        
        path_and_filename = os.path.join(
            self.method_results_folder, 
            (self.method_short_name + "-single-double_fault_measure-results-ranked.xlsx")
        )
        filtered_dfm_df_ranked.to_excel(path_and_filename, sheet_name='dfm', index=True)
        
        path_and_filename = os.path.join(
            self.method_results_folder, 
            (self.method_short_name + "-single-disagreement_measure-results-ranked.xlsx")
        )
        filtered_dm_df_ranked.to_excel(path_and_filename, sheet_name='dm', index=True)
        
        path_and_filename = os.path.join(
            self.method_results_folder, 
            (self.method_short_name + "-single-interrater_agreement-results-ranked.xlsx")
        )
        filtered_ia_df_ranked.to_excel(path_and_filename, sheet_name='ia', index=True)
        
        path_and_filename = os.path.join(
            self.method_results_folder, 
            (self.method_short_name + "-single-q_statistic-results-ranked.xlsx")
        )
        filtered_qstat_df_ranked.to_excel(path_and_filename, sheet_name='qstat', index=True)     

        # select the top T object detection models from each diversity measure for the next step of model fusion
        self.cor_selected_models = DiversityUtils.select_models(filtered_cor_df_ranked, self.top_t)
        self.dfm_selected_models = DiversityUtils.select_models(filtered_dfm_df_ranked, self.top_t)
        self.dm_selected_models = DiversityUtils.select_models(filtered_dm_df_ranked, self.top_t)
        self.ia_selected_models = DiversityUtils.select_models(filtered_ia_df_ranked, self.top_t)
        self.qstat_selected_models = DiversityUtils.select_models(filtered_qstat_df_ranked, self.top_t)

        # saving selected models for each diversity measure 
        DiversityUtils.save_selected_models_json(self.method_short_name, self.method_description, 
                        self.dataset_name, self.cor_selected_models, "cor", 
                        self.method_results_folder, self.top_t, self.f1_score_threshold)
        DiversityUtils.save_selected_models_json(self.method_short_name, self.method_description, 
                        self.dataset_name, self.dfm_selected_models, "dfm", 
                        self.method_results_folder, self.top_t, self.f1_score_threshold)
        DiversityUtils.save_selected_models_json(self.method_short_name, self.method_description, 
                        self.dataset_name, self.dm_selected_models, "dm", 
                        self.method_results_folder, self.top_t, self.f1_score_threshold)
        DiversityUtils.save_selected_models_json(self.method_short_name, self.method_description, 
                        self.dataset_name, self.ia_selected_models, "ia", 
                        self.method_results_folder, self.top_t, self.f1_score_threshold)
        DiversityUtils.save_selected_models_json(self.method_short_name, self.method_description, 
                        self.dataset_name, self.qstat_selected_models, "qstat", 
                        self.method_results_folder, self.top_t, self.f1_score_threshold)
