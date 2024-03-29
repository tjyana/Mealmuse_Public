import itertools
import re
import numpy as np
import pandas as pd
import openai
import csv 

###helll00000"""


def combinations_of_two(ingredients_input): ###dealt with the issue of missing space crash

    '''
    The function generates all unique pairs of ingredients that can be made from the input list of ingredients.
    NOTE FOR FRONT-END: The output of this function is the input for data_query()
    '''

    ingredients_combinations = []
    powerpowerset = []
    ingredients = re.split(r',', ingredients_input.strip())
    ingredients_list = list(set(ingredient.strip() for ingredient in ingredients))
    for r in range(len(ingredients_list)+1):
        combinations = itertools.combinations(ingredients_list, r)
        #powerset.extend(subset for subset in combinations if len(subset) > 1)
        for comb in combinations:
            if len(comb) > 1:
                if len(comb) < 3:
                    ingredients_combinations.append(comb)
                else:
                    powerpowerset.append(comb)
                    for power in powerpowerset:
                        lowerset = []
                        combins = itertools.combinations(power, 2)
                        for arrange in combins:
                            lowerset.append(arrange)
                    ingredients_combinations.append(lowerset)
    return ingredients_combinations

'''-----------------------------------------------------------------------------------------------------------'''

def data_query(df, ingredients_combinations): ##Added a penalty of -5 for pairings that are not in the dataframe
    """
    INPUT: get_dataframe(), combinations_of_two()
    NOTE FOR FRONT-END: The output of this function is the input for muse_comb()
    """
    data = []
    for combination in ingredients_combinations:
        if len(combination) < 3:
            ingredient1, ingredient2 = combination
            query_str = f'(ingredient1 == "{ingredient1}" & ingredient2 == "{ingredient2}") | (ingredient1 == "{ingredient2}" & ingredient2 == "{ingredient1}")'
            score = df.query(query_str)['scaled_col'].values
            if len(score) > 0:
                data.append({'Combination': combination, 'Score': score})
            else:
                continue
        else:
            scores = []
            for i in combination:
                ingredient1, ingredient2 = i
                query_str = f'(ingredient1 == "{ingredient1}" & ingredient2 == "{ingredient2}") | (ingredient1 == "{ingredient2}" & ingredient2 == "{ingredient1}")'
                score = df.query(query_str)['scaled_col'].values
                if len(score) > 0:
                    scores.append(score[0])
                else:
                    scores.append(-5)
            data.append({'Combination': combination, 'Score': scores})
        
    df_comb = pd.DataFrame(data) 
    return df_comb

'''-----------------------------------------------------------------------------------------------------------'''
def get_dataframe(file):
    """reads the parquet.gzip file ["Halved-DF.parquet.gzip"]
    NOTE FOR FRON-END: the output of this function is an input for data_query()
    """
    df = pd.read_parquet(file)

    return df
'''-----------------------------------------------------------------------------------------------------------'''
def final_recipes(recipes, scores):  ###<=== Function for evaluating if the score passes the threshold and regenerating if it doesn't
    """
    This evaluates whether the score of a recipe passes or fails the threshold.
    If the recipe doesn't meet the threshold after 3 attempts, the last generated recipe is added.
    INPUT: Output of the recipe generator function
    NOTE FOR FRONT-END: it's important to make sure that the outputs of the new recipe generator are the same as the
                        old version for this function to still work.
                        optimized_gptrecipe() and scoring_model() must be replaced with the actual functions
    """
    final_recipes = {"Title": [], "Ingredients": [], "Instructions": []}
    threshold = 2
    
    for i in range(len(recipes["Title"])):
        if scores[i] >= threshold:
            final_recipes["Title"].append(recipes["Title"][i])
            final_recipes["Ingredients"].append(recipes["Ingredients"][i])
            final_recipes["Instructions"].append(recipes["Instructions"][i])
        else:
            n = 0
            tmp_recipe = {
                "Title":recipes["Title"][i],
                "Ingredients":recipes["Ingredients"][i],
                "Instructions":recipes["Instructions"][i]
                         }
            last_recipe = {
                "Title":recipes["Title"][i],
                "Ingredients":recipes["Ingredients"][i],
                "Instructions":recipes["Instructions"][i]
                         }
            while n < 3:
                new_recipe = optimized_gptrecipe(tmp_recipe["Ingredients"][0]) ###<=== insert actual recipe generator
                new_score = scoring_model(new_recipe["Instructions"][0]) ###<=== insert the actual scoring model function here
                if new_score >= threshold:
                    final_recipes["Title"].append(new_recipe["Title"])
                    final_recipes["Ingredients"].append(new_recipe["Ingredients"])
                    final_recipes["Instructions"].append(new_recipe["Instructions"])
                    break  # Exit loop if the new recipe passes the threshold
                else:
                    last_recipe = new_recipe  # Update tmp_recipe with the new recipe if the threshold isn't met
                    n += 1
            else: # Add the last generated recipe if the loop completes without finding a passing recipe
                final_recipes["Title"].append(last_recipe["Title"])
                final_recipes["Ingredients"].append(last_recipe["Ingredients"])
                final_recipes["Instructions"].append(last_recipe["Instructions"])
                break  # Exit the outer loop to prevent an unending loop

    return final_recipes


'''-----------------------------------------------------------------------------------------------------------'''
def muse_comb(data_query_df): ###If this takes too long, consider taking the nested calculate_sum(array) outside of the function
    '''
     the function calculates the sum of the "Score" values and returns the three combinations with the largest sums
     OUTPUT: [['yeast', 'butter', 'eggs', 'pepper', 'cabbage', 'pork', 'flour', 'sugar'],
                 ['butter', 'eggs', 'pepper', 'cabbage', 'pork', 'flour', 'sugar'],
                 ['yeast', 'butter', 'eggs', 'pepper', 'cabbage', 'flour', 'sugar']]
                 
     NOTE FOR FRONT-END: The return is a list of lists so access the values by indexing e.g. output[0]
     
                         The output of this function is the input for the recipe generator
                         
                         We might need a function to convert each lists into strings if
                         the recipe generator doesn't do this automatically.
    '''
    
    def calculate_sum(array):
        return sum(array)
    
    def ingredients_to_lists(lists):
        ingredients_list = []
        for i in range(3):
            tmp_list = []
            for x in lists[i]:
                tmp_list.append(x[0])
                tmp_list.append(x[1])
            ingredients_list.append(list(set(tmp_list)))
    
        return ingredients_list

    for i in range(len(data_query_df)):
        data_query_df["Sum"] = data_query_df["Score"].apply(calculate_sum)

    max_values = data_query_df.nlargest(3, "Sum")
    
    max_values = max_values["Combination"].reset_index(drop=True)
    
    ingredients_lists = ingredients_to_lists(max_values) 
    
    return ingredients_lists

'''--------------------------------------------------------------------------------------------------------------'''

#Guide...

# ingredients_combinations = combinations_of_two(input("Enter the list of ingredients separated by commas: "))
# #butter, honey, salt, olive oil, mexican seasoning, bread, chicken
# df = pd.read_csv("Compatibility.csv")
# df_comb = data_matching(df, ingredients_combinations)
# combinatins = muse_comb(df_comb)
# recipes = gptrecipe(combinatins)
# recipes
