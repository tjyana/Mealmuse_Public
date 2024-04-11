import itertools
import re
import numpy as np
import pandas as pd
import openai
import pandas as pd
import itertools
import streamlit as st

# Anna's imports
from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from gradio_client import Client

# Set API Key
open_key = st.secrets['key_a']

###helll00000"""


'''
former get_ingredients_combinations, new compatibility function.
Input:
    ingredients = "chicken,tomato,onion,mushroom"
    find_top_3_groups(ingredients,verified_pairings)
Output:
    [('chicken', 'onion', 'mushroom'),
    ('chicken', 'onion'),
    ('chicken', 'mushroom')]

'''

def count_verified_pairings(combination, verified_pairings):
    count = 0
    for pair in itertools.combinations(combination, 2):
        print(pair[0])
        if pair in verified_pairings or (pair[1], pair[0]) in verified_pairings:
            count += 1
    return count


# Function to find the top 3 largest acceptable groups of ingredients
# Input: user input (string) ingredients
# Output: 3 ingredients combinations (list) candidates
def find_top_3_groups(ingredients, verified_pairings):

    ingredients = [ingredient.strip() for ingredient in ingredients.split(',')]
    n = len(ingredients)
    valid_combinations = []  # Store all combinations that meet the criteria

    # Generate and check all combinations for meeting the required pairings
    for size in range(n, 1, -1):  # We start from n and go down since we're interested in larger groups first
        for combination in itertools.combinations(ingredients, size):
            print(combination)
            required_pairings = size * (size - 1) / 2 * 0.8
            verified_count = count_verified_pairings(combination, verified_pairings)
            if verified_count >= required_pairings:
                valid_combinations.append(combination)  # Add valid combination to the list

    # Sort the valid combinations by their size (number of ingredients) in descending order
    valid_combinations.sort(key=lambda x: len(x), reverse=True)
    print('====================')
    print(valid_combinations)
    # print(verified_pairings)
    print('====================')
    # Return the top 3 largest groups, but there might be fewer than 3
    return valid_combinations[:3]

"""End of new compatibility function"""


def get_ingredients_combinations(ingredients: str):
    ingredients = [x.strip() for x in ingredients.split(',')]
    candidates = []
    for i in range(2, len(ingredients) + 1):
        for c in itertools.combinations(ingredients, i):
            keep = True
            min_score = 1000
            for a, b in itertools.combinations(c, 2):
                min_score = min(scores.get((a, b), 0), min_score)
                if min_score == 0:
                    keep = False
                    break
            if keep:
                candidates.append((c, min_score))
        return candidates

def combinations_of_two(ingredients_input):

    '''The function generates all unique pairs of ingredients that can be made from the input list of ingredients.'''

    powerset = []
    powerpowerset = []
    ingredients = re.split(r',\s*', ingredients_input.strip())
    ingredients_list = list(set(ingredients))
    for r in range(len(ingredients_list)+1):
        combinations = itertools.combinations(ingredients_list, r)
        #powerset.extend(subset for subset in combinations if len(subset) > 1)
        for comb in combinations:
            if len(comb) > 1:
                if len(comb) < 3:
                    powerset.append(comb)
                else:
                    powerpowerset.append(comb)
                    for power in powerpowerset:
                        lowerset = []
                        combins = itertools.combinations(power, 2)
                        for arrange in combins:
                            lowerset.append(arrange)
                    powerset.append(lowerset)
    return powerset

'''-----------------------------------------------------------------------------------------------------------'''

def data_matching(df, ingredients_combinations):
    '''
     The function generates all unique pairs of ingredients that can be made from the input list of ingredients.

    '''
    data = []
    for combination in ingredients_combinations:
        if len(combination) < 3:
            ingredient1, ingredient2 = combination
            score = df[(df['ingredient1'] == ingredient1) & (df['ingredient2'] == ingredient2)]['scaled_col'].values
            if len(score) > 0:
                data.append({'Combination': combination, 'Score': score})
            else:
                continue
        else:
            scores = []
            for i in combination:
                ingredient1, ingredient2 = i
                score = df[(df['ingredient1'] == ingredient1) & (df['ingredient2'] == ingredient2)]['scaled_col'].values
                if len(score) > 0:
                    scores.append(score[0])
                else:
                    scores.append(0)
            data.append({'Combination': combination, 'Score': scores})

    df_comb = pd.DataFrame(data)
    return df_comb

'''-----------------------------------------------------------------------------------------------------------'''

def muse_comb(df):
    '''
     the function calculates the products and cube roots of 'Score' values in a DataFrame, then returns the top 3 'Combination'
     values where the cube root is greater than 0
    '''
    product = []
    for i in range(len(df)):
        product.append(np.prod(df['Score'][i]))

    df['Product'] = product
    df['cbrt'] = np.cbrt(product)

    max_values = df.loc[df[df['cbrt'] > 0]['cbrt'].nlargest(3).index, 'Combination']
    return max_values

'''-----------------------------------------------------------------------------------------------------------'''

def prompt_muse(ingredients):

    '''
    The function returns the recipe generated by the model.
    '''

    openai.api_key = open_key
    ingredients = ', '.join(ingredients)
    prompt = f'Using only these {ingredients} give me a recipe with the format of Title, Ingredients and Instructions only'

    recipe = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {'role': 'system', 'content': 'you are a world-class chef with innovative recipes'},
            {'role': 'user', 'content': prompt},
                    ],
        max_tokens = 500
    )

    return recipe

'''-----------------------------------------------------------------------------------------------------------'''

def get_recipe_info(recipe):
    content = recipe["choices"][0]["message"]["content"]
    title_match = re.search(r"Title:(.*?)\n\nIngredients:", content, re.DOTALL)
    title = title_match.group(1) if title_match else None

    ingredients_match = re.search(r"Ingredients:(.+?)\n\nInstructions:", content, re.DOTALL)
    ingredients = ingredients_match.group(1).strip() if ingredients_match else None

    instructions_match = re.search(r"Instructions:(.+)", content, re.DOTALL)
    instructions = instructions_match.group(1).strip() if instructions_match else None

    return title, ingredients, instructions


def gptrecipe(max_values):

    '''
    function takes a list of ingredient combinations as input. it generates a unique set of ingredients,
    and uses the prompt_muse function to generate a recipe based on these ingredients
    '''
    muse_recipes = []
    titles = []
    ingredients_list = []
    instructions_list = []
    for value in max_values:
        unique_values_set = set(val for pair in value for val in pair)
        ingredients = ', '.join(unique_values_set)
        muse_recipes.append(prompt_muse(ingredients))

    for recipes in muse_recipes:
        recipe = recipes["choices"][0]["message"]["content"]

        title_match = re.search(r"Title:(.*?)\n\nIngredients:", recipe, re.DOTALL)
        title = title_match.group(1) if title_match else None
        titles.append(title)

        # Use regex to extract ingredients
        ingredients_match = re.search(r"Ingredients:(.+?)\n\nInstructions:", recipe, re.DOTALL)
        ingredients = ingredients_match.group(1).strip() if ingredients_match else None
        ingredients_list.append(ingredients)

        instructions_match = re.search(r"Instructions:(.+)", recipe, re.DOTALL)
        instructions = instructions_match.group(1).strip() if instructions_match else None
        instructions_list.append(instructions)

    evaluation_dict = {
        'Title': [],
        'Ingredients': [],
        'Instructions': []
    }

    for title, ingredients, instructions in zip(titles, ingredients_list, instructions_list):
    # Update the dictionary with new values for each key
        evaluation_dict['Title'] = evaluation_dict.get('Title', []) + [title]
        evaluation_dict['Ingredients'] = (evaluation_dict.get('Ingredients', []) + [ingredients])
        evaluation_dict['Instructions'] = evaluation_dict.get('Instructions', []) + [instructions]


    return evaluation_dict


'''-----------------------------------------------------------------------------------------------------------'''
def final_recipes(recipes, scores, model):  ###<=== Function for evaluatimg if the score passes the threshold and regenerating if it doesn't
    """
    This evaluates whether the score of a recipe passes or fails the threshold.
    If the recipe doesn't meet the threshold after 3 attempts, the last generated recipe is added.
    """
    final_recipes = {"Title": [], "Ingredients": [], "Instructions": []}
    threshold = 0.4

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
                new_recipe = gptrecipe(tmp_recipe["Ingredients"][0])
                new_score = model.predict_proba(new_recipe["Instructions"][0]) ###<=== insert the actual scoring model function here
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
#image generation function

def imagegen(title):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=f"{title}",
        size="1024x1024",
        quality="standard",
        n=1,

    )
    return response



#Guide...

# ingredients_combinations = combinations_of_two(input("Enter the list of ingredients separated by commas: "))
# #butter, honey, salt, olive oil, mexican seasoning, bread, chicken
# df = pd.read_csv("Compatibility.csv")
# df_comb = data_matching(df, ingredients_combinations)
# combinatins = muse_comb(df_comb)
# recipes = gptrecipe(combinatins)
# recipes









####################################
# Christine's code
####################################



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








####################################
# Anna's code
####################################






def recipe_generator(ingredients):
    MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

    prefix = "items: "
    generation_kwargs = {
        "max_length": 512,
        "min_length": 64,
        "no_repeat_ngram_size": 3,
        "do_sample": True,
        "top_k": 60,
        "top_p": 0.95
    }
    special_tokens = tokenizer.all_special_tokens
    tokens_map = {
        "<sep>": "--",
        "<section>": "\n"
    }
    def skip_special_tokens(text, special_tokens):
        for token in special_tokens:
            text = text.replace(token, "")

        return text

    def target_postprocessing(texts, special_tokens):
        if not isinstance(texts, list):
            texts = [texts]

        new_texts = []
        for text in texts:
            text = skip_special_tokens(text, special_tokens)

            for k, v in tokens_map.items():
                text = text.replace(k, v)

            new_texts.append(text)

        return new_texts

    def generation_function(texts):
        _inputs = texts if isinstance(texts, list) else [texts]
        inputs = [prefix + inp for inp in _inputs]
        inputs = tokenizer(
            inputs,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="jax"
        )

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        generated = output_ids.sequences
        generated_recipe = target_postprocessing(
            tokenizer.batch_decode(generated, skip_special_tokens=False),
            special_tokens
        )
        return generated_recipe
    recipe = generation_function(ingredients)
    return recipe


def convert_to_dictionary(recipes):
    recipe_dicts = []
    for recipe in recipes:
        recipe_dict = {}
        parts = recipe.split('\n')
        for part in parts:
            key, value = part.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key in ['ingredients', 'directions']:
                items = [f"{i+1}. {info.strip().capitalize()}" for i, info in enumerate(value.split("--"))]
                recipe_dict[key] = "\n".join(items)
            else:
                recipe_dict[key] = value
        recipe_dicts.append(recipe_dict)

    return recipe_dicts


def image_generator(recipe):
    client = Client("https://playgroundai-playground-v2-5.hf.space/--replicas/o9oxl/")
    result = client.predict(
            recipe,
            " ",
            False,
            820,
            1024,
            1024,
            3,
            True,
            api_name="/run"
    )

    image_path = result[0][0]['image']
    return image_path
