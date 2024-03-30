import itertools
import re
import numpy as np
import pandas as pd
import openai
# import csv

openkagi = st.secrets['key_a']

###helll00000"""


# scores = {}
# with open('Compatibility.csv') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     next(spamreader)
#     for a, b, score in spamreader:
#         scores[a, b] = float(score)

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
import pandas as pd
# filtered_df = pd.read_parquet('df.parquet.gzip')

# # Vectorized operations to strip quotes and create tuples
# ingredient1 = filtered_df['ingredient1'].str.strip("'")
# ingredient2 = filtered_df['ingredient2'].str.strip("'")

# # Use zip to pair elements and convert to set directly
# verified_pairings = set(zip(ingredient1, ingredient2))


import itertools
def count_verified_pairings(combination, verified_pairings):
    count = 0
    for pair in itertools.combinations(combination, 2):
        print(pair[0])
        if pair in verified_pairings or (pair[1], pair[0]) in verified_pairings:
            count += 1
    return count


# Function to find the top 3 largest acceptable groups of ingredients
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

    openai.api_key = openkagi
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
