from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from gradio_client import Client

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
            key = key.strip().upper()
            value = value.strip()

            if key in ['INGREDIENTS', 'DIRECTIONS']:
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
