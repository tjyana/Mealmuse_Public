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

    def format_recipe(recipe_text):
        for text in recipe:
            sections = text.split("\n")
            for section in sections:
                section = section.strip()
                if section.startswith("title:"):
                    section = section.replace("title:", "")
                    headline = "TITLE"
                elif section.startswith("ingredients:"):
                    section = section.replace("ingredients:", "")
                    headline = "INGREDIENTS"
                elif section.startswith("directions:"):
                    section = section.replace("directions:", "")
                    headline = "DIRECTIONS"

                if headline == "TITLE":
                    print(f"[{headline}]: {section.strip().capitalize()}")
                else:
                    section_info = [f"  - {i+1}: {info.strip().capitalize()}" for i, info in enumerate(section.split("--"))]
                    print(f"[{headline}]:")
                    print("\n".join(section_info))

            print("-" * 130)

    recipe = generation_function(ingredients)
    recipe = format_recipe(recipe)
    return recipe


def image_generator(recipe):
# Making API request using gradio_client
    client = Client("https://playgroundai-playground-v2-5.hf.space/--replicas/o9oxl/")
    result = client.predict(
            recipe, # str  in 'Promp'Textbox component
            " ",    # str  in 'Negative prompt' Textbox component
            False,  # bool  in 'Use negative prompt' Checkbox component
            820,  # float (numeric value between 0 and 2147483647) in 'Seed' Slider component
            1024,   # float (numeric value between 256 and 1536) in 'Width' Slider component
            1024,   # float (numeric value between 256 and 1536) in 'Height' Slider component
            3,  # float (numeric value between 0.1 and 20) in 'Guidance Scale' Slider component
            True,   # bool  in 'Randomize seed' Checkbox component
            api_name="/run"
    )

    # Extracting image path from the result
    image_path = result[0][0]['image']

    # Load and display the image
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axis
    plt.show()
    print(result)
    return img
