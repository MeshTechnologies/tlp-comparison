import os
import json
import openai
from time import sleep


def get_embedding(text, engine, break_lines=False):
    if break_lines:
        text = text.replace('\n', ' ')

    return openai.Embedding.create(input=[text], engine=engine)['data'][0]['embedding']


def get_classification_embeddings(dataset_path, models):
    # create subdirectory for this dataset
    if not os.path.exists(os.path.join('openai_embeddings', 'output', 'classification')):
        os.mkdir(os.path.join('openai_embeddings', 'output', 'classification'))

    output_path = os.path.join('openai_embeddings', 'output', 'classification')

    for model in models:
        # create subdirectory for embeddings from specific model
        if not os.path.exists(os.path.join(output_path, model)):
            os.mkdir(os.path.join(output_path, model))

        for subset in ['train', 'valid', 'test']:
            # create subdirectories for train/valid/test data
            if not os.path.exists(os.path.join(output_path, model, subset)):
                os.mkdir(os.path.join(output_path, model, subset))

            for language in os.listdir(os.path.join(dataset_path, subset)):
                # do not use temporary files/directories
                if not language.startswith('.'):
                    # create subdirectories for every programming language
                    if not os.path.exists(os.path.join(output_path, model, subset, language)):
                        os.mkdir(os.path.join(output_path, model, subset, language))

                    for file in os.listdir(os.path.join(dataset_path, subset, language)):
                        # do not use temporary files/directories
                        if not file.startswith('.'):
                            # read text from dataset
                            text = open(os.path.join(dataset_path, subset, language, file), 'r').read()
                            print('Generating ' + os.path.join(output_path, model, subset, language, f'{file.split(".")[0]}.json'))
                            # generate embeddings
                            json.dump(
                                {'text': text, 'embeddings': get_embedding(text, model)},
                                open(os.path.join(output_path, model, subset, language, f'{file.split(".")[0]}.json'), 'w'),
                                indent=4
                            )
                            # rate limit
                            sleep(1.5)


def get_textvscode_embeddings(dataset_path, models):
    # create subdirectory for this dataset
    if not os.path.exists(os.path.join('openai_embeddings', 'output', 'textvscode')):
        os.mkdir(os.path.join('openai_embeddings', 'output', 'textvscode'))

    output_path = os.path.join('openai_embeddings', 'output', 'textvscode')

    for model in models:
        # create subdirectory for embeddings from specific model
        if not os.path.exists(os.path.join(output_path, model)):
            os.mkdir(os.path.join(output_path, model))

        for subset in ['train', 'valid', 'test']:
            # create subdirectories for train/valid/test data
            if not os.path.exists(os.path.join(output_path, model, subset)):
                os.mkdir(os.path.join(output_path, model, subset))

            for data_type in ['code', 'text']:
                # create subdirectories for both code and text
                if not os.path.exists(os.path.join(output_path, model, subset, data_type)):
                    os.mkdir(os.path.join(output_path, model, subset, data_type))

                for file in os.listdir(os.path.join(dataset_path, subset, data_type)):
                    # do not use temporary files/directories
                    if not file.startswith('.'):
                        # read text from dataset
                        text = open(os.path.join(dataset_path, subset, data_type, file), 'r').read()
                        # generate embeddings
                        if data_type == 'code':
                            embeddings = get_embedding(text, model)
                        else:
                            embeddings = get_embedding(text, model, break_lines=True)
                        print('Generating ' + os.path.join(output_path, model, subset, data_type, f'{file.split(".")[0]}.json'))
                        json.dump(
                            {'text': text, 'embeddings': embeddings},
                            open(os.path.join(output_path, model, subset, data_type, f'{file.split(".")[0]}.json'), 'w'),
                            indent=4
                        )
                        # rate limit
                        sleep(1.5)


def main():
    # load user key for openAI
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # create directory for output
    if not os.path.exists(os.path.join('openai_embeddings', 'output')):
        os.mkdir(os.path.join('openai_embeddings', 'output'))

    # programming language classification dataset
    get_classification_embeddings(
        dataset_path='56_lang_sampled_dataset_weak_cobol',
        models=[
            # 'text-similarity-ada-001',
            # 'text-similarity-babbage-001',
            # 'text-similarity-curie-001',
            # 'text-similarity-davinci-001',
            'code-search-ada-code-001',
            # 'code-search-ada-text-001',
            # 'code-search-babbage-code-001',
            # 'code-search-babbage-text-001'
        ]
    )

    # TextVsCode dataset
    # get_textvscode_embeddings(
    #     dataset_path='diverse_sample_v1_test',
    #     models=[
    #         'text-similarity-ada-001',
    #         'text-similarity-babbage-001',
    #         'text-similarity-curie-001',
    #         'text-similarity-davinci-001',
    #         'code-search-ada-code-001',
    #         'code-search-ada-text-001',
    #         'code-search-babbage-code-001',
    #         'code-search-babbage-text-001'
    #     ]
    # )


if __name__ == '__main__':
    main()
