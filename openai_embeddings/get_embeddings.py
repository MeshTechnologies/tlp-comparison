import openai
import argparse


def get_embedding(text, engine):
    if engine.startswith('text'):
        text = text.replace('\n', ' ')

    return openai.Embedding.create(input=[text], engine=engine)['data'][0]['embedding']


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    main(args)
