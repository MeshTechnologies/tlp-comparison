import base64
import requests
import argparse


def get_text(img):
    return requests.get(
        url=f'https://api.apilayer.com/image_to_text/url?url={img}',
        headers={
            'apikey': 'YOUR API KEY'
        }
    ).json()


def image_to_data_url(path):
    return f"data:image/{path.split('.')[-1]};base64,{base64.b64encode(open(path, 'rb').read()).decode('utf-8')}"


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    main(args)
