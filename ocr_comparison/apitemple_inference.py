import os
import base64
import requests


def get_text(img):
    return requests.get(
        url=f'https://api.apilayer.com/image_to_text/url?url={img}',
        headers={
            'apikey': 'YOUR API KEY'
        }
    ).json()


def image_to_data_url(path):
    return f"data:image/{path.split('.')[-1]};base64,{base64.b64encode(open(path, 'rb').read()).decode('utf-8')}"


def main():
    dataset_path = '30langs_200lines_fragmented'


if __name__ == '__main__':
    main()
