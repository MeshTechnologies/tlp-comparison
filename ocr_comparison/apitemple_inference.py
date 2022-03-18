import os
import requests


def get_text(img, key):
    payload = {}
    return requests.get(
        url=f'https://api.apilayer.com/image_to_text/url?url={img}',
        headers={
            'apikey': key
        },
        data=payload
    ).json()


def main():
    # load user key for APITemple
    key = os.getenv("TEMPLE_API_KEY")

    for image in os.listdir(os.path.join('01-manually_labelled_code_screenshots', 'ocr', 'evaluation', 'images')):
        if image.endswith('.png'):
            url = f'https://www.linkpicture.com/q/{image}'
            response = get_text(url, key)
            if 'all_text' in response.keys():
                print(f'Generating text for {image}...')
                open(os.path.join('output', image.split('.')[0] + '.txt'), 'w').write(response['all_text'])


if __name__ == '__main__':
    main()
