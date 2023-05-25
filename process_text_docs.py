import glob
import pandas as pd

DATA_DIR = './text_docs/autoapi/abacusai'


def process_text_docs():
    data = []
    for file in glob.glob(f'{DATA_DIR}/**/*.txt'):
        with open(file, 'r') as f:
            data.append({'content': f.read(), 'filename': file})
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)


if __name__ == '__main__':
    process_text_docs()

