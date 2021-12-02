import json
import os
from base64 import b64decode
from functools import lru_cache

import rsa


def verify_response(public_key: str, response: dict):
    if not response or not isinstance(response, dict):
        raise ValueError('Response is not a valid dictionary')
    signature = response.pop('signature', None)
    if not signature:
        raise ValueError('No signature found in response')

    pub_key = rsa.PublicKey.load_pkcs1(public_key.encode())
    try:
        rsa.verify(json.dumps(response, sort_keys=True).encode(),
                   b64decode(signature), pub_key)
    except rsa.VerificationError:
        raise ValueError('Signature Verification Failed')


@lru_cache()
def get_public_key():
    script_dir = os.path.dirname(__file__)
    with open(script_dir + '/public.pem', 'r') as f:
        return f.read()
