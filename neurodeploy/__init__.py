import sys
import os
from collections.abc import Callable
import inspect
import requests

DEV = "playingwithml"
PROD = "neurodeploy"
DOMAIN_NAME = DEV


null_function = lambda: None


def set_dev():
    DOMAIN_NAME = DEV


def set_prod():
    DOMAIN_NAME = PROD


def login(username: str = "") -> dict:
    from getpass import getpass

    _username = username if username else input("Username: ")
    password: str = getpass("Password: ")
    response = requests.post(
        url=f"https://user-api.{DOMAIN_NAME}.com/sessions",
        headers={"username": _username, "password": password},
    )
    return response.json()


def save_model(model) -> str:
    """Save a tensorflow model locally as /tmp/model.h5"""
    # model : tf.keras.models.Sequential
    if "tensorflow" not in sys.modules:
        raise Exception("tensorflow not imported")
    path = "/tmp/model.h5"
    model.save(path)

    return path


def save_preprocessing(preprocessing: Callable) -> str:
    """Save preprocessing function locally as /tmp/preprocessing.py"""
    if not preprocessing:
        return ""

    source = inspect.getsource(preprocessing)
    if not source.startswith("def preprocess("):
        raise Exception("The preprocessing function must be named 'preprocess'.")

    path = "/tmp/preprocessing.py"
    with open(path, "w") as f:
        f.write(source)

    return path


def upload_with_presigned_url(presigned: dict, filepath: str) -> requests.Response:
    """Upload file using the presigned url in the dict `presigned`"""
    return requests.post(
        presigned["url"], data=presigned["fields"], files={"file": open(filepath, "rb")}
    )


def print_success_or_failure(name: str, response: requests.Response):
    print(f'{name}: {"success" if response.status_code == 204 else "failure"}')


def deploy(
    name: str,
    model,
    token: str,
    preprocessing: Callable = null_function,
    lib: str = "tensorflow",
    filetype: str = "h5",
    is_public: bool = False,
):
    # model: : tf.keras.models.Sequential

    # save model and preprocessing if exists
    filepath = save_model(model)
    preprocessing_path = save_preprocessing(preprocessing)

    # Get upload presigned urls
    http_response = requests.put(
        url=f"https://user-api.{DOMAIN_NAME}.com/ml-models/{name}",
        params={
            "lib": lib,
            "filetype": filetype,
            "has_preprocessing": preprocessing != null_function,
            "is_public": is_public,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    x = http_response.json()

    # upload model
    response = upload_with_presigned_url(x["model"], filepath)
    print_success_or_failure("Upload model", response)

    # upload preprocessing function
    if x["preprocessing"]:
        response = upload_with_presigned_url(x["preprocessing"], preprocessing_path)
        print_success_or_failure("Upload preprocessing", response)
    elif preprocessing:
        raise Exception("No presigned url to upload preprocessing function with")

    # remove model and preprocessing function files
    os.remove(filepath)
    os.remove(preprocessing_path)
