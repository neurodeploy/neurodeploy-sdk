import sys
import os
import json
from collections.abc import Callable
import inspect
import requests

DOMAIN_NAME = "neurodeploy"

null_function = lambda: None
creds: dict = {}


def login(username: str = ""):
    global creds
    from getpass import getpass

    _username = username if username else input("Username: ")
    password: str = getpass("Password: ")
    response = requests.post(
        url=f"https://user-api.{DOMAIN_NAME}.com/sessions",
        headers={"username": _username, "password": password},
    )

    tmp = response.json()
    if all(["token" in tmp, "expiration" in tmp, "error_message" not in tmp]):
        creds = tmp
        print("Successfully logged in")
    else:
        print("Failed to logged in")


def get_token() -> str:
    if not creds:
        raise Exception("Missing session token")
    return creds["token"]


def save_model(model) -> str:
    """Save a tensorflow model locally as /tmp/model.h5"""
    # model: tf.keras.models.Sequential
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
    preprocessing: Callable = null_function,
    lib: str = "tensorflow",
    filetype: str = "h5",
    is_public: bool = False,
    token: str = "",
):
    """
    Deploy an ML model.

    model:
        The ML model to be deployed.
        Note: Currently, only Tensorflow models are accepted.

    preprocessing: Callable (optional)
        A preprocessing function.
        If provided, then any input to the deployed model will first be passed to this preprocessing function.
        Note: The function must be named `preprocess`.

    lib: str (optional)
        The model type.
        Note: Currently, only Tensorflow models are accepted.

    filetype: str (optional)
        The file type you wish for the provided model to be stored in.
        Note: Currently, only the H5 file type is accepted.

    is_public: bool
        Whether or not you want the deployed model to be publicly accessible.
        If set to false, then the end user must provide a valid API key along with any execution request.
        API keys can be created via the function `create_api_key`.

    token: str
        Either the user's `access_key` or the `token` from logging in.
        If this function is called without either first logging in or providing a valid `access_key`,
        an exception will be raised.
    """
    if not token:
        token = get_token()
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


def predict(
    username: str,
    name: str,
    payload: dict,
    api_key: str = "",
) -> requests.Response:
    """
    Make a prediction using the deployed inference endpoint.

    username: str
        Name of the user whose account contains the model.

    name: str
        Name of the ML model to make inference with.

    payload: dict
        A dict of the input to be passed to the inference endpoint

    api_key: str (optional)
        If the inference endpoint is public, then this field is not used.
        If the inference endpoint is private, then this is the `api key`
        that provides access to the model
    """
    return requests.post(
        url=f"https://api.{DOMAIN_NAME}.com/{username}/{name}",
        headers={"api-key": api_key} if api_key else None,
        data=json.dumps(payload),
    )


def list_models(token: str = "") -> list[dict]:
    """
    List all models.

    token: str
        Either the user's `access_key` or the `token` from logging in.
        If this function is called without either first logging in or providing a valid `access_key`,
        an exception will be raised.
    """
    if not token:
        token = get_token()

    http_response = requests.get(
        url=f"https://user-api.{DOMAIN_NAME}.com/ml-models",
        headers={"Authorization": f"Bearer {token}"},
    )
    x = http_response.json()
    return x.get("models", [])


def list_credentials(token: str = "") -> list[dict]:
    """
    List all valid credentials.

    token: str
        Either the user's `access_key` or the `token` from logging in.
        If this function is called without either first logging in or providing a valid `access_key`,
        an exception will be raised.
    """
    if not token:
        token = get_token()

    http_response = requests.get(
        url=f"https://user-api.{DOMAIN_NAME}.com/credentials",
        headers={"Authorization": f"Bearer {token}"},
    )
    x = http_response.json()
    return x.get("creds", [])


def list_api_keys(model_name: str = "", token: str = "") -> list[dict]:
    """
    List API keys.

    model_name: str (optional)
        If specified, will return all API keys specific to the `model_name`.
        If unspecified, will return all API keys in the account.

    token: str
        Either the user's `access_key` or the `token` from logging in.
        If this function is called without either first logging in or providing a valid `access_key`,
        an exception will be raised.
    """
    if not token:
        token = get_token()

    http_response = requests.get(
        url=f"https://user-api.{DOMAIN_NAME}.com/api-keys",
        params={"model_name": model_name} if model_name else {},
        headers={"Authorization": f"Bearer {token}"},
    )
    x = http_response.json()
    return x.get("creds", [])


def create_credential(
    name: str,
    description: str,
    token: str = "",
) -> dict:
    """
    Create a new credential (access token).

    name: str
        Name you want to assign to this credential

    description: str
        A description of the purpose of this credential.

    token: str
        Either the user's `access_key` or the `token` from logging in.
        If this function is called without either first logging in or providing a valid `access_key`,
        an exception will be raised.
    """
    if not token:
        token = get_token()

    http_response = requests.post(
        url=f"https://user-api.{DOMAIN_NAME}.com/credentials",
        headers={
            "Authorization": f"Bearer {token}",
            "credentials_name": name,
            "description": description,
        },
    )
    return http_response.json()


def create_api_key(
    model_name: str = "",
    description: str = "",
    expires_after: int = 0,
    token: str = "",
) -> dict:
    """
    Create a new API key.

    model_name: str (optional)
        If specified, then the resulting API key will only grant access to the specified model.
        If unspecified, then the resulting API key will grant access to all models.

    description: str (optional)
        A description of the purpose of the API key.

    expires_after: int (optional)
        The number of minutes that the API key should remain valid for.
        If unspecified, then the resulting API key will remain valid until it is deleted.

    token: str
        Either the user's `access_key` or the `token` from logging in.
        If this function is called without either first logging in or providing a valid `access_key`,
        an exception will be raised.
    """
    if not token:
        token = get_token()

    params = {}
    if model_name:
        params["model_name"] = model_name
    if description:
        params["description"] = description
    if expires_after > 0:
        params["expires_after"] = expires_after

    http_response = requests.post(
        url=f"https://user-api.{DOMAIN_NAME}.com/api-keys",
        params=params,
        headers={"Authorization": f"Bearer {token}"},
    )
    return http_response.json()
