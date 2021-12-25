import json
import requests
import pandas as pd
import logging
from ift6758.data.tidyData_adv import tidyData_adv
from ift6758.data.functions import pre_process

# import ift6758.data.tidyData_adv as tidyData_adv


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 8080, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing serving client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features
        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        # X = tidyData_adv(X)
        # X = pre_process(X)
        # X = X.drop(['isGoal'], axis=1)
        # X = X[self.features]

        r = requests.post(
        	f"{self.base_url}/predict", 
        	json = X.to_dict()
        )
        print(r)
        print(r.json())
        return pd.DataFrame(r.json())

    def logs(self) -> dict:
        """Get server logs"""
        #request server api
        r = requests.get(f"{self.base_url}/logs")
        #print(r)
        logs = r.json()
        print(logs)

        return logs

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 
        See more here:
            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        req = {
            'workspace': workspace,
            'model': model,
            'version': version,
        }

        r = requests.post(
        	f"{self.base_url}/download_registry_model", 
        	json=json.loads(json.dumps(req, indent = 4))
        )
        
        logs = r.json()
        print(logs)

        return logs

# abc = ServingClient()
# X = pd.DataFrame({'distance': [1,2,3], 'isGoal': [2,3,4]})
# #X = pd.DataFrame[{'Distance': [1,2,3,4,5,6,7,8,9,10], 'isGoal': [0,0,0,1,0,0,0,0,0,0]}]
# print(X.shape)
# abc.predict(X)