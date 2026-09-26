import uvicorn
import sys
import decouple
sys.path.insert(0, '..')
decouple.config = decouple.Config(decouple.RepositoryEnv('config/config.env'))


log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"

from src.app import app
uvicorn.run(app, host="0.0.0.0", port=8386, log_config=log_config)


