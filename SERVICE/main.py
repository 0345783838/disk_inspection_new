import uvicorn
import sys
import decouple
sys.path.insert(0, '..')
decouple.config = decouple.Config(decouple.RepositoryEnv('config/config.env'))


log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"

from src.app import app
# Keep the production default; isolated smoke tests may choose a free port.
uvicorn.run(app, host="0.0.0.0", port=decouple.config('DISK_API_PORT', default=8386, cast=int), log_config=log_config)


