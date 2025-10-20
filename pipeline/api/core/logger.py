import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),  
        logging.StreamHandler()               
    ]
)

logger = logging.getLogger("api_logger")
