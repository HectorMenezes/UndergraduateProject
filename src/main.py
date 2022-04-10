import time

from src.handler import create_model, evaluate_estimators
from utils import argp, load_configuration


def main():
    """
    Main fuction called by the script.
    """
    start = time.time()
    args = argp.parse_args()
    configuration = load_configuration(args.configuration)
   
    
    
    for case in configuration.cases:
        evaluate_estimators(**case.dict())
        #create_model(**case.dict())
        

    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
