import time

from src.handler import create_model, evaluate_estimators, perform_cross_validation
from src.model import Model
from utils import argp, load_configuration


def main():
    """
    Main fuction called by the script.
    """
    args = argp.parse_args()
    configuration = load_configuration(args.configuration)
    cross_validation = args.cross_validation
    max_data_points = args.max_data

    if cross_validation != 0 and max_data_points != 0:
        for case in configuration.cases:
            model_dict = case.dict()
            model_dict.pop("supervised")
            model = Model(**model_dict)
            time_elapsed, mean = perform_cross_validation(model, cross_validation, max_data_points)

            print("Datapoints: " + str(max_data_points))
            print("Time: " + str(time_elapsed))
            print("Accuracy: " + str(mean))
    else:
        for case in configuration.cases:
            evaluate_estimators(**case.dict())
            #create_model(**case.dict())
            


if __name__ == "__main__":
    main()
