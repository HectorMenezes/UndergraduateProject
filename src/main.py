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

    for case in configuration.cases:
        model_args = {x: case.dict()[x] for x in case.dict() if x not in ["supervised", "predict"]}
        model = Model(**model_args)

        if cross_validation != 0 and max_data_points != 0:
            
            time_elapsed, mean = perform_cross_validation(model, cross_validation, max_data_points)

            print("Datapoints: " + str(max_data_points))
            print("Time: " + str(time_elapsed))
            print("Accuracy: " + str(mean))
        elif case.predict and max_data_points != 0:
            model = create_model(model, max_data_points)

            results = model.model.predict([line[:-1] for line in case.predict["data"]])
            counter = 0

            print("Expect\tGot")

            for index, expected in enumerate(case.predict["data"]):
                print(str(expected[-1]) + "\t" + str(results[index]))
                counter += expected[-1] ==results[index]
            print("Number of right predictions: " + str(counter))
            print("Rate: " + str(counter/len(results)))

            

if __name__ == "__main__":
    main()
