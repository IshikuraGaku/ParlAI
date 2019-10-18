from sklearn.model_selection import train_test_split
import json
import argparse
import pandas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data.json")
    args = parser.parse_args()

    input_txt = open(args.input_file).read()
    input_pandas = pandas.read_json(input_txt)
    
    train, test = train_test_split(input_pandas, test_size=0.2, shuffle=True, random_state=2434)
    valid, test = train_test_split(test, test_size=0.01, shuffle=True, random_state=221)

    train.to_json("train.json", orient="records")
    valid.to_json("valid_random_split.json", orient="records")
    test.to_json("test_random_split.json", orient="records")


if __name__ == "__main__":
    main()


