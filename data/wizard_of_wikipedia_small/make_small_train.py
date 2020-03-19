from sklearn.model_selection import train_test_split
import json
import argparse
import pandas
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="train_raw.json")
    args = parser.parse_args()

    input_txt = open(args.input_file).read()
    input_pandas = pandas.read_json(input_txt)

    train_split1, train_split2 = train_test_split(input_pandas, test_size=0.5, shuffle=True, random_state=2434)

    train_split1.to_json("train.json", orient="records")
    train_split2.to_json("train2.json", orient="records")




def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data.json")
    args = parser.parse_args()

    input_txt = open(args.input_file).read()
    input_pandas = pandas.read_json(input_txt)
    
    """
    train, test = train_test_split(input_pandas, test_size=0.2, shuffle=True, random_state=2434)
    valid, test = train_test_split(test, test_size=0.01, shuffle=True, random_state=221)

    train.to_json("train.json", orient="records")
    valid.to_json("valid_random_split.json", orient="records")
    test.to_json("test_random_split.json", orient="records")
    """

    train, topic = train_test_split(input_pandas, test_size=0.2, shuffle=True, random_state=2434)
    train, random = train_test_split(train, test_size=0.25, shuffle=True, random_state=149)

    chosen_topic_list = [] #topicの中のトピック
    for chosen in topic["chosen_topic"]:
        chosen_topic_list.append(chosen)
        """
        chosen_topic
        persona
        wizard_eval
        dialog
        chosen_topic_passage
        """

    in_topic_list = [] #trainとtopicで重なってるやつ
    not_topic_list = []
    for tmp_topic in train["chosen_topic"]:
        if tmp_topic in chosen_topic_list:
            in_topic_list.append(tmp_topic)
        else:
            not_topic_list.append(tmp_topic)
    
    #print(len(in_topic_list))
    #print(len(not_topic_list))
    
    #print(train[train["chosen_topic"].isin(["Blue"])].info) #134
    print(len(topic[topic["chosen_topic"].isin(["Blue"])])) #38
    
    flag = [not i for i in train["chosen_topic"].isin(chosen_topic_list)]
    #print(in_topic_list)
    train = train[flag]
    #print(train.info)

    """
    print(in_topic_list)
    print(train)

    train.to_json("train.json", orient="records")
    topic_test, topic_valid = train_test_split(topic, test_size=0.5, shuffle=True, random_state=2434)
    random_test, random_valid = train_test_split(random, test_size=0.5, shuffle=True, random_state=2434)
    
    topic_valid.to_json("valid_topic_split.json", orient="records")
    topic_test.to_json("test_topic_split.json", orient="records")
    random_valid.to_json("valid_random_split.json", orient="records")
    random_test.to_json("test_random_split.json", orient="records")
"""


    

    #print(chosen_topic_list)

if __name__ == "__main__":
    main()


