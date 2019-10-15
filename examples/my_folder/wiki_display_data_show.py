#入力と出力だけを取り出す
#display_modelの出力を保存したファイル＝＞入出力だけのout.txt

import re
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="input.txt", help="input_file_pass")
    parser.add_argument("--output_file", default="output.txt", help="output_file_pass")
    args=parser.parse_args()

    with open(args.input_file, "r") as inputF, open(args.output_file, "w") as outputF:
        inputTexts = inputF.read()
        outputList1 = re.findall(r"\[WizTeacher\]\:[^\[]*", inputTexts)
        outputList2 = re.findall(r"\[EndToEnd\]\:[^\~]*", inputTexts)
        #print(outputList[0]) ("[wizard_of_wikipedia]: Gardening\nI like Gardening, even when I've only been doing it for a short time.\n", '')
        lines = ""
        for line1, line2 in zip(outputList1,outputList2):
             lines += line1
             lines += line2
        outputF.write(lines)
