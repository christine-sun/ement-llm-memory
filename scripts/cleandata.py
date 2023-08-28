import re

def read_text(file_path):
    with open(file_path, "r") as file:
        return file.read()

def separate_sentences(text):
    text = text.replace('\n', ' ').replace('\r', '')
    return re.split(r'(?<!\d)(?<!vs)\.(?!\d)', text)

def write_text(file_path, text_list):
    with open(file_path, "w") as file:
        for line in text_list:
            stripped_line = line.strip()
            if stripped_line:
                if stripped_line[-1] == '.':
                    file.write("{}\n".format(stripped_line))
                else:
                    file.write("{}.{}\n".format(stripped_line, ' '))

if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"

    text_content = read_text(input_file)
    text_by_sentence = separate_sentences(text_content)
    write_text(output_file, text_by_sentence)