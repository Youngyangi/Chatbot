def get_pattern(path="./util/dialogue.txt"):
    dict = {}
    with open(path, 'r', encoding='utf8') as f:
        for x in f:
            query, ans = x.strip().split('\t')
            dict[query] = ans
    return dict

def get_answer(input, dict):
    return -1 if input not in dict else dict[input]