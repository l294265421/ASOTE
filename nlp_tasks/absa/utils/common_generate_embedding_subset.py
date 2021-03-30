import sys


if __name__ == u'__main__':
    args = sys.argv[1:]
    embedding_filepath = args[0]
    words_filepath = args[1]
    output_filepath = args[2]

    word_vector = {}
    with open(embedding_filepath, encoding='utf-8') as embedding_file:
        for line in embedding_file:
            parts = line.strip().split()
            word_vector[parts[0]] = ' '.join(parts[1:])
    with open(words_filepath, encoding='utf-8') as words_file, open(output_filepath, mode='w', encoding='utf-8') as out_file:
        for line in words_file:
            word = line.strip()
            if word in word_vector:
                vector = word_vector[word]
                out_file.write(word + ' ' + vector + '\n')
