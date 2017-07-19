'''
http://www.statmt.org/wmt14/translation-task.html

News Crawl: articles from 2011	(EN) 784 MB  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.fr.shuffled.gz
News Crawl: articles from 2013	(EN) 1.1 GB  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz
'''

import os
import itertools
import keras_spell


def read_and_count(filename):
    filtered_lines = open(filename).read().decode('utf-8').split("\n")
    print "{0} - {1:,} rows.".format(filename, len(filtered_lines))


if __name__ == '__main__':

    year_list = [2007, 2011, 2013]
    files_list = ["clean", "filtered", "split", "split2", "split3"]

    for y, f in list(itertools.product(year_list, files_list)):
        NEWS_FILE_NAME = os.path.join(keras_spell.DATA_FILES_FULL_PATH, "news.{0}.en.{1}".format(y, f))
        read_and_count(NEWS_FILE_NAME)


