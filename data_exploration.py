'''
http://www.statmt.org/wmt14/translation-task.html


News Crawl: articles from 2011	(EN) 784 MB  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.fr.shuffled.gz
News Crawl: articles from 2013	(EN) 1.1 GB  http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz

news.2007.en.clean - 3,782,549 rows.
news.2007.en.filtered - 3,780,034 rows.
news.2007.en.split - 13,745,293 rows.
news.2007.en.split2 - 708,160 rows.
news.2007.en.split3 - 245,469 rows.
news.2011.en.clean - 15,437,675 rows.
news.2011.en.filtered - 15,420,582 rows.
news.2011.en.split - 54,897,272 rows.
news.2011.en.split2 - 3,186,453 rows.
news.2011.en.split3 - 900,391 rows.
news.2013.en.clean - 21,688,363 rows.
news.2013.en.filtered - 21,641,708 rows.
news.2013.en.split - 76,415,056 rows.
news.2013.en.split2 - 4,304,937 rows.
news.2013.en.split3 - 1,248,269 rows.

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


