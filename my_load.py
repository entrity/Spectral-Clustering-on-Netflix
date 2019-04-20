import io, csv, pickle
from my_util import *

TITL_COL = 2
YEAR_COL = 1
MVID_COL = 0
USER_COL = 1
RATE_COL = 2
DATE_COL = 3
N_MOVIES = 17770
N_USERS  = 5907
# Computed N_USERS by `cat train.csv test.csv | cut -f2 -d, | sort | uniq | wc -l`

def load_movies(fpath='movie_titles.txt'):
  # Read csv
  with io.open(fpath, encoding='cp1252') as fin:
    reader = csv.reader(fin, delimiter=',')
    header = next(reader)
    _year = lambda row : int(row[YEAR_COL]) if row[YEAR_COL].isnumeric() else None
    data = [r for r in reader]
  N = len(data)
  assert N == N_MOVIES, N
  return data

def load_csv(fpath='train.csv', has_header=True, savepath=None):
  with open(fpath) as fin:
    reader = csv.reader(fin, delimiter=',')
    if has_header:
      header = next(reader)
    data = [( int(r[MVID_COL],10), int(r[USER_COL]), int(r[RATE_COL]), parsedate(r[DATE_COL]) ) for r in reader if r[0].isnumeric()]
  if savepath is not None:
    with open(savepath, 'wb') as fout: pickle.dump(data, fout)
  return data

# Load the test csv, which has '?' values instead of numbers
def load_test_csv(fpath='test.csv', has_header=False, savepath=None):
  with open(fpath) as fin:
    reader = csv.reader(fin, delimiter=',')
    if has_header:
      header = next(reader)
    data = [( int(r[MVID_COL],10), int(r[USER_COL],10), r[RATE_COL], parsedate(r[DATE_COL]) ) for r in reader if r[0].isnumeric()]
  if savepath is not None:
    with open(savepath, 'wb') as fout: pickle.dump(data, fout)
  return data

def construct_user_id_map(savepath='user-id-map.pkl', trainpath='train.csv', testpath='test.csv'):
  with open(trainpath) as fin:
    reader = csv.reader(fin, delimiter=',')
    train_ids = [row[USER_COL] for row in reader]
  with open(testpath) as fin:
    reader = csv.reader(fin, delimiter=',')
    test_ids = [row[USER_COL] for row in reader]
  user_ids = [int(x) for x in sorted(train_ids + test_ids) if x.isnumeric()]
  uidmap = {}
  cursor = 0
  for val in user_ids:
    if val not in uidmap:
      uidmap[val] = cursor
      cursor += 1
  with open(savepath, 'wb') as fout:
    pickle.dump(uidmap, fout)
  return uidmap

if __name__ == '__main__':
  load_csv('train.csv', True, 'trainset.pkl')
  load_test_csv('test.csv', False, 'testset.pkl')
