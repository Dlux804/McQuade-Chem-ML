# File for random commands


# to change the working directory inside of a context manager
# it will revert to previous directory when finished with loop.
from contextlib import contextmanager
import os
import tqdm
import concurrent.futures as cf


def foo(self, string):
    print('\n', self.algorithm)
    print(string)
    return string


@contextmanager
def cd(newdir):
    """
    Change the working directory inside of a context manager.
    It will revert to previous directory when finished with loop.
    """
    prevdir = os.getcwd()
    # print('Previous PATH:', prevdir)
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        # print('Switching back to previous PATH:', prevdir)
        os.chdir(prevdir)

# Example usage
'''
os.chdir('/home')

with cd('/tmp'):
    # ...
    raise Exception("There's no place like home.")
# Directory is now back to '/home'.
'''


def __aw__(df_column, function, **props):
    """
    Wrapper function for parallel apply. Actual runs the pandas.apply on an individual CPU.
    """

    new_df_column = df_column.apply(function, **props)
    return new_df_column


def parallel_apply(df_column, function, number_of_workers, loading_bars, **props):
    """
    This function will run pandas.apply in parallel depending on the number of CPUS the user specifies.
    """

    steps = len(df_column) / number_of_workers
    mid_dfs = []
    for x in range(number_of_workers):
        if x == number_of_workers - 1:
            mid_dfs.append(df_column.iloc[int(steps * x):])
        else:
            mid_dfs.append(df_column.iloc[int(steps * x):int(steps * (x + 1))])

    main_df = None
    with cf.ProcessPoolExecutor(max_workers=number_of_workers) as executor:

        results = []
        for mid_df in mid_dfs:
            results.append(executor.submit(__aw__, mid_df, function, **props))

        if loading_bars:
            for f in tqdm.tqdm(cf.as_completed(results), total=number_of_workers):
                if main_df is None:
                    main_df = f.result()
                else:
                    main_df = main_df.append(f.result())
        else:
            for f in cf.as_completed(results):
                if main_df is None:
                    main_df = f.result()
                else:
                    main_df = main_df.append(f.result())
    return main_df
