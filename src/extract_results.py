# Simple script to extract time and re

import os
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
from natsort import natsorted

if __name__ == "__main__":

    results_files = [folder for folder in os.listdir('results') if folder.endswith('.txt')]

    results = []
    times = []
    names = []
    for item in natsorted(results_files):
        with open(os.path.join('results', item)) as file:
            lines = file.readlines()
            line_result = [line for line in lines if line.startswith('Best solution found had diversity')][0]
            line_time = [line for line in lines if line.startswith('Total time elapsed')][0]
            names.append(item)
            results.append(line_result.split()[-1])
            times.append(line_time.split()[-2])


    data = list(zip(names, results, times))

    df = pd.DataFrame(data, columns = ['Filename', 'Result', 'Time'])
    df.astype({'Filename': str, 'Result': float, 'Time': float})
    print(df.Result)
    # df.to_csv('my_results.csv', index=False)





