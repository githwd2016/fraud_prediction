def merge(file_list, years):
    assert len(file_list) == len(years)
    result = []
    for file, year in zip(file_list, years):
        with open(file, 'r') as f:
            for line in f.readlines():
                i, j, k = map(float, line.split())
                result.append(f'{int(i)} {year} {k}\n')
    return result


if __name__ == '__main__':
    files = [f'result/rnn/ICD_dataset_240918_1700/{year}_2_300_0.9_200.txt' for year in range(2013, 2022)]
    r = merge(files, [year for year in range(2013, 2022)])
    with open('result/rnn/ICD_dataset_240918_1700/2_300_0.9_200.txt', 'w') as f:
        f.writelines(r)
