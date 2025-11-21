import pandas as pd
from pathlib import Path

RAW = Path('1_DatasetCharacteristics/raw_data')
OUT = Path('1_DatasetCharacteristics/processed_data')

# --- holiday helpers (copied from notebook) ---

def get_fixed_holidays(years):
    fixed = []
    for year in years:
        fixed.extend([
            pd.Timestamp(f'{year}-01-01'),
            pd.Timestamp(f'{year}-05-01'),
            pd.Timestamp(f'{year}-10-03'),
            pd.Timestamp(f'{year}-10-31'),
            pd.Timestamp(f'{year}-11-01'),
            pd.Timestamp(f'{year}-12-25'),
            pd.Timestamp(f'{year}-12-26'),
        ])
    return fixed


easter_dates = {
    2013: pd.Timestamp('2013-03-31'),
    2014: pd.Timestamp('2014-04-20'),
    2015: pd.Timestamp('2015-04-05'),
    2016: pd.Timestamp('2016-03-27'),
    2017: pd.Timestamp('2017-04-16'),
    2018: pd.Timestamp('2018-04-01'),
}


def get_easter_holidays(easter_dates):
    easter_holidays = []
    for year, easter in easter_dates.items():
        easter_holidays.append(easter - pd.Timedelta(days=2))
        easter_holidays.append(easter)
        easter_holidays.append(easter + pd.Timedelta(days=1))
        easter_holidays.append(easter + pd.Timedelta(days=39))
        easter_holidays.append(easter + pd.Timedelta(days=49))
        easter_holidays.append(easter + pd.Timedelta(days=50))
        easter_holidays.append(easter + pd.Timedelta(days=60))
    return easter_holidays

carnival_dates = {
    2013: pd.Timestamp('2013-02-12'),
    2014: pd.Timestamp('2014-03-04'),
    2015: pd.Timestamp('2015-02-17'),
    2016: pd.Timestamp('2016-02-09'),
    2017: pd.Timestamp('2017-02-28'),
    2018: pd.Timestamp('2018-02-13'),
}

years = [2013, 2014, 2015, 2016, 2017, 2018]

schleswig_holstein_holidays = sorted(list(set(
    get_fixed_holidays(years) + get_easter_holidays(easter_dates) + list(carnival_dates.values())
)))


def add_holiday_indicator(df, holidays, date_column='Datum'):
    df = df.copy()
    df['is_holiday'] = df[date_column].isin(holidays).astype(int)
    return df


def main():
    files = {
        'umsatz': RAW / 'umsatzdaten_gekuerzt.csv',
        'test': RAW / 'test.csv',
        'kiwo': RAW / 'kiwo.csv',
        'wetter': RAW / 'wetter.csv',
        'niederschlag': RAW / 'Niederschlag.csv',
    }

    for k, p in files.items():
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    print('Loading umsatz and test...')
    df_umsatz = pd.read_csv(files['umsatz'], parse_dates=['Datum'])
    df_test = pd.read_csv(files['test'], parse_dates=['Datum'])

    df_umsatz_combined = pd.concat([df_umsatz, df_test], ignore_index=True)
    print('Combined umsatz + test rows:', len(df_umsatz_combined))

    df_kiwo = pd.read_csv(files['kiwo'], parse_dates=['Datum'])
    df_wetter = pd.read_csv(files['wetter'], parse_dates=['Datum'])
    df_niederschlag = pd.read_csv(files['niederschlag'], parse_dates=['Datum'])

    df_umsatz_combined = df_umsatz_combined.set_index('Datum')
    df_kiwo = df_kiwo.set_index('Datum')
    df_wetter = df_wetter.set_index('Datum')
    df_niederschlag = df_niederschlag.set_index('Datum')

    other_dfs = [df_kiwo, df_wetter, df_niederschlag]
    final_df = df_umsatz_combined.join(other_dfs, how='outer')

    final_df = final_df.reset_index()
    final_df = final_df.sort_values(by='Datum')
    final_df = add_holiday_indicator(final_df, schleswig_holstein_holidays, date_column='Datum')

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / 'combined_data_outer.csv'
    final_df.to_csv(out, index=False)
    print('Wrote', out, 'rows:', len(final_df))


if __name__ == '__main__':
    main()
