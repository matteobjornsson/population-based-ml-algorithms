import pandas as pd

def average_every_3():
    data_sets = ["soybean", "glass", "abalone","cancer","fire", "machine"] 

    for data_set in data_sets:
        filename = f"./PSO_tune/PSO_tune_{data_set}.csv"
        df = pd.read_csv(filename)
        l = []
        base = 0
        for i in range(len(df)):
            if i % 3 == 0: base = i
            av = (df['loss2'][base] + df['loss2'][base+1] + df['loss2'][base+2])/3
            l.append(av)
        df['average'] = l
        df.to_csv(filename, index=False)


def fix_soybean():
    omega = [.2, .5, .8 ]
    c1 = [.1, .5, .9, 5]
    c2 = [.1, .5, .9, 5]
    vmax = [1, 7, 15]
    pop_size = [10, 100, 1000]

    df = pd.read_csv('./PSO_tune/PSO_tune_soybean.csv')
    drop_rows = []
    for i in range(len(df)):
        v_omega = float(df['omega'][i])
        v_c1 = float(df['c1'][i])
        v_c2 = float(df['c2'][i])
        v_vmax = float(df['vmax'][i])
        v_pop = float(df['pop_size'][i])
        if (v_omega not in omega) or (v_c1 not in c1) or (v_c2 not in c2) or (v_vmax not in vmax) or (v_pop not in pop_size):
            drop_rows.append(i)
    df.drop(drop_rows, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('./PSO_tune/PSO_tune_soybean2.csv', index=False)