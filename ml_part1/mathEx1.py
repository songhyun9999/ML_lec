import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('ch2_scores_em.csv',index_col = 'student number')
df.info()
print(df.head())
print()

scores = np.array(df['english'])[:10]
print(scores)

scores_df = pd.DataFrame({'score':scores},
                         index = pd.Index(list('ABCEDFGHIJ'),name = 'student'))
print(scores_df)
print()

sorted_scores = np.sort(scores)
print(sorted_scores)
print()

n = len(sorted_scores)
if n%2==0:
    m0 = sorted_scores[n//2 -1]
    m1 = sorted_scores[n//2]
    median = (m0+m1)/2
else:
    median = sorted_scores[(n+1)//2-1]

print(median)
print(np.median(sorted_scores))
print(scores_df.median())

mean = np.mean(scores)
deviation = scores - mean
print(deviation)

another_scores = [50,60,58,54,51,56,57,53,67,59]
another_mean = np.mean(another_scores)
print(another_mean)
another_deviation = another_scores - another_mean
print(another_deviation)

print(np.mean(deviation))
summary_df = scores_df.copy()
summary_df['deviation'] = deviation
print(summary_df)
print()
print(np.mean(deviation**2))
print(np.var(scores))
print(scores_df.var(ddof=0))
print()

print(np.sqrt(np.mean(deviation**2)))
print()

z = (scores - np.mean(scores))/ np.std(scores)
print(z)

print(round(np.mean(z)), round(np.std(z),3))
