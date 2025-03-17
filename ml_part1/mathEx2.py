import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('ch2_scores_em.csv')

eng_scores = np.array(df['english'])[:10]
math_scores = np.array(df['mathematics'])[:10]
scores_df = pd.DataFrame({'english':eng_scores,
                          'mathematics':math_scores},
                         index = pd.Index(list('ABCDEFGHIJ'),name = 'student'))

print(scores_df)

summary_df = scores_df.copy()
summary_df['english_deviation'] = summary_df['english'] - summary_df['english'].mean()
summary_df['mathematics_deviation'] = summary_df['mathematics'] - summary_df['mathematics'].mean()
summary_df['product_of_deviation'] = summary_df['english_deviation'] * summary_df['mathematics_deviation']
print(summary_df)
print()
print(summary_df['product_of_deviation'].mean())
print(np.cov(eng_scores,math_scores,ddof=0))
print()

print(round((np.cov(eng_scores,math_scores,ddof=0)[0,1]) / (np.std(eng_scores)*(np.std(math_scores))),3))
print(np.corrcoef(eng_scores,math_scores))
print(scores_df.corr())

english_scores = np.array(df['english'])
math_scores = np.array(df['mathematics'])


poly_fit = np.polyfit(english_scores,math_scores,deg=1)
poly_1d = np.poly1d(poly_fit)
print(poly_fit)

xs = np.linspace(english_scores.min(),english_scores.max(),50)
ys = poly_1d(xs)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
ax.scatter(english_scores,math_scores)
ax.plot(xs,ys,color='gray',label=f'{poly_fit[0]:.2f}x + {poly_fit[1]:.2f}')
ax.legend(loc='best')
ax.set_xlabel('english')
ax.set_ylabel('mathematics')
plt.show()

