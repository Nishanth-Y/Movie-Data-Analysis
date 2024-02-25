---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.6
  nbformat: 4
  nbformat_minor: 2
---

<div class="cell markdown">

## Movie Data Analysis

</div>

<div class="cell markdown">

#### Your goal is to Analyze the data and find the Highest correlated features in the data using Python and listed libraries

#### Libraries used are:

#### - Pandas → for holding data and structure the data

#### - Seaborn & Matplotlib → for visualizing data features

#### - Numpay → for data manipulation and data structure needs

</div>

<div class="cell code" execution_count="2">

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)
```

</div>

<div class="cell markdown">

#### Read Data from File

</div>

<div class="cell code" execution_count="3">

```python
#read data
df = pd.read_csv('./movies.csv')
df.head()
```

<div class="output execute_result" execution_count="3">

                                                 name rating      genre  year  \
    0                                     The Shining      R      Drama  1980
    1                                 The Blue Lagoon      R  Adventure  1980
    2  Star Wars: Episode V - The Empire Strikes Back     PG     Action  1980
    3                                       Airplane!     PG     Comedy  1980
    4                                      Caddyshack      R     Comedy  1980

                            released  score      votes         director  \
    0  June 13, 1980 (United States)    8.4   927000.0  Stanley Kubrick
    1   July 2, 1980 (United States)    5.8    65000.0   Randal Kleiser
    2  June 20, 1980 (United States)    8.7  1200000.0   Irvin Kershner
    3   July 2, 1980 (United States)    7.7   221000.0     Jim Abrahams
    4  July 25, 1980 (United States)    7.3   108000.0     Harold Ramis

                        writer            star         country      budget  \
    0             Stephen King  Jack Nicholson  United Kingdom  19000000.0
    1  Henry De Vere Stacpoole  Brooke Shields   United States   4500000.0
    2           Leigh Brackett     Mark Hamill   United States  18000000.0
    3             Jim Abrahams     Robert Hays   United States   3500000.0
    4       Brian Doyle-Murray     Chevy Chase   United States   6000000.0

             gross             company  runtime
    0   46998772.0        Warner Bros.    146.0
    1   58853106.0   Columbia Pictures    104.0
    2  538375067.0           Lucasfilm    124.0
    3   83453539.0  Paramount Pictures     88.0
    4   39846344.0      Orion Pictures     98.0

</div>

</div>

<div class="cell markdown">

## Data Cleaning

#### Finding missing value

</div>

<div class="cell code" execution_count="4">

```python
# data cleaning
# - find missing values
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print(f'{col} - {pct_missing}%')

```

<div class="output stream stdout">

    name - 0.0%
    rating - 0.010041731872717789%
    genre - 0.0%
    year - 0.0%
    released - 0.0002608242044861763%
    score - 0.0003912363067292645%
    votes - 0.0003912363067292645%
    director - 0.0%
    writer - 0.0003912363067292645%
    star - 0.00013041210224308815%
    country - 0.0003912363067292645%
    budget - 0.2831246739697444%
    gross - 0.02464788732394366%
    company - 0.002217005738132499%
    runtime - 0.0005216484089723526%

</div>

</div>

<div class="cell markdown">

#### Replace nan values to median for budget and gross

</div>

<div class="cell code" execution_count="5">

```python
# - Replace nan values to median for budget and gross
print('\n',df['budget'].describe())
print(f'\nMedian of budget {df["budget"].median():,.0f}')
df['budget'].fillna(df['budget'].median(), inplace=True)
df['gross'].fillna(df['gross'].median(), inplace=True)


for col in ['budget', 'gross']:
    pct_missing = np.mean(df[col].isnull())
    print(f'{col} - {pct_missing}%')
```

<div class="output stream stdout">

     count    5.497000e+03
    mean     3.558988e+07
    std      4.145730e+07
    min      3.000000e+03
    25%      1.000000e+07
    50%      2.050000e+07
    75%      4.500000e+07
    max      3.560000e+08
    Name: budget, dtype: float64

    Median of budget 20,500,000
    budget - 0.0%
    gross - 0.0%

</div>

</div>

<div class="cell markdown">

#### Change data type of budget and gross

</div>

<div class="cell code" execution_count="6">

```python
# - Change data type of budget and gross
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
df.dtypes
```

<div class="output execute_result" execution_count="6">

    name         object
    rating       object
    genre        object
    year          int64
    released     object
    score       float64
    votes       float64
    director     object
    writer       object
    star         object
    country      object
    budget        int64
    gross         int64
    company      object
    runtime     float64
    dtype: object

</div>

</div>

<div class="cell code" execution_count="7">

```python
df.head(10)
```

<div class="output execute_result" execution_count="7">

                                                 name rating      genre  year  \
    0                                     The Shining      R      Drama  1980
    1                                 The Blue Lagoon      R  Adventure  1980
    2  Star Wars: Episode V - The Empire Strikes Back     PG     Action  1980
    3                                       Airplane!     PG     Comedy  1980
    4                                      Caddyshack      R     Comedy  1980
    5                                 Friday the 13th      R     Horror  1980
    6                              The Blues Brothers      R     Action  1980
    7                                     Raging Bull      R  Biography  1980
    8                                     Superman II     PG     Action  1980
    9                                 The Long Riders      R  Biography  1980

                                released  score      votes            director  \
    0      June 13, 1980 (United States)    8.4   927000.0     Stanley Kubrick
    1       July 2, 1980 (United States)    5.8    65000.0      Randal Kleiser
    2      June 20, 1980 (United States)    8.7  1200000.0      Irvin Kershner
    3       July 2, 1980 (United States)    7.7   221000.0        Jim Abrahams
    4      July 25, 1980 (United States)    7.3   108000.0        Harold Ramis
    5        May 9, 1980 (United States)    6.4   123000.0  Sean S. Cunningham
    6      June 20, 1980 (United States)    7.9   188000.0         John Landis
    7  December 19, 1980 (United States)    8.2   330000.0     Martin Scorsese
    8      June 19, 1981 (United States)    6.8   101000.0      Richard Lester
    9       May 16, 1980 (United States)    7.0    10000.0         Walter Hill

                        writer             star         country    budget  \
    0             Stephen King   Jack Nicholson  United Kingdom  19000000
    1  Henry De Vere Stacpoole   Brooke Shields   United States   4500000
    2           Leigh Brackett      Mark Hamill   United States  18000000
    3             Jim Abrahams      Robert Hays   United States   3500000
    4       Brian Doyle-Murray      Chevy Chase   United States   6000000
    5            Victor Miller     Betsy Palmer   United States    550000
    6              Dan Aykroyd     John Belushi   United States  27000000
    7             Jake LaMotta   Robert De Niro   United States  18000000
    8             Jerry Siegel     Gene Hackman   United States  54000000
    9              Bill Bryden  David Carradine   United States  10000000

           gross                       company  runtime
    0   46998772                  Warner Bros.    146.0
    1   58853106             Columbia Pictures    104.0
    2  538375067                     Lucasfilm    124.0
    3   83453539            Paramount Pictures     88.0
    4   39846344                Orion Pictures     98.0
    5   39754601            Paramount Pictures     95.0
    6  115229890            Universal Pictures    133.0
    7   23402427  Chartoff-Winkler Productions    129.0
    8  108185706                Dovemead Films    127.0
    9   15795189                United Artists    100.0

</div>

</div>

<div class="cell code" execution_count="8">

```python
import re
x = df['released'].astype(str).str[:]
y = []
for i in x:
    m = re.search('\d{4}', i)
    if m:
        y.append(m.group())
    else:
        y.append(None)
print(y[:10])
```

<div class="output stream stdout">

    ['1980', '1980', '1980', '1980', '1980', '1980', '1980', '1980', '1981', '1980']

</div>

</div>

<div class="cell code" execution_count="9">

```python
df['yearcorrect'] = y
```

</div>

<div class="cell code" execution_count="10">

```python
df.head()
```

<div class="output execute_result" execution_count="10">

                                                 name rating      genre  year  \
    0                                     The Shining      R      Drama  1980
    1                                 The Blue Lagoon      R  Adventure  1980
    2  Star Wars: Episode V - The Empire Strikes Back     PG     Action  1980
    3                                       Airplane!     PG     Comedy  1980
    4                                      Caddyshack      R     Comedy  1980

                            released  score      votes         director  \
    0  June 13, 1980 (United States)    8.4   927000.0  Stanley Kubrick
    1   July 2, 1980 (United States)    5.8    65000.0   Randal Kleiser
    2  June 20, 1980 (United States)    8.7  1200000.0   Irvin Kershner
    3   July 2, 1980 (United States)    7.7   221000.0     Jim Abrahams
    4  July 25, 1980 (United States)    7.3   108000.0     Harold Ramis

                        writer            star         country    budget  \
    0             Stephen King  Jack Nicholson  United Kingdom  19000000
    1  Henry De Vere Stacpoole  Brooke Shields   United States   4500000
    2           Leigh Brackett     Mark Hamill   United States  18000000
    3             Jim Abrahams     Robert Hays   United States   3500000
    4       Brian Doyle-Murray     Chevy Chase   United States   6000000

           gross             company  runtime yearcorrect
    0   46998772        Warner Bros.    146.0        1980
    1   58853106   Columbia Pictures    104.0        1980
    2  538375067           Lucasfilm    124.0        1980
    3   83453539  Paramount Pictures     88.0        1980
    4   39846344      Orion Pictures     98.0        1980

</div>

</div>

<div class="cell markdown">

#### Order by Gross column

</div>

<div class="cell code" execution_count="11">

```python
# Order by Gross column
df.sort_values(by=['gross'], inplace=True, ascending=False)
```

</div>

<div class="cell markdown">

#### Drop any duplicates

</div>

<div class="cell code" execution_count="12">

```python
# Drop any duplicates

df['company'].drop_duplicates().sort_values(ascending=False)
```

<div class="output execute_result" execution_count="12">

    7129                                thefyzz
    5664                            micro_scope
    6412               iDeal Partners Film Fund
    4007                               i5 Films
    6793                             i am OTHER
                           ...
    3748                     1+2 Seisaku Iinkai
    3024                        .406 Production
    7525    "Weathering With You" Film Partners
    4345        "DIA" Productions GmbH & Co. KG
    7657                                    NaN
    Name: company, Length: 2386, dtype: object

</div>

</div>

<div class="cell code" execution_count="13">

```python
df.head()
```

<div class="output execute_result" execution_count="13">

                                                name rating   genre  year  \
    5445                                      Avatar  PG-13  Action  2009
    7445                           Avengers: Endgame  PG-13  Action  2019
    3045                                     Titanic  PG-13   Drama  1997
    6663  Star Wars: Episode VII - The Force Awakens  PG-13  Action  2015
    7244                      Avengers: Infinity War  PG-13  Action  2018

                                   released  score      votes       director  \
    5445  December 18, 2009 (United States)    7.8  1100000.0  James Cameron
    7445     April 26, 2019 (United States)    8.4   903000.0  Anthony Russo
    3045  December 19, 1997 (United States)    7.8  1100000.0  James Cameron
    6663  December 18, 2015 (United States)    7.8   876000.0    J.J. Abrams
    7244     April 27, 2018 (United States)    8.4   897000.0  Anthony Russo

                      writer               star        country     budget  \
    5445       James Cameron    Sam Worthington  United States  237000000
    7445  Christopher Markus  Robert Downey Jr.  United States  356000000
    3045       James Cameron  Leonardo DiCaprio  United States  200000000
    6663     Lawrence Kasdan       Daisy Ridley  United States  245000000
    7244  Christopher Markus  Robert Downey Jr.  United States  321000000

               gross                company  runtime yearcorrect
    5445  2847246203  Twentieth Century Fox    162.0        2009
    7445  2797501328         Marvel Studios    181.0        2019
    3045  2201647264  Twentieth Century Fox    194.0        1997
    6663  2069521700              Lucasfilm    138.0        2015
    7244  2048359754         Marvel Studios    149.0        2018

</div>

</div>

<div class="cell markdown">

#### Assuming The Most correlated fields with gross are

#### - Budget v/s Gross

#### - Company v/s Gross

#### To find or visualize the relation between budget and gross is by scatter plot

</div>

<div class="cell code" execution_count="14">

```python
# Assuming The Most correlated fields with gross are
#  - Budget v/s Gross
#  - Company v/s Gross

# To find or visualize the relation between budget and gross is by scatter plot

plt.scatter(x=df['budget'], y=df['gross'])

# Lets give some information
plt.title('Budget V/S Gross')
plt.xlabel('Budget of the Movie')
plt.ylabel('Gross of the Movie')

plt.show()
```

<div class="output display_data">

![](db1ec967b7f1d42a5eeb827173fbabc645cc4c36.png)

</div>

</div>

<div class="cell markdown">

</div>

<div class="cell markdown">

#### Visualize using seaborn

</div>

<div class="cell code" execution_count="15">

```python
# Visualize using seaborn

sns.regplot(x=df['budget'], y=df['gross'], data=df, scatter_kws={"color": "red"}, line_kws={"color": "blue"})
```

<div class="output execute_result" execution_count="15">

    <Axes: xlabel='budget', ylabel='gross'>

</div>

<div class="output display_data">

![](e2b65fb1dcc03907eacea8e317fd456cdaab4532.png)

</div>

</div>

<div class="cell markdown">

### Find correlation for only numeric types

### Methods of correlation are :

- pearson
- kendall
- spearman

</div>

<div class="cell code" execution_count="16">

```python
# Find correlation for only numeric types
# Methods of correlation are :
# - pearson
# - kendall
# - spearman


print(df.corr(numeric_only=True))
print('We found that budget and gross has high correlation')
```

<div class="output stream stdout">

                 year     score     votes    budget     gross   runtime
    year     1.000000  0.097995  0.222945  0.291690  0.259504  0.120811
    score    0.097995  1.000000  0.409182  0.061979  0.185583  0.399451
    votes    0.222945  0.409182  1.000000  0.460932  0.632103  0.309212
    budget   0.291690  0.061979  0.460932  1.000000  0.745881  0.273363
    gross    0.259504  0.185583  0.632103  0.745881  1.000000  0.244360
    runtime  0.120811  0.399451  0.309212  0.273363  0.244360  1.000000
    We found that budget and gross has high correlation

</div>

</div>

<div class="cell code" execution_count="17">

```python
correlation_matrix = df.corr(method='pearson', numeric_only=True)
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix for Numeric Features")
plt.xlabel('Movie Fields')
plt.ylabel('Movie Fields')
plt.show()
```

<div class="output display_data">

![](2e6fdbb446b05a5b71c0a9db0bad7cc8e132b9e5.png)

</div>

</div>

<div class="cell markdown">

### Convert Object type feature like Company Features to numeric or categoric type

</div>

<div class="cell code" execution_count="18">

```python
# Convert Object type feature like Company Features to numeric or categoric type

df_normalized = df.copy()

for col in df_normalized.columns:
    if df_normalized[col].dtype == 'object':
        df_normalized[col] = df_normalized[col].astype('category')
        df_normalized[col] = df_normalized[col].cat.codes

df_normalized
```

<div class="output execute_result" execution_count="18">

          name  rating  genre  year  released  score      votes  director  writer  \
    5445   533       5      0  2009       696    7.8  1100000.0      1155    1778
    7445   535       5      0  2019       183    8.4   903000.0       162     743
    3045  6896       5      6  1997       704    7.8  1100000.0      1155    1778
    6663  5144       5      0  2015       698    7.8   876000.0      1125    2550
    7244   536       5      0  2018       192    8.4   897000.0       162     743
    ...    ...     ...    ...   ...       ...    ...        ...       ...     ...
    3818  3360       6      4  2000      1119    6.5     5200.0       730    1123
    7625  6720      -1      4  2019      1149    5.7      320.0      2546    2565
    7580  4664       3      5  2019      1835    5.2      735.0      1445    2203
    2417  3406      -1      6  1993        85    7.3     5100.0        33    1718
    3203  6990       5      4  1997      2811    5.7     5800.0       961     229

          star  country     budget       gross  company  runtime  yearcorrect
    5445  2334       55  237000000  2847246203     2253    162.0           29
    7445  2241       55  356000000  2797501328     1606    181.0           39
    3045  1595       55  200000000  2201647264     2253    194.0           17
    6663   524       55  245000000  2069521700     1540    138.0           35
    7244  2241       55  321000000  2048359754     1606    149.0           38
    ...    ...      ...        ...         ...      ...      ...          ...
    3818  2319       54   20500000        1400      477    103.0           21
    7625  1915       55   20500000         790     2308    104.0           39
    7580  2278       55   20500000         682     1992     93.0           40
    2417  2563       27   11900000         596      796    134.0           13
    3203  2758       55   15000000         309      821     85.0           17

    [7668 rows x 16 columns]

</div>

</div>

<div class="cell code" execution_count="19">

```python
df_normalized.sort_values(by=['gross'], inplace=True, ascending=False)

print(df_normalized.corr())
print('We found that budget and gross has high correlation')
```

<div class="output stream stdout">

                     name    rating     genre      year  released     score  \
    name         1.000000 -0.008069  0.016355  0.011453 -0.011311  0.017097
    rating      -0.008069  1.000000  0.072423  0.008779  0.016613 -0.001314
    genre        0.016355  0.072423  1.000000 -0.081261  0.029822  0.027965
    year         0.011453  0.008779 -0.081261  1.000000 -0.000695  0.097995
    released    -0.011311  0.016613  0.029822 -0.000695  1.000000  0.042788
    score        0.017097 -0.001314  0.027965  0.097995  0.042788  1.000000
    votes        0.013088  0.033225 -0.145307  0.222945  0.016097  0.409182
    director     0.009079  0.019483 -0.015258 -0.020795 -0.001478  0.009559
    writer       0.009081 -0.005921  0.006567 -0.008656 -0.002404  0.019416
    star         0.006472  0.013405 -0.005477 -0.027242  0.015777 -0.001609
    country     -0.010737  0.081244 -0.037615 -0.070938 -0.020427 -0.133348
    budget       0.020921 -0.108776 -0.328484  0.291690  0.011120  0.061979
    gross        0.006601 -0.097213 -0.233385  0.259504  0.000806  0.185583
    company      0.009211 -0.032943 -0.071067 -0.010431 -0.010474  0.001030
    runtime      0.010392  0.062145 -0.052711  0.120811  0.000868  0.399451
    yearcorrect  0.010225  0.006403 -0.078210  0.996397 -0.003775  0.106295

                    votes  director    writer      star   country    budget  \
    name         0.013088  0.009079  0.009081  0.006472 -0.010737  0.020921
    rating       0.033225  0.019483 -0.005921  0.013405  0.081244 -0.108776
    genre       -0.145307 -0.015258  0.006567 -0.005477 -0.037615 -0.328484
    year         0.222945 -0.020795 -0.008656 -0.027242 -0.070938  0.291690
    released     0.016097 -0.001478 -0.002404  0.015777 -0.020427  0.011120
    score        0.409182  0.009559  0.019416 -0.001609 -0.133348  0.061979
    votes        1.000000  0.000260  0.000892 -0.019282  0.073625  0.460932
    director     0.000260  1.000000  0.299067  0.039234  0.017490 -0.003584
    writer       0.000892  0.299067  1.000000  0.027245  0.015343 -0.030641
    star        -0.019282  0.039234  0.027245  1.000000 -0.012998 -0.018534
    country      0.073625  0.017490  0.015343 -0.012998  1.000000  0.082334
    budget       0.460932 -0.003584 -0.030641 -0.018534  0.082334  1.000000
    gross        0.632103 -0.014758 -0.023064 -0.001529  0.093994  0.745881
    company      0.133204  0.004404  0.005646  0.012442  0.095548  0.167250
    runtime      0.309212  0.017624 -0.003511  0.010174 -0.078412  0.273363
    yearcorrect  0.218289 -0.020385 -0.008391 -0.027606 -0.079009  0.284099

                    gross   company   runtime  yearcorrect
    name         0.006601  0.009211  0.010392     0.010225
    rating      -0.097213 -0.032943  0.062145     0.006403
    genre       -0.233385 -0.071067 -0.052711    -0.078210
    year         0.259504 -0.010431  0.120811     0.996397
    released     0.000806 -0.010474  0.000868    -0.003775
    score        0.185583  0.001030  0.399451     0.106295
    votes        0.632103  0.133204  0.309212     0.218289
    director    -0.014758  0.004404  0.017624    -0.020385
    writer      -0.023064  0.005646 -0.003511    -0.008391
    star        -0.001529  0.012442  0.010174    -0.027606
    country      0.093994  0.095548 -0.078412    -0.079009
    budget       0.745881  0.167250  0.273363     0.284099
    gross        1.000000  0.155786  0.244360     0.252749
    company      0.155786  1.000000  0.034402    -0.014144
    runtime      0.244360  0.034402  1.000000     0.120636
    yearcorrect  0.252749 -0.014144  0.120636     1.000000
    We found that budget and gross has high correlation

</div>

</div>

<div class="cell code" execution_count="20">

```python
correlation_matrix = df_normalized.corr(method='pearson', numeric_only=True)
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix for Numeric Features")
plt.xlabel('Movie Fields')
plt.ylabel('Movie Fields')
plt.show()
```

<div class="output display_data">

![](f95e3fa104a90bb2298d13143f580a38bf075c27.png)

</div>

</div>

<div class="cell markdown">

### Lets find the correlation between features which have above 0.5 value

</div>

<div class="cell code" execution_count="21">

```python
# Lets find the correlation between features which have above 0.5 value


sorted_corr = df_normalized.corr().unstack().sort_values()
high_corr = sorted_corr[(sorted_corr) > 0.5]
high_corr = high_corr[(high_corr) < 0.9]
```

</div>

<div class="cell code" execution_count="22">

```python
high_corr
```

<div class="output execute_result" execution_count="22">

    gross   votes     0.632103
    votes   gross     0.632103
    gross   budget    0.745881
    budget  gross     0.745881
    dtype: float64

</div>

</div>

<div class="cell markdown">

#### We found that there is high correlation between 'Gross' feature and 'Votes' feature as well as with 'Gross' feature and 'Budget' features

</div>
