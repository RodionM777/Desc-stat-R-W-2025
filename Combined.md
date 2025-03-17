# Exercise 1. - Getting and Knowing your Data

This time we are going to pull data directly from the internet.
Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out [Occupation Exercises Video Tutorial](https://www.youtube.com/watch?v=W8AB5s-L3Rw&list=PLgJhDSE2ZLxaY_DigHeiIDC1cD09rXgJv&index=4) to watch a data scientist go through the exercises

### Step 1. Import the necessary libraries!


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user). 


```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
df = pd.read_csv(url, sep='|')
```

### Step 3. Assign it to a variable called users and use the 'user_id' as index


```python
users = df
users.set_index('user_id', inplace=True)
```

### Step 4. See the first 25 entries


```python
users.head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>85711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>F</td>
      <td>other</td>
      <td>94043</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>M</td>
      <td>writer</td>
      <td>32067</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>43537</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>F</td>
      <td>other</td>
      <td>15213</td>
    </tr>
    <tr>
      <th>6</th>
      <td>42</td>
      <td>M</td>
      <td>executive</td>
      <td>98101</td>
    </tr>
    <tr>
      <th>7</th>
      <td>57</td>
      <td>M</td>
      <td>administrator</td>
      <td>91344</td>
    </tr>
    <tr>
      <th>8</th>
      <td>36</td>
      <td>M</td>
      <td>administrator</td>
      <td>05201</td>
    </tr>
    <tr>
      <th>9</th>
      <td>29</td>
      <td>M</td>
      <td>student</td>
      <td>01002</td>
    </tr>
    <tr>
      <th>10</th>
      <td>53</td>
      <td>M</td>
      <td>lawyer</td>
      <td>90703</td>
    </tr>
    <tr>
      <th>11</th>
      <td>39</td>
      <td>F</td>
      <td>other</td>
      <td>30329</td>
    </tr>
    <tr>
      <th>12</th>
      <td>28</td>
      <td>F</td>
      <td>other</td>
      <td>06405</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>M</td>
      <td>educator</td>
      <td>29206</td>
    </tr>
    <tr>
      <th>14</th>
      <td>45</td>
      <td>M</td>
      <td>scientist</td>
      <td>55106</td>
    </tr>
    <tr>
      <th>15</th>
      <td>49</td>
      <td>F</td>
      <td>educator</td>
      <td>97301</td>
    </tr>
    <tr>
      <th>16</th>
      <td>21</td>
      <td>M</td>
      <td>entertainment</td>
      <td>10309</td>
    </tr>
    <tr>
      <th>17</th>
      <td>30</td>
      <td>M</td>
      <td>programmer</td>
      <td>06355</td>
    </tr>
    <tr>
      <th>18</th>
      <td>35</td>
      <td>F</td>
      <td>other</td>
      <td>37212</td>
    </tr>
    <tr>
      <th>19</th>
      <td>40</td>
      <td>M</td>
      <td>librarian</td>
      <td>02138</td>
    </tr>
    <tr>
      <th>20</th>
      <td>42</td>
      <td>F</td>
      <td>homemaker</td>
      <td>95660</td>
    </tr>
    <tr>
      <th>21</th>
      <td>26</td>
      <td>M</td>
      <td>writer</td>
      <td>30068</td>
    </tr>
    <tr>
      <th>22</th>
      <td>25</td>
      <td>M</td>
      <td>writer</td>
      <td>40206</td>
    </tr>
    <tr>
      <th>23</th>
      <td>30</td>
      <td>F</td>
      <td>artist</td>
      <td>48197</td>
    </tr>
    <tr>
      <th>24</th>
      <td>21</td>
      <td>F</td>
      <td>artist</td>
      <td>94533</td>
    </tr>
    <tr>
      <th>25</th>
      <td>39</td>
      <td>M</td>
      <td>engineer</td>
      <td>55107</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. See the last 10 entries


```python
users.tail(10)
```

### Step 6. What is the number of observations in the dataset?


```python
users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 943 entries, 1 to 943
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   age         943 non-null    int64 
     1   gender      943 non-null    object
     2   occupation  943 non-null    object
     3   zip_code    943 non-null    object
    dtypes: int64(1), object(3)
    memory usage: 36.8+ KB
    

### Step 7. What is the number of columns in the dataset?


```python
num_columns = df.shape[1]
print(num_columns)
```

    4
    

### Step 8. Print the name of all the columns.


```python
print(df.columns)
```

    Index(['age', 'gender', 'occupation', 'zip_code'], dtype='object')
    

### Step 9. How is the dataset indexed?


```python
users.index
```




    Index([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
           ...
           934, 935, 936, 937, 938, 939, 940, 941, 942, 943],
          dtype='int64', name='user_id', length=943)



### Step 10. What is the data type of each column?


```python
users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 943 entries, 1 to 943
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   age         943 non-null    int64 
     1   gender      943 non-null    object
     2   occupation  943 non-null    object
     3   zip_code    943 non-null    object
    dtypes: int64(1), object(3)
    memory usage: 36.8+ KB
    

### Step 11. Print only the occupation column


```python
users.occupation.head()
```




    user_id
    1    technician
    2         other
    3        writer
    4    technician
    5         other
    Name: occupation, dtype: object



### Step 12. How many different occupations are in this dataset?


```python
users.occupation.nunique()
```




    21



### Step 13. What is the most frequent occupation?


```python
users.occupation.value_counts()
```




    occupation
    student          196
    other            105
    educator          95
    administrator     79
    engineer          67
    programmer        66
    librarian         51
    writer            45
    executive         32
    scientist         31
    artist            28
    technician        27
    marketing         26
    entertainment     18
    healthcare        16
    retired           14
    lawyer            12
    salesman          12
    none               9
    homemaker          7
    doctor             7
    Name: count, dtype: int64



### Step 14. Summarize the DataFrame.


```python
users.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>943.000000</td>
      <td>943</td>
      <td>943</td>
      <td>943</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>2</td>
      <td>21</td>
      <td>795</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>M</td>
      <td>student</td>
      <td>55414</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>670</td>
      <td>196</td>
      <td>9</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>34.051962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.192740</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>31.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>73.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Step 15. Summarize all the columns


```python
print(df.describe()) 
```

                  age
    count  943.000000
    mean    34.051962
    std     12.192740
    min      7.000000
    25%     25.000000
    50%     31.000000
    75%     43.000000
    max     73.000000
    

### Step 16. Summarize only the occupation column


```python
print(df['occupation'].value_counts())
print(df['occupation'].describe())
```

    occupation
    student          196
    other            105
    educator          95
    administrator     79
    engineer          67
    programmer        66
    librarian         51
    writer            45
    executive         32
    scientist         31
    artist            28
    technician        27
    marketing         26
    entertainment     18
    healthcare        16
    retired           14
    lawyer            12
    salesman          12
    none               9
    homemaker          7
    doctor             7
    Name: count, dtype: int64
    count         943
    unique         21
    top       student
    freq          196
    Name: occupation, dtype: object
    

### Step 17. What is the mean age of users?


```python
users.age.mean()
```




    np.float64(34.05196182396607)



### Step 18. What is the age with least occurrence?


```python
users.age.value_counts(ascending=True)[:5]
```




    age
    73    1
    10    1
    66    1
    7     1
    11    1
    Name: count, dtype: int64




```python
pd.cut(users.age, 10).value_counts(ascending=True)[:5]
```




    age
    (6.934, 13.6]     8
    (66.4, 73.0]      8
    (59.8, 66.4]     23
    (53.2, 59.8]     36
    (40.0, 46.6]     94
    Name: count, dtype: int64


# Exercise 2. - Filtering and Sorting Data

Check out [Euro 12 Exercises Video Tutorial](https://youtu.be/iqk5d48Qisg) to watch a data scientist go through the exercises

This time we are going to pull data directly from the internet.

### Step 1. Import the necessary libraries


```python
import pandas as pd 
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv). 


```python
url = 'https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv' 
```

### Step 3. Assign it to a variable called euro12.


```python
euro12 = pd.read_csv(url, sep='|')
```

### Step 4. Select only the Goal column.


```python
goals = euro12["Goals"]
print(goals)
```

    0      4
    1      4
    2      4
    3      5
    4      3
    5     10
    6      5
    7      6
    8      2
    9      2
    10     6
    11     1
    12     5
    13    12
    14     5
    15     2
    Name: Goals, dtype: int64
    

### Step 5. How many team participated in the Euro2012?


```python
um_teams = euro12["Team"].nunique()
print(f"Number of teams that participated in Euro 2012: {num_teams}")
```

    Number of teams that participated in Euro 2012: 16
    

### Step 6. What is the number of columns in the dataset?


```python
num_columns = len(euro12.columns)
print(f"Number of columns in the dataset: {num_columns}")
```

    Number of columns in the dataset: 35
    

### Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline


```python
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
print(discipline.head())  
```

                 Team  Yellow Cards  Red Cards
    0         Croatia             9          0
    1  Czech Republic             7          0
    2         Denmark             4          0
    3         England             5          0
    4          France             6          0
    

### Step 8. Sort the teams by Red Cards, then to Yellow Cards


```python
discipline_sorted = discipline.sort_values(by=['Red Cards', 'Yellow Cards'], ascending=False)
print(discipline_sorted)
```

                       Team  Yellow Cards  Red Cards
    6                Greece             9          1
    9                Poland             7          1
    11  Republic of Ireland             6          1
    7                 Italy            16          0
    10             Portugal            12          0
    13                Spain            11          0
    0               Croatia             9          0
    1        Czech Republic             7          0
    14               Sweden             7          0
    4                France             6          0
    12               Russia             6          0
    3               England             5          0
    8           Netherlands             5          0
    15              Ukraine             5          0
    2               Denmark             4          0
    5               Germany             4          0
    

### Step 9. Calculate the mean Yellow Cards given per Team


```python
mean_yellow_cards = discipline["Yellow Cards"].mean()
print(mean_yellow_cards)
```

    7.4375
    

### Step 10. Filter teams that scored more than 6 goals


```python
teams_more_than_6_goals = euro12[euro12["Goals"] > 6]
print(teams_more_than_6_goals)
```

           Team  Goals  Shots on target  Shots off target Shooting Accuracy  \
    5   Germany     10               32                32             47.8%   
    13    Spain     12               42                33             55.9%   
    
       % Goals-to-shots  Total shots (inc. Blocked)  Hit Woodwork  Penalty goals  \
    5             15.6%                          80             2              1   
    13            16.0%                         100             0              1   
    
        Penalties not scored  ...  Saves made  Saves-to-shots ratio  Fouls Won  \
    5                      0  ...          10                 62.6%         63   
    13                     0  ...          15                 93.8%        102   
    
       Fouls Conceded  Offsides  Yellow Cards  Red Cards  Subs on  Subs off  \
    5              49        12             4          0       15        15   
    13             83        19            11          0       17        17   
    
        Players Used  
    5             17  
    13            18  
    
    [2 rows x 35 columns]
    

### Step 11. Select the teams that start with G


```python
teams_starting_with_G = euro12[euro12["Team"].str.startswith("G")]
print(teams_starting_with_G)
```

          Team  Goals  Shots on target  Shots off target Shooting Accuracy  \
    5  Germany     10               32                32             47.8%   
    6   Greece      5                8                18             30.7%   
    
      % Goals-to-shots  Total shots (inc. Blocked)  Hit Woodwork  Penalty goals  \
    5            15.6%                          80             2              1   
    6            19.2%                          32             1              1   
    
       Penalties not scored  ...  Saves made  Saves-to-shots ratio  Fouls Won  \
    5                     0  ...          10                 62.6%         63   
    6                     1  ...          13                 65.1%         67   
    
      Fouls Conceded  Offsides  Yellow Cards  Red Cards  Subs on  Subs off  \
    5             49        12             4          0       15        15   
    6             48        12             9          1       12        12   
    
       Players Used  
    5            17  
    6            20  
    
    [2 rows x 35 columns]
    

### Step 12. Select the first 7 columns


```python
first_7_columns = euro12.iloc[:, :7]
print(first_7_columns)
```

                       Team  Goals  Shots on target  Shots off target  \
    0               Croatia      4               13                12   
    1        Czech Republic      4               13                18   
    2               Denmark      4               10                10   
    3               England      5               11                18   
    4                France      3               22                24   
    5               Germany     10               32                32   
    6                Greece      5                8                18   
    7                 Italy      6               34                45   
    8           Netherlands      2               12                36   
    9                Poland      2               15                23   
    10             Portugal      6               22                42   
    11  Republic of Ireland      1                7                12   
    12               Russia      5                9                31   
    13                Spain     12               42                33   
    14               Sweden      5               17                19   
    15              Ukraine      2                7                26   
    
       Shooting Accuracy % Goals-to-shots  Total shots (inc. Blocked)  
    0              51.9%            16.0%                          32  
    1              41.9%            12.9%                          39  
    2              50.0%            20.0%                          27  
    3              50.0%            17.2%                          40  
    4              37.9%             6.5%                          65  
    5              47.8%            15.6%                          80  
    6              30.7%            19.2%                          32  
    7              43.0%             7.5%                         110  
    8              25.0%             4.1%                          60  
    9              39.4%             5.2%                          48  
    10             34.3%             9.3%                          82  
    11             36.8%             5.2%                          28  
    12             22.5%            12.5%                          59  
    13             55.9%            16.0%                         100  
    14             47.2%            13.8%                          39  
    15             21.2%             6.0%                          38  
    

### Step 13. Select all columns except the last 3.


```python
all_except_last_3 = euro12.iloc[:, :-3]
print(all_except_last_3)
```

                       Team  Goals  Shots on target  Shots off target  \
    0               Croatia      4               13                12   
    1        Czech Republic      4               13                18   
    2               Denmark      4               10                10   
    3               England      5               11                18   
    4                France      3               22                24   
    5               Germany     10               32                32   
    6                Greece      5                8                18   
    7                 Italy      6               34                45   
    8           Netherlands      2               12                36   
    9                Poland      2               15                23   
    10             Portugal      6               22                42   
    11  Republic of Ireland      1                7                12   
    12               Russia      5                9                31   
    13                Spain     12               42                33   
    14               Sweden      5               17                19   
    15              Ukraine      2                7                26   
    
       Shooting Accuracy % Goals-to-shots  Total shots (inc. Blocked)  \
    0              51.9%            16.0%                          32   
    1              41.9%            12.9%                          39   
    2              50.0%            20.0%                          27   
    3              50.0%            17.2%                          40   
    4              37.9%             6.5%                          65   
    5              47.8%            15.6%                          80   
    6              30.7%            19.2%                          32   
    7              43.0%             7.5%                         110   
    8              25.0%             4.1%                          60   
    9              39.4%             5.2%                          48   
    10             34.3%             9.3%                          82   
    11             36.8%             5.2%                          28   
    12             22.5%            12.5%                          59   
    13             55.9%            16.0%                         100   
    14             47.2%            13.8%                          39   
    15             21.2%             6.0%                          38   
    
        Hit Woodwork  Penalty goals  Penalties not scored  ...  Clean Sheets  \
    0              0              0                     0  ...             0   
    1              0              0                     0  ...             1   
    2              1              0                     0  ...             1   
    3              0              0                     0  ...             2   
    4              1              0                     0  ...             1   
    5              2              1                     0  ...             1   
    6              1              1                     1  ...             1   
    7              2              0                     0  ...             2   
    8              2              0                     0  ...             0   
    9              0              0                     0  ...             0   
    10             6              0                     0  ...             2   
    11             0              0                     0  ...             0   
    12             2              0                     0  ...             0   
    13             0              1                     0  ...             5   
    14             3              0                     0  ...             1   
    15             0              0                     0  ...             0   
    
        Blocks  Goals conceded Saves made  Saves-to-shots ratio  Fouls Won  \
    0       10               3         13                 81.3%         41   
    1       10               6          9                 60.1%         53   
    2       10               5         10                 66.7%         25   
    3       29               3         22                 88.1%         43   
    4        7               5          6                 54.6%         36   
    5       11               6         10                 62.6%         63   
    6       23               7         13                 65.1%         67   
    7       18               7         20                 74.1%        101   
    8        9               5         12                 70.6%         35   
    9        8               3          6                 66.7%         48   
    10      11               4         10                 71.5%         73   
    11      23               9         17                 65.4%         43   
    12       8               3         10                 77.0%         34   
    13       8               1         15                 93.8%        102   
    14      12               5          8                 61.6%         35   
    15       4               4         13                 76.5%         48   
    
        Fouls Conceded  Offsides  Yellow Cards  Red Cards  
    0               62         2             9          0  
    1               73         8             7          0  
    2               38         8             4          0  
    3               45         6             5          0  
    4               51         5             6          0  
    5               49        12             4          0  
    6               48        12             9          1  
    7               89        16            16          0  
    8               30         3             5          0  
    9               56         3             7          1  
    10              90        10            12          0  
    11              51        11             6          1  
    12              43         4             6          0  
    13              83        19            11          0  
    14              51         7             7          0  
    15              31         4             5          0  
    
    [16 rows x 32 columns]
    

### Step 14. Present only the Shooting Accuracy from England, Italy and Russia


```python
shooting_accuracy = euro12[euro12["Team"].isin(["England", "Italy", "Russia"])]["Shooting Accuracy"]
print(shooting_accuracy)
```

    3     50.0%
    7     43.0%
    12    22.5%
    Name: Shooting Accuracy, dtype: object
    
# Exercise 3. - GroupBy

### Introduction:

GroupBy can be summarized as Split-Apply-Combine.

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out this [Diagram](http://i.imgur.com/yjNkiwL.png)  

Check out [Alcohol Consumption Exercises Video Tutorial](https://youtu.be/az67CMdmS6s) to watch a data scientist go through the exercises


### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv). 


```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv'
df = pd.read_csv(url, sep=',')
```

### Step 3. Assign it to a variable called drinks.


```python
drinks= df
drinks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>AS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>AF</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>AF</td>
    </tr>
  </tbody>
</table>
</div>



### Step 4. Which continent drinks more beer on average?


```python
drinks.groupby('continent').agg({'beer_servings': 'mean'}).sort_values(by='beer_servings', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EU</th>
      <td>193.777778</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>175.083333</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>89.687500</td>
    </tr>
    <tr>
      <th>AF</th>
      <td>61.471698</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>37.045455</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. For each continent print the statistics for wine consumption.


```python
drinks.groupby('continent')['wine_servings'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>53.0</td>
      <td>16.264151</td>
      <td>38.846419</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>13.00</td>
      <td>233.0</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>44.0</td>
      <td>9.068182</td>
      <td>21.667034</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.00</td>
      <td>123.0</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>45.0</td>
      <td>142.222222</td>
      <td>97.421738</td>
      <td>0.0</td>
      <td>59.0</td>
      <td>128.0</td>
      <td>195.00</td>
      <td>370.0</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>16.0</td>
      <td>35.625000</td>
      <td>64.555790</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.5</td>
      <td>23.25</td>
      <td>212.0</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>12.0</td>
      <td>62.416667</td>
      <td>88.620189</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>98.50</td>
      <td>221.0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6. Print the mean alcohol consumption per continent for every column


```python
drinks.groupby('continent').mean(numeric_only=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>61.471698</td>
      <td>16.339623</td>
      <td>16.264151</td>
      <td>3.007547</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>37.045455</td>
      <td>60.840909</td>
      <td>9.068182</td>
      <td>2.170455</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>193.777778</td>
      <td>132.555556</td>
      <td>142.222222</td>
      <td>8.617778</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>89.687500</td>
      <td>58.437500</td>
      <td>35.625000</td>
      <td>3.381250</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>175.083333</td>
      <td>114.750000</td>
      <td>62.416667</td>
      <td>6.308333</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. Print the median alcohol consumption per continent for every column


```python
median_alcohol = drinks.groupby('continent').median(numeric_only=True)
print(median_alcohol)
```

               beer_servings  spirit_servings  wine_servings  \
    continent                                                  
    AF                  32.0              3.0            2.0   
    AS                  17.5             16.0            1.0   
    EU                 219.0            122.0          128.0   
    OC                  52.5             37.0            8.5   
    SA                 162.5            108.5           12.0   
    
               total_litres_of_pure_alcohol  
    continent                                
    AF                                 2.30  
    AS                                 1.20  
    EU                                10.00  
    OC                                 1.75  
    SA                                 6.85  
    

### Step 8. Print the mean, min and max values for spirit consumption.
#### This time output a DataFrame


```python
spirit_stats = drinks['spirit_servings'].agg(['mean', 'min', 'max'])
spirit_stats_df = spirit_stats.to_frame().T
print(spirit_stats_df)
```

                          mean  min    max
    spirit_servings  80.994819  0.0  438.0
    
