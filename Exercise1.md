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


