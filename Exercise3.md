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
    
