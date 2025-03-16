Learn how to summarize the columns available in an R data frame. 
  You will also learn how to chain operations together with the
  pipe operator, and how to compute grouped summaries using.

## Welcome!

Hey there! Ready for the first lesson?

The dfply package makes it possible to do R's dplyr-style data manipulation with pipes in python on pandas DataFrames.

[dfply website here](https://github.com/kieferk/dfply)

[![](https://www.rforecology.com/pipes_image0.png "https://github.com/kieferk/dfply"){width="600"}](https://github.com/kieferk/dfply)


```python
import pandas as pd
import seaborn as sns
cars = sns.load_dataset('mpg')
from dfply import *
cars >> head(3)
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth satellite</td>
    </tr>
  </tbody>
</table>
</div>



### Exercise 1.

Select the columns 'mpg' and 'horsepower' from the cars DataFrame.


```python
selected_columns = cars[['mpg', 'horsepower']]
print(selected_columns)
```

          mpg  horsepower
    0    18.0       130.0
    1    15.0       165.0
    2    18.0       150.0
    3    16.0       150.0
    4    17.0       140.0
    ..    ...         ...
    393  27.0        86.0
    394  44.0        52.0
    395  32.0        84.0
    396  28.0        79.0
    397  31.0        82.0
    
    [398 rows x 2 columns]
    

## Selecting and dropping

There are two functions for selection, inverse of each other: select and drop. The select and drop functions accept string labels, integer positions, and/or symbolically represented column names (X.column). They also accept symbolic "selection filter" functions, which will be covered shortly.

### Exercise 2.

Select the columns 'mpg' and 'horsepower' from the cars DataFrame using the drop function.


```python
selected_columns = cars >> drop(X.columns.difference(["mpg","horsepower"]))
print(selected_columns)
```

          mpg  horsepower
    0    18.0       130.0
    1    15.0       165.0
    2    18.0       150.0
    3    16.0       150.0
    4    17.0       140.0
    ..    ...         ...
    393  27.0        86.0
    394  44.0        52.0
    395  32.0        84.0
    396  28.0        79.0
    397  31.0        82.0
    
    [398 rows x 2 columns]
    

## Selection using \~

One particularly nice thing about dplyr's selection functions is that you can drop columns inside of a select statement by putting a subtraction sign in front, like so: ... %>% select(-col). The same can be done in dfply, but instead of the subtraction operator you use the tilde ~.

### Exercise 3.

Select all columns except 'model_year', and 'name' from the cars DataFrame.


```python
from dfply import *

selected_data = cars >> select(~X.model_year, ~X.name)
print(selected_data)
```

          mpg  cylinders  displacement  horsepower  weight  acceleration  origin
    0    18.0          8         307.0       130.0    3504          12.0     usa
    1    15.0          8         350.0       165.0    3693          11.5     usa
    2    18.0          8         318.0       150.0    3436          11.0     usa
    3    16.0          8         304.0       150.0    3433          12.0     usa
    4    17.0          8         302.0       140.0    3449          10.5     usa
    ..    ...        ...           ...         ...     ...           ...     ...
    393  27.0          4         140.0        86.0    2790          15.6     usa
    394  44.0          4          97.0        52.0    2130          24.6  europe
    395  32.0          4         135.0        84.0    2295          11.6     usa
    396  28.0          4         120.0        79.0    2625          18.6     usa
    397  31.0          4         119.0        82.0    2720          19.4     usa
    
    [398 rows x 7 columns]
    

## Filtering columns

The vanilla select and drop functions are useful, but there are a variety of selection functions inspired by dplyr available to make selecting and dropping columns a breeze. These functions are intended to be put inside of the select and drop functions, and can be paired with the ~ inverter.

First, a quick rundown of the available functions:

-   starts_with(prefix): find columns that start with a string prefix.
-   ends_with(suffix): find columns that end with a string suffix.
-   contains(substr): find columns that contain a substring in their name.
-   everything(): all columns.
-   columns_between(start_col, end_col, inclusive=True): find columns between a specified start and end column. The inclusive boolean keyword argument indicates whether the end column should be included or not.
-   columns_to(end_col, inclusive=True): get columns up to a specified end column. The inclusive argument indicates whether the ending column should be included or not.
-   columns_from(start_col): get the columns starting at a specified column.

### Exercise 4.

The selection filter functions are best explained by example. Let's say I wanted to select only the columns that started with a "c":


```python
from dfply import *

selected_data = cars >> select(starts_with("c"))
print(selected_data)
```

         cylinders
    0            8
    1            8
    2            8
    3            8
    4            8
    ..         ...
    393          4
    394          4
    395          4
    396          4
    397          4
    
    [398 rows x 1 columns]
    

### Exercise 5.

Select the columns that contain the substring "e" from the cars DataFrame.


```python
from dfply import *

selected_data = cars >> select(contains("e"))
print(selected_data)
```

         cylinders  displacement  horsepower  weight  acceleration  model_year  \
    0            8         307.0       130.0    3504          12.0          70   
    1            8         350.0       165.0    3693          11.5          70   
    2            8         318.0       150.0    3436          11.0          70   
    3            8         304.0       150.0    3433          12.0          70   
    4            8         302.0       140.0    3449          10.5          70   
    ..         ...           ...         ...     ...           ...         ...   
    393          4         140.0        86.0    2790          15.6          82   
    394          4          97.0        52.0    2130          24.6          82   
    395          4         135.0        84.0    2295          11.6          82   
    396          4         120.0        79.0    2625          18.6          82   
    397          4         119.0        82.0    2720          19.4          82   
    
                              name  
    0    chevrolet chevelle malibu  
    1            buick skylark 320  
    2           plymouth satellite  
    3                amc rebel sst  
    4                  ford torino  
    ..                         ...  
    393            ford mustang gl  
    394                  vw pickup  
    395              dodge rampage  
    396                ford ranger  
    397                 chevy s-10  
    
    [398 rows x 7 columns]
    

### Exercise 6.

Select the columns that are between 'mpg' and 'origin' from the cars DataFrame.


```python
from dfply import *

selected_data = cars >> select(columns_between("mpg", "origin"))
print(selected_data)
```

          mpg  cylinders  displacement  horsepower  weight  acceleration  \
    0    18.0          8         307.0       130.0    3504          12.0   
    1    15.0          8         350.0       165.0    3693          11.5   
    2    18.0          8         318.0       150.0    3436          11.0   
    3    16.0          8         304.0       150.0    3433          12.0   
    4    17.0          8         302.0       140.0    3449          10.5   
    ..    ...        ...           ...         ...     ...           ...   
    393  27.0          4         140.0        86.0    2790          15.6   
    394  44.0          4          97.0        52.0    2130          24.6   
    395  32.0          4         135.0        84.0    2295          11.6   
    396  28.0          4         120.0        79.0    2625          18.6   
    397  31.0          4         119.0        82.0    2720          19.4   
    
         model_year  origin  
    0            70     usa  
    1            70     usa  
    2            70     usa  
    3            70     usa  
    4            70     usa  
    ..          ...     ...  
    393          82     usa  
    394          82  europe  
    395          82     usa  
    396          82     usa  
    397          82     usa  
    
    [398 rows x 8 columns]
    

## Subsetting and filtering

### row_slice()

Slices of rows can be selected with the row_slice() function. You can pass single integer indices or a list of indices to select rows as with. This is going to be the same as using pandas' .iloc.

#### Exercise 7.

Select the first three rows from the cars DataFrame.


```python
from dfply import *

selected_rows = cars >> row_slice([0, 1, 2])  
print(selected_rows)
```

        mpg  cylinders  displacement  horsepower  weight  acceleration  \
    0  18.0          8         307.0       130.0    3504          12.0   
    1  15.0          8         350.0       165.0    3693          11.5   
    2  18.0          8         318.0       150.0    3436          11.0   
    
       model_year origin                       name  
    0          70    usa  chevrolet chevelle malibu  
    1          70    usa          buick skylark 320  
    2          70    usa         plymouth satellite  
    

### distinct()

Selection of unique rows is done with distinct(), which similarly passes arguments and keyword arguments through to the DataFrame's .drop_duplicates() method.

#### Exercise 8.

Select the unique rows from the 'origin' column in the cars DataFrame.


```python
from dfply import *

unique_origins = cars >> distinct(X.origin)
print(unique_origins)
```

         mpg  cylinders  displacement  horsepower  weight  acceleration  \
    0   18.0          8         307.0       130.0    3504          12.0   
    14  24.0          4         113.0        95.0    2372          15.0   
    19  26.0          4          97.0        46.0    1835          20.5   
    
        model_year  origin                          name  
    0           70     usa     chevrolet chevelle malibu  
    14          70   japan         toyota corona mark ii  
    19          70  europe  volkswagen 1131 deluxe sedan  
    

## mask()

Filtering rows with logical criteria is done with mask(), which accepts boolean arrays "masking out" False labeled rows and keeping True labeled rows. These are best created with logical statements on symbolic Series objects as shown below. Multiple criteria can be supplied as arguments and their intersection will be used as the mask.

### Exercise 9.

Filter the cars DataFrame to only include rows where the 'mpg' is greater than 20, origin Japan, and display the first three rows:


```python
from dfply import *

filtered_cars = (cars >> mask((X.mpg > 20) & (X.origin == "japan")))
print(filtered_cars.head(3))
```

         mpg  cylinders  displacement  horsepower  weight  acceleration  \
    14  24.0          4         113.0        95.0    2372          15.0   
    18  27.0          4          97.0        88.0    2130          14.5   
    29  27.0          4          97.0        88.0    2130          14.5   
    
        model_year origin                   name  
    14          70  japan  toyota corona mark ii  
    18          70  japan           datsun pl510  
    29          71  japan           datsun pl510  
    

## pull()

The pull() function is used to extract a single column from a DataFrame as a pandas Series. This is useful for passing a single column to a function or for further manipulation.

### Exercise 10.

Extract the 'mpg' column from the cars DataFrame, japanese origin, model year 70s, and display the first three rows.


```python
from dfply import *

filtered_mpg = (cars >> mask((X.origin == "japan") & (X.model_year >= 70) & (X.model_year <= 79)) >> select(X.mpg) >> head(3))

print(filtered_mpg)
```

         mpg
    14  24.0
    18  27.0
    29  27.0
    

## DataFrame transformation

*mutate()*

The mutate() function is used to create new columns or modify existing columns. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 11.

Create a new column 'mpg_per_cylinder' in the cars DataFrame that is the result of dividing the 'mpg' column by the 'cylinders' column.


```python
from dfply import *

cars_transformed = cars >> mutate(mpg_per_cylinder=X.mpg / X.cylinders)
print(cars_transformed)
```

          mpg  cylinders  displacement  horsepower  weight  acceleration  \
    0    18.0          8         307.0       130.0    3504          12.0   
    1    15.0          8         350.0       165.0    3693          11.5   
    2    18.0          8         318.0       150.0    3436          11.0   
    3    16.0          8         304.0       150.0    3433          12.0   
    4    17.0          8         302.0       140.0    3449          10.5   
    ..    ...        ...           ...         ...     ...           ...   
    393  27.0          4         140.0        86.0    2790          15.6   
    394  44.0          4          97.0        52.0    2130          24.6   
    395  32.0          4         135.0        84.0    2295          11.6   
    396  28.0          4         120.0        79.0    2625          18.6   
    397  31.0          4         119.0        82.0    2720          19.4   
    
         model_year  origin                       name  mpg_pre_cylinder  \
    0            70     usa  chevrolet chevelle malibu             2.250   
    1            70     usa          buick skylark 320             1.875   
    2            70     usa         plymouth satellite             2.250   
    3            70     usa              amc rebel sst             2.000   
    4            70     usa                ford torino             2.125   
    ..          ...     ...                        ...               ...   
    393          82     usa            ford mustang gl             6.750   
    394          82  europe                  vw pickup            11.000   
    395          82     usa              dodge rampage             8.000   
    396          82     usa                ford ranger             7.000   
    397          82     usa                 chevy s-10             7.750   
    
         mpg_per_cylinder  
    0               2.250  
    1               1.875  
    2               2.250  
    3               2.000  
    4               2.125  
    ..                ...  
    393             6.750  
    394            11.000  
    395             8.000  
    396             7.000  
    397             7.750  
    
    [398 rows x 11 columns]
    


*transmute()*

The transmute() function is a combination of a mutate and a selection of the created variables.

### Exercise 12.

Create a new column 'mpg_per_cylinder' in the cars DataFrame that is the result of dividing the 'mpg' column by the 'cylinders' column, and display only the new column.


```python
from dfply import *
cars_transformed = cars >> transmute(mpg_per_cylinder=X.mpg / X.cylinders)
print(cars_transformed)

```

         mpg_per_cylinder
    0               2.250
    1               1.875
    2               2.250
    3               2.000
    4               2.125
    ..                ...
    393             6.750
    394            11.000
    395             8.000
    396             7.000
    397             7.750
    
    [398 rows x 1 columns]
    

## Grouping

*group_by() and ungroup()*

The group_by() function is used to group the DataFrame by one or more columns. This is useful for creating groups of rows that can be summarized or transformed together. The ungroup() function is used to remove the grouping.

### Exercise 13.

Group the cars DataFrame by the 'origin' column and calculate the lead of the 'mpg' column.


```python
from dfply import *

cars_grouped = cars >> group_by(X.origin) >> mutate(mpg_lead=lead(X.mpg))
print(cars_grouped)
```

          mpg  cylinders  displacement  horsepower  weight  acceleration  \
    19   26.0          4          97.0        46.0    1835          20.5   
    20   25.0          4         110.0        87.0    2672          17.5   
    21   24.0          4         107.0        90.0    2430          14.5   
    22   25.0          4         104.0        95.0    2375          17.5   
    23   26.0          4         121.0       113.0    2234          12.5   
    ..    ...        ...           ...         ...     ...           ...   
    392  27.0          4         151.0        90.0    2950          17.3   
    393  27.0          4         140.0        86.0    2790          15.6   
    395  32.0          4         135.0        84.0    2295          11.6   
    396  28.0          4         120.0        79.0    2625          18.6   
    397  31.0          4         119.0        82.0    2720          19.4   
    
         model_year  origin                          name  mpg_pre_cylinder  \
    19           70  europe  volkswagen 1131 deluxe sedan              6.50   
    20           70  europe                   peugeot 504              6.25   
    21           70  europe                   audi 100 ls              6.00   
    22           70  europe                      saab 99e              6.25   
    23           70  europe                      bmw 2002              6.50   
    ..          ...     ...                           ...               ...   
    392          82     usa              chevrolet camaro              6.75   
    393          82     usa               ford mustang gl              6.75   
    395          82     usa                 dodge rampage              8.00   
    396          82     usa                   ford ranger              7.00   
    397          82     usa                    chevy s-10              7.75   
    
         mpg_lead  
    19       25.0  
    20       24.0  
    21       25.0  
    22       26.0  
    23       28.0  
    ..        ...  
    392      27.0  
    393      32.0  
    395      28.0  
    396      31.0  
    397       NaN  
    
    [398 rows x 11 columns]
    

## Reshaping

*arrange()*

The arrange() function is used to sort the DataFrame by one or more columns. This is useful for reordering the rows of the DataFrame.

### Exercise 14.

Sort the cars DataFrame by the 'mpg' column in descending order.


```python
from dfply import *

sorted_cars = cars >> arrange(desc(X.mpg))
print(sorted_cars)
```

          mpg  cylinders  displacement  horsepower  weight  acceleration  \
    322  46.6          4          86.0        65.0    2110          17.9   
    329  44.6          4          91.0        67.0    1850          13.8   
    325  44.3          4          90.0        48.0    2085          21.7   
    394  44.0          4          97.0        52.0    2130          24.6   
    326  43.4          4          90.0        48.0    2335          23.7   
    ..    ...        ...           ...         ...     ...           ...   
    67   11.0          8         429.0       208.0    4633          11.0   
    27   11.0          8         318.0       210.0    4382          13.5   
    25   10.0          8         360.0       215.0    4615          14.0   
    26   10.0          8         307.0       200.0    4376          15.0   
    28    9.0          8         304.0       193.0    4732          18.5   
    
         model_year  origin                  name  mpg_pre_cylinder  
    322          80   japan             mazda glc            11.650  
    329          80   japan   honda civic 1500 gl            11.150  
    325          80  europe  vw rabbit c (diesel)            11.075  
    394          82  europe             vw pickup            11.000  
    326          80  europe    vw dasher (diesel)            10.850  
    ..          ...     ...                   ...               ...  
    67           72     usa       mercury marquis             1.375  
    27           70     usa            dodge d200             1.375  
    25           70     usa             ford f250             1.250  
    26           70     usa             chevy c20             1.250  
    28           70     usa              hi 1200d             1.125  
    
    [398 rows x 10 columns]
    


*rename()*

The rename() function is used to rename columns in the DataFrame. It accepts keyword arguments of the form new_column_name = old_column_name.

### Exercise 15.

Rename the 'mpg' column to 'miles_per_gallon' in the cars DataFrame.


```python
from dfply import *

renamed_cars = cars >> rename(miles_per_gallon=X.mpg)
print(renamed_cars)
```

         miles_per_gallon  cylinders  displacement  horsepower  weight  \
    0                18.0          8         307.0       130.0    3504   
    1                15.0          8         350.0       165.0    3693   
    2                18.0          8         318.0       150.0    3436   
    3                16.0          8         304.0       150.0    3433   
    4                17.0          8         302.0       140.0    3449   
    ..                ...        ...           ...         ...     ...   
    393              27.0          4         140.0        86.0    2790   
    394              44.0          4          97.0        52.0    2130   
    395              32.0          4         135.0        84.0    2295   
    396              28.0          4         120.0        79.0    2625   
    397              31.0          4         119.0        82.0    2720   
    
         acceleration  model_year  origin                       name  \
    0            12.0          70     usa  chevrolet chevelle malibu   
    1            11.5          70     usa          buick skylark 320   
    2            11.0          70     usa         plymouth satellite   
    3            12.0          70     usa              amc rebel sst   
    4            10.5          70     usa                ford torino   
    ..            ...         ...     ...                        ...   
    393          15.6          82     usa            ford mustang gl   
    394          24.6          82  europe                  vw pickup   
    395          11.6          82     usa              dodge rampage   
    396          18.6          82     usa                ford ranger   
    397          19.4          82     usa                 chevy s-10   
    
         mpg_pre_cylinder  
    0               2.250  
    1               1.875  
    2               2.250  
    3               2.000  
    4               2.125  
    ..                ...  
    393             6.750  
    394            11.000  
    395             8.000  
    396             7.000  
    397             7.750  
    
    [398 rows x 10 columns]
    


*gather()*

The gather() function is used to reshape the DataFrame from wide to long format. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 16.

Reshape the cars DataFrame from wide to long format by gathering the columns 'mpg', 'horsepower', 'weight', 'acceleration', and 'displacement' into a new column 'variable' and their values into a new column 'value'.


```python
from dfply import *

long_cars = cars >> gather("variable", "value", X.mpg, X.horsepower, X.weight, X.acceleration, X.displacement)
print(long_cars)
```

          cylinders  model_year  origin                       name  \
    0             8          70     usa  chevrolet chevelle malibu   
    1             8          70     usa          buick skylark 320   
    2             8          70     usa         plymouth satellite   
    3             8          70     usa              amc rebel sst   
    4             8          70     usa                ford torino   
    ...         ...         ...     ...                        ...   
    1985          4          82     usa            ford mustang gl   
    1986          4          82  europe                  vw pickup   
    1987          4          82     usa              dodge rampage   
    1988          4          82     usa                ford ranger   
    1989          4          82     usa                 chevy s-10   
    
          mpg_pre_cylinder      variable  value  
    0                2.250           mpg   18.0  
    1                1.875           mpg   15.0  
    2                2.250           mpg   18.0  
    3                2.000           mpg   16.0  
    4                2.125           mpg   17.0  
    ...                ...           ...    ...  
    1985             6.750  displacement  140.0  
    1986            11.000  displacement   97.0  
    1987             8.000  displacement  135.0  
    1988             7.000  displacement  120.0  
    1989             7.750  displacement  119.0  
    
    [1990 rows x 7 columns]
    


*spread()*

Likewise, you can transform a "long" DataFrame into a "wide" format with the spread(key, values) function. Converting the previously created elongated DataFrame for example would be done like so.

### Exercise 17.

Reshape the cars DataFrame from long to wide format by spreading the 'variable' column into columns and their values into the 'value' column.


```python
from dfply import *

wide_cars = long_cars >> spread(X.variable, X.value)
print(wide_cars)
```

          level_0  index  cylinders  model_year  origin  \
    0           0      0          8          70     usa   
    1           1      1          8          70     usa   
    2           2      2          8          70     usa   
    3           3      3          8          70     usa   
    4           4      4          8          70     usa   
    ...       ...    ...        ...         ...     ...   
    1985     1985   1985          4          82     usa   
    1986     1986   1986          4          82  europe   
    1987     1987   1987          4          82     usa   
    1988     1988   1988          4          82     usa   
    1989     1989   1989          4          82     usa   
    
                               name  acceleration  displacement  horsepower   mpg  \
    0     chevrolet chevelle malibu           NaN           NaN         NaN  18.0   
    1             buick skylark 320           NaN           NaN         NaN  15.0   
    2            plymouth satellite           NaN           NaN         NaN  18.0   
    3                 amc rebel sst           NaN           NaN         NaN  16.0   
    4                   ford torino           NaN           NaN         NaN  17.0   
    ...                         ...           ...           ...         ...   ...   
    1985            ford mustang gl           NaN         140.0         NaN   NaN   
    1986                  vw pickup           NaN          97.0         NaN   NaN   
    1987              dodge rampage           NaN         135.0         NaN   NaN   
    1988                ford ranger           NaN         120.0         NaN   NaN   
    1989                 chevy s-10           NaN         119.0         NaN   NaN   
    
          weight  
    0        NaN  
    1        NaN  
    2        NaN  
    3        NaN  
    4        NaN  
    ...      ...  
    1985     NaN  
    1986     NaN  
    1987     NaN  
    1988     NaN  
    1989     NaN  
    
    [1990 rows x 11 columns]
    


## Summarization

*summarize()*

The summarize() function is used to calculate summary statistics for groups of rows. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 18.

Calculate the mean 'mpg' for each group of 'origin' in the cars DataFrame.


```python
from dfply import *

mean_mpg_by_origin = cars >> group_by(X.origin) >> summarize(mean_mpg=X.mpg.mean())
print(mean_mpg_by_origin)
```

       origin   mean_mpg
    0  europe  27.891429
    1   japan  30.450633
    2     usa  20.083534
    


*summarize_each()*

The summarize_each() function is used to calculate summary statistics for groups of rows. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 19.

Calculate the mean 'mpg' and 'horsepower' for each group of 'origin' in the cars DataFrame.


```python
from dfply import *

mean_stats_by_origin = (cars >> group_by(X.origin) >> summarize(mean_mpg=X.mpg.mean(), mean_horsepower=X.horsepower.mean()))
print(mean_stats_by_origin)
```

       origin   mean_mpg  mean_horsepower
    0  europe  27.891429        80.558824
    1   japan  30.450633        79.835443
    2     usa  20.083534       119.048980
    


*summarize() can of course be used with groupings as well.*

### Exercise 20.

Calculate the mean 'mpg' for each group of 'origin' and 'model_year' in the cars DataFrame.


```python
from dfply import *

mean_mpg_by_origin_year = (cars >> group_by(X.origin, X.model_year) >> summarize(mean_mpg=X.mpg.mean()))
print(mean_mpg_by_origin_year)
```

        model_year  origin   mean_mpg
    0           70  europe  25.200000
    1           71  europe  28.750000
    2           72  europe  22.000000
    3           73  europe  24.000000
    4           74  europe  27.000000
    5           75  europe  24.500000
    6           76  europe  24.250000
    7           77  europe  29.250000
    8           78  europe  24.950000
    9           79  europe  30.450000
    10          80  europe  37.288889
    11          81  europe  31.575000
    12          82  europe  40.000000
    13          70   japan  25.500000
    14          71   japan  29.500000
    15          72   japan  24.200000
    16          73   japan  20.000000
    17          74   japan  29.333333
    18          75   japan  27.500000
    19          76   japan  28.000000
    20          77   japan  27.416667
    21          78   japan  29.687500
    22          79   japan  32.950000
    23          80   japan  35.400000
    24          81   japan  32.958333
    25          82   japan  34.888889
    26          70     usa  15.272727
    27          71     usa  18.100000
    28          72     usa  16.277778
    29          73     usa  15.034483
    30          74     usa  18.333333
    31          75     usa  17.550000
    32          76     usa  19.431818
    33          77     usa  20.722222
    34          78     usa  21.772727
    35          79     usa  23.478261
    36          80     usa  25.914286
    37          81     usa  27.530769
    38          82     usa  29.450000
    
