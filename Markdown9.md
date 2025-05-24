---
title: Bivariate Statistics
subtitle: Foundations of Statistical Analysis in Python
abstract: This notebook explores bivariate relationships through linear correlations, highlighting their strengths and limitations. Practical examples and visualizations are provided to help users understand and apply these statistical concepts effectively.
author:
  - name: Karol Flisikowski
    affiliations: 
      - Gdansk University of Technology
      - Chongqing Technology and Business University
    orcid: 0000-0002-4160-1297
    email: karol@ctbu.edu.cn
date: 2025-05-03
---

## Goals of this lecture

There are many ways to *describe* a distribution. 

Here we will discuss:
- Measurement of the relationship between distributions using **linear, rank correlations**.
- Measurement of the relationship between qualitative variables using **contingency**.

## Importing relevant libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns ### importing seaborn
import pandas as pd
import scipy.stats as ss
```


```python
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
```


```python
import pandas as pd
df_pokemon = pd.read_csv("pokemon.csv")
```

## Describing *bivariate* data with correlations

- So far, we've been focusing on *univariate data*: a single distribution.
- What if we want to describe how *two distributions* relate to each other?
   - For today, we'll focus on *continuous distributions*.

### Bivariate relationships: `height`

- A classic example of **continuous bivariate data** is the `height` of a `parent` and `child`.  
- [These data were famously collected by Karl Pearson](https://www.kaggle.com/datasets/abhilash04/fathersandsonheight).


```python
df_height = pd.read_csv("Pearson.csv")
df_height.head(2)

df_height[['Father', 'Son']] = df_height['Father\tSon'].str.split('\t', expand=True)

df_height = df_height.drop('Father\tSon', axis=1)

df_height['Father'] = pd.to_numeric(df_height['Father'], errors='coerce')
df_height['Son'] = pd.to_numeric(df_height['Son'], errors='coerce')

#print(df_height.columns)  
#print(df_height.head())   

```

#### Plotting Pearson's height data


```python
sns.scatterplot(data = df_height, x = "Father", y = "Son", alpha = 0.5);
```


    
![png](Markdown9_files/Markdown9_10_0.png)
    


### Introducing linear correlations

> A **correlation coefficient** is a number between $[–1, 1]$ that describes the relationship between a pair of variables.

Specifically, **Pearson's correlation coefficient** (or Pearson's $r$) describes a (presumed) *linear* relationship.

Two key properties:

- **Sign**: whether a relationship is positive (+) or negative (–).  
- **Magnitude**: the strength of the linear relationship.

$$
r = \frac{ \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) }{ \sqrt{ \sum_{i=1}^{n} (x_i - \bar{x})^2 } \sqrt{ \sum_{i=1}^{n} (y_i - \bar{y})^2 } }
$$

Where:
- $r$ - Pearson correlation coefficient
- $x_i$, $y_i$ - values of the variables
- $\bar{x}$, $\bar{y}$ - arithmetic means
- $n$ - number of observations

Pearson's correlation coefficient measures the strength and direction of the linear relationship between two continuous variables. Its value ranges from -1 to 1:
- 1 → perfect positive linear correlation
- 0 → no linear correlation
- -1 → perfect negative linear correlation

This coefficient does not tell about nonlinear correlations and is sensitive to outliers.

### Calculating Pearson's $r$ with `scipy`

`scipy.stats` has a function called `pearsonr`, which will calculate this relationship for you.

Returns two numbers:

- $r$: the correlation coefficent.  
- $p$: the **p-value** of this correlation coefficient, i.e., whether it's *significantly different* from `0`.


```python
ss.pearsonr(df_height['Father'], df_height['Son'])
```




    PearsonRResult(statistic=np.float64(0.5011626808075912), pvalue=np.float64(1.2729275743661585e-69))



#### Check-in

Using `scipy.stats.pearsonr` (here, `ss.pearsonr`), calculate Pearson's $r$ for the relationship between the `Attack` and `Defense` of Pokemon.

- Is this relationship positive or negative?  
- How strong is this relationship?


```python
from scipy.stats import pearsonr

attack = df_pokemon["Attack"]
defense = df_pokemon["Defense"]
r, p_value = pearsonr(attack, defense)

print(f"Pearson's r: {r:.3f}")


#Direction: The value r is positive, indicating that as Attack increases, Defense tends to increase as well.
#Strength: The relationship is moderate 
```

    Pearson's r: 0.439
    

#### Solution


```python
ss.pearsonr(df_pokemon['Attack'], df_pokemon['Defense'])
```




    PearsonRResult(statistic=np.float64(0.4386870551184896), pvalue=np.float64(5.858479864288896e-39))



#### Check-in

Pearson'r $r$ measures the *linear correlation* between two variables. Can anyone think of potential limitations to this approach?

### Limitations of Pearson's $r$

- Pearson's $r$ *presumes* a linear relationship and tries to quantify its strength and direction.  
- But many relationships are **non-linear**!  
- Unless we visualize our data, relying only on Pearson'r $r$ could mislead us.

#### Non-linear data where $r = 0$


```python
x = np.arange(1, 40)
y = np.sin(x)
p = sns.lineplot(x = x, y = y)
```


    
![png](Markdown9_files/Markdown9_23_0.png)
    



```python
### r is close to 0, despite there being a clear relationship!
ss.pearsonr(x, y)
```




    PearsonRResult(statistic=np.float64(-0.04067793461845847), pvalue=np.float64(0.8057827185936635))



#### When $r$ is invariant to the real relationship

All these datasets have roughly the same **correlation coefficient**.


```python
df_anscombe = sns.load_dataset("anscombe")
sns.relplot(data = df_anscombe, x = "x", y = "y", col = "dataset");
```


    
![png](Markdown9_files/Markdown9_26_0.png)
    



```python
# Compute correlation matrix
corr = df_pokemon.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr, 
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](Markdown9_files/Markdown9_27_0.png)
    


## Rank Correlations

Rank correlations are measures of the strength and direction of a monotonic (increasing or decreasing) relationship between two variables. Instead of numerical values, they use ranks, i.e., positions in an ordered set.

They are less sensitive to outliers and do not require linearity (unlike Pearson's correlation).

### Types of Rank Correlations

1. $ρ$ (rho) **Spearman's**
- Based on the ranks of the data.
- Value: from –1 to 1.
- Works well for monotonic but non-linear relationships.

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

Where:
- $d_i$ – differences between the ranks of observations,
- $n$ – number of observations.

2. $τ$ (tau) **Kendall's**
- Measures the number of concordant vs. discordant pairs.
- More conservative than Spearman's – often yields smaller values.
- Also ranges from –1 to 1.

$$
\tau = \frac{(C - D)}{\frac{1}{2}n(n - 1)}
$$

Where:
- $τ$ — Kendall's correlation coefficient,
- $C$ — number of concordant pairs,
- $D$ — number of discordant pairs,
- $n$ — number of observations,
- $\frac{1}{2}n(n - 1)$ — total number of possible pairs of observations.

What are concordant and discordant pairs?
- Concordant pair: if $x_i$ < $x_j$ and $y_i$ < $y_j$, or $x_i$ > $x_j$ and $y_i$ > $y_j$.
- Discordant pair: if $x_i$ < $x_j$ and $y_i$ > $y_j$, or $x_i$ > $x_j$ and $y_i$ < $y_j$.

### When to use rank correlations?
- When the data are not normally distributed.
- When you suspect a non-linear but monotonic relationship.
- When you have rank correlations, such as grades, ranking, preference level.

| Correlation type | Description | When to use |
|------------------|-----------------------------------------------------|----------------------------------------|
| Spearman's (ρ) | Monotonic correlation, based on ranks | When data are nonlinear or have outliers |
| Kendall's (τ) | Counts the proportion of congruent and incongruent pairs | When robustness to ties is important |

### Interpretation of correlation values

| Range of values | Correlation interpretation |
|------------------|----------------------------------|
| 0.8 - 1.0 | very strong positive |
| 0.6 - 0.8 | strong positive |
| 0.4 - 0.6 | moderate positive |
| 0.2 - 0.4 | weak positive |
| 0.0 - 0.2 | very weak or no correlation |
| < 0 | similarly - negative correlation |


```python
# Compute Kendall rank correlation
corr_kendall = df_pokemon.corr(method='kendall', numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr, 
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](Markdown9_files/Markdown9_32_0.png)
    


### Comparison of Correlation Coefficients

| Property                | Pearson (r)                   | Spearman (ρ)                        | Kendall (τ)                          |
|-------------------------|-------------------------------|--------------------------------------|---------------------------------------|
| What it measures?       | Linear relationship           | Monotonic relationship (based on ranks) | Monotonic relationship (based on pairs) |
| Data type               | Quantitative, normal distribution | Ranks or ordinal/quantitative data  | Ranks or ordinal/quantitative data   |
| Sensitivity to outliers | High                          | Lower                               | Low                                   |
| Value range             | –1 to 1                       | –1 to 1                             | –1 to 1                               |
| Requires linearity      | Yes                           | No                                  | No                                    |
| Robustness to ties      | Low                           | Medium                              | High                                  |
| Interpretation          | Strength and direction of linear relationship | Strength and direction of monotonic relationship | Proportion of concordant vs discordant pairs |
| Significance test       | Yes (`scipy.stats.pearsonr`)  | Yes (`spearmanr`)                   | Yes (`kendalltau`)                   |

Brief summary:
- Pearson - best when the data are normal and the relationship is linear.
- Spearman - works better for non-linear monotonic relationships.
- Kendall - more conservative, often used in social research, less sensitive to small changes in data.

### Your Turn

For the Pokemon dataset, find the pairs of variables that are most appropriate for using one of the quantitative correlation measures. Calculate them, then visualize them.


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


quantitative_vars = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
df_quantitative = df_pokemon[quantitative_vars].dropna()

print("\nQuantitative variables selected.")
print(df_quantitative.head())


selected_pairs = [
    ('Attack', 'Defense'),
    ('Sp. Atk', 'Sp. Def'),
    ('Speed', 'HP'),
]

print("\nCalculating and Visualizing selected pairs:")

for var1, var2 in selected_pairs:
    if var1 in df_quantitative.columns and var2 in df_quantitative.columns:
        x = df_quantitative[var1]
        y = df_quantitative[var2]
        
        pearson_corr, pearson_p_value = pearsonr(x, y)
        spearman_corr, spearman_p_value = spearmanr(x, y)
        kendall_corr, kendall_p_value = kendalltau(x, y)

        print(f"\nAnalyzing relationship between {var1} and {var2}:")
        print(f"  Pearson r: {pearson_corr:.3f} (p-value: {pearson_p_value:.4f})")
        print(f"  Spearman rho: {spearman_corr:.3f} (p-value: {spearman_p_value:.4f})")
        print(f"  Kendall Tau: {kendall_corr:.3f} (p-value: {kendall_p_value:.4f})")
        print("-" * 30)

        plt.figure(figsize=(8, 6))
        sns.regplot(x=var1, y=var2, data=df_quantitative, scatter_kws={'alpha':0.5})
        plt.title(f'Scatterplot of {var1} vs. {var2}\nPearson r = {pearson_corr:.3f}')
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    else:
        print(f"\nWarning: One or both of {var1}, {var2} not found in quantitative columns.")

# --- Interpretation Notes ---
print("\nInterpretation Notes:")
print("Pearson r measures the strength and direction of a *linear* relationship.")
print("Spearman rho and Kendall Tau measure the strength and direction of a *monotonic* relationship (based on ranks), which may be more appropriate if the relationship is not strictly linear or if there are outliers.")
print("The p-value indicates the statistical significance. A small p-value (typically < 0.05) suggests that the observed correlation is unlikely to be due to random chance.")
print("\nTo interpret the strength of correlation coefficients (r, rho, or Tau):")
print("- Absolute value near 1: Strong relationship")
print("- Absolute value around 0.5: Moderate relationship")
print("- Absolute value around 0.1-0.3: Weak relationship")
print("- Absolute value near 0: Very weak or no relationship")
```

    
    Quantitative variables selected.
       Total  HP  Attack  Defense  Sp. Atk  Sp. Def  Speed  Generation
    0    318  45      49       49       65       65     45           1
    1    405  60      62       63       80       80     60           1
    2    525  80      82       83      100      100     80           1
    3    625  80     100      123      122      120     80           1
    4    309  39      52       43       60       50     65           1
    
    Calculating and Visualizing selected pairs:
    
    Analyzing relationship between Attack and Defense:
      Pearson r: 0.439 (p-value: 0.0000)
      Spearman rho: 0.515 (p-value: 0.0000)
      Kendall Tau: 0.370 (p-value: 0.0000)
    ------------------------------
    


    
![png](Markdown9_files/Markdown9_36_1.png)
    


    
    Analyzing relationship between Sp. Atk and Sp. Def:
      Pearson r: 0.506 (p-value: 0.0000)
      Spearman rho: 0.572 (p-value: 0.0000)
      Kendall Tau: 0.423 (p-value: 0.0000)
    ------------------------------
    


    
![png](Markdown9_files/Markdown9_36_3.png)
    


    
    Analyzing relationship between Speed and HP:
      Pearson r: 0.176 (p-value: 0.0000)
      Spearman rho: 0.266 (p-value: 0.0000)
      Kendall Tau: 0.180 (p-value: 0.0000)
    ------------------------------
    


    
![png](Markdown9_files/Markdown9_36_5.png)
    


    
    Interpretation Notes:
    Pearson r measures the strength and direction of a *linear* relationship.
    Spearman rho and Kendall Tau measure the strength and direction of a *monotonic* relationship (based on ranks), which may be more appropriate if the relationship is not strictly linear or if there are outliers.
    The p-value indicates the statistical significance. A small p-value (typically < 0.05) suggests that the observed correlation is unlikely to be due to random chance.
    
    To interpret the strength of correlation coefficients (r, rho, or Tau):
    - Absolute value near 1: Strong relationship
    - Absolute value around 0.5: Moderate relationship
    - Absolute value around 0.1-0.3: Weak relationship
    - Absolute value near 0: Very weak or no relationship
    

## Correlation of Qualitative Variables

A categorical variable is one that takes descriptive values ​​that represent categories—e.g. Pokémon type (Fire, Water, Grass), gender, status (Legendary vs. Normal), etc.

Such variables cannot be analyzed directly using correlation methods for numbers (Pearson, Spearman, Kendall). Other techniques are used instead.

### Contingency Table

A contingency table is a special cross-tabulation table that shows the frequency (i.e., the number of cases) for all possible combinations of two categorical variables.

It is a fundamental tool for analyzing relationships between qualitative features.

#### Chi-Square Test of Independence

The Chi-Square test checks whether there is a statistically significant relationship between two categorical variables.

Concept:

We compare:
- observed values (from the contingency table),
- with expected values, assuming the variables are independent.

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

Where:
- $O_{ij}$ – observed count in cell ($i$, $j$),
- $E_{ij}$ – expected count in cell ($i$, $j$), assuming independence.

### Example: Calculating Expected Values and Chi-Square Statistic in Python

Here’s how you can calculate the **expected values** and **Chi-Square statistic (χ²)** step by step using Python.

---

#### Step 1: Create the Observed Contingency Table
We will use the Pokémon example:

| Type 1 | Legendary = False | Legendary = True | Total |
|--------|-------------------|------------------|-------|
| Fire   | 18                | 5                | 23    |
| Water  | 25                | 3                | 28    |
| Grass  | 20                | 2                | 22    |
| Total  | 63                | 10               | 73    |


```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Observed values (contingency table)
observed = np.array([
    [18, 5],  # Fire
    [25, 3],  # Water
    [20, 2]   # Grass
])

# Convert to DataFrame for better visualization
observed_df = pd.DataFrame(
    observed,
    columns=["Legendary = False", "Legendary = True"],
    index=["Fire", "Water", "Grass"]
)
print("Observed Table:")
print(observed_df)
```

    Observed Table:
           Legendary = False  Legendary = True
    Fire                  18                 5
    Water                 25                 3
    Grass                 20                 2
    

Step 2: Calculate Expected Values
The expected values are calculated using the formula:

$$ E_{ij} = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}} $$

You can calculate this manually or use scipy.stats.chi2_contingency, which automatically computes the expected values.


```python
# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(observed)

# Convert expected values to DataFrame for better visualization
expected_df = pd.DataFrame(
    expected,
    columns=["Legendary = False", "Legendary = True"],
    index=["Fire", "Water", "Grass"]
)
print("\nExpected Table:")
print(expected_df)
```

    
    Expected Table:
           Legendary = False  Legendary = True
    Fire           19.849315          3.150685
    Water          24.164384          3.835616
    Grass          18.986301          3.013699
    

Step 3: Calculate the Chi-Square Statistic
The Chi-Square statistic is calculated using the formula:

$$ \chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$

This is done automatically by scipy.stats.chi2_contingency, but you can also calculate it manually:


```python
# Manual calculation of Chi-Square statistic
chi2_manual = np.sum((observed - expected) ** 2 / expected)
print(f"\nChi-Square Statistic (manual): {chi2_manual:.4f}")
```

    
    Chi-Square Statistic (manual): 1.8638
    

Step 4: Interpret the Results
The chi2_contingency function also returns:

p-value: The probability of observing the data if the null hypothesis (independence) is true.
Degrees of Freedom (dof): Calculated as (rows - 1) * (columns - 1).


```python
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
```

    
    Chi-Square Statistic: 1.8638
    p-value: 0.3938
    Degrees of Freedom: 2
    

**Interpretation of the Chi-Square Test Result:**

| Value               | Meaning                                         |
|---------------------|-------------------------------------------------|
| High χ² value       | Large difference between observed and expected values |
| Low p-value         | Strong basis to reject the null hypothesis of independence |
| p < 0.05            | Statistically significant relationship between variables |

### Qualitative Correlations

#### Cramér's V

**Cramér's V** is a measure of the strength of association between two categorical variables. It is based on the Chi-Square test but scaled to a range of 0–1, making it easier to interpret the strength of the relationship.

$$
V = \sqrt{ \frac{\chi^2}{n \cdot (k - 1)} }
$$

Where:
- $\chi^2$ – Chi-Square test statistic,
- $n$ – number of observations,
- $k$ – the smaller number of categories (rows/columns) in the contingency table.

---

#### Phi Coefficient ($φ$)

Application:
- Both variables must be dichotomous (e.g., Yes/No, 0/1), meaning the table must have the smallest size of **2×2**.
- Ideal for analyzing relationships like gender vs purchase, type vs legendary.

$$
\phi = \sqrt{ \frac{\chi^2}{n} }
$$

Where:
- $\chi^2$ – Chi-Square test statistic for a 2×2 table,
- $n$ – number of observations.

---

#### Tschuprow’s T

**Tschuprow’s T** is a measure of association similar to **Cramér's V**, but it has a different scale. It is mainly used when the number of categories in the two variables differs. This is a more advanced measure applicable to a broader range of contingency tables.

$$
T = \sqrt{\frac{\chi^2}{n \cdot (k - 1)}}
$$

Where:
- $\chi^2$ – Chi-Square test statistic,
- $n$ – number of observations,
- $k$ – the smaller number of categories (rows or columns) in the contingency table.

Application: Tschuprow’s T is useful when dealing with contingency tables with varying numbers of categories in rows and columns.

---

### Summary - Qualitative Correlations

| Measure            | What it measures                                       | Application                     | Value Range     | Strength Interpretation       |
|--------------------|--------------------------------------------------------|---------------------------------|------------------|-------------------------------|
| **Cramér's V**     | Strength of association between nominal variables      | Any categories                  | 0 – 1           | 0.1–weak, 0.3–moderate, >0.5–strong |
| **Phi ($φ$)**      | Strength of association in a **2×2** table             | Two binary variables            | -1 – 1          | Similar to correlation        |
| **Tschuprow’s T**  | Strength of association, alternative to Cramér's V     | Tables with similar category counts | 0 – 1      | Less commonly used            |
| **Chi² ($χ²$)**    | Statistical test of independence                       | All categorical variables       | 0 – ∞           | Higher values indicate stronger differences |

### Example

Let's investigate whether the Pokémon's type (type_1) is affected by whether the Pokémon is legendary.

We'll use the **scipy** library.

This library already has built-in functions for calculating various qualitative correlation measures.


```python
from scipy.stats.contingency import association

# Contingency table:
ct = pd.crosstab(df_pokemon["Type 1"], df_pokemon["Legendary"])

# Calculating Cramér's V measure
V = association(ct, method="cramer") # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html#association

print(f"Cramer's V: {V}") # interpret!
# Cramér's V interpretation:
# 0.0 - 0.1: No association
# 0.1 - 0.3: Weak association
# 0.3 - 0.5: Moderate association <-- our result
# 0.5 - 0.7: Strong association
# 0.7 - 1.0: Very strong association
# between the two categorical variables
```

    Cramer's V: 0.3361928228447545
    

### Your turn

What visualization would be most appropriate for presenting a quantitative, ranked, and qualitative relationship?

Try to think about which pairs of variables could have which type of analysis based on the Pokemon data.

---


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_pokemon = df_pokemon.dropna(subset=['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Type 1', 'Type 2', 'Legendary'])
print(f"Using {len(df_pokemon)} rows after dropping missing values.")
print("\nGenerating visualizations for different variable relationship types...")

print("\nVisualizing Quantitative vs. Quantitative (Attack vs Defense)")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Attack', y='Defense', data=df_pokemon, alpha=0.6) 
plt.title('Scatterplot of Attack vs. Defense')
plt.xlabel('Attack Stat')
plt.ylabel('Defense Stat')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Visualizing Quantitative vs. Quantitative with Regression Line (Attack vs Defense)")
plt.figure(figsize=(8, 6))
sns.regplot(x='Attack', y='Defense', data=df_pokemon, scatter_kws={'alpha':0.5})
plt.title('Scatterplot with Regression Line: Attack vs. Defense')
plt.xlabel('Attack Stat')
plt.ylabel('Defense Stat')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\nVisualizing Quantitative vs. Ranked (Total stat across Generations)")
plt.figure(figsize=(10, 7))
sns.violinplot(x='Generation', y='Total', data=df_pokemon)
plt.title('Distribution of Total Stats by Generation (Violin Plot)')
plt.xlabel('Generation')
plt.ylabel('Total Stat')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

plt.figure(figsize=(10, 7))
sns.boxplot(x='Generation', y='Total', data=df_pokemon)
plt.title('Distribution of Total Stats by Generation (Box Plot)')
plt.xlabel('Generation')
plt.ylabel('Total Stat')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

print("\nVisualizing Quantitative vs. Qualitative (Attack stat by Type 1 - subset)")
common_types = ['Grass', 'Fire', 'Water', 'Electric', 'Normal', 'Psychic']
df_subset_types = df_pokemon[df_pokemon['Type 1'].isin(common_types)]

plt.figure(figsize=(12, 7))
sns.boxplot(x='Type 1', y='Attack', data=df_subset_types)
plt.title('Distribution of Attack Stats by Primary Type (Subset)')
plt.xlabel('Primary Type')
plt.ylabel('Attack Stat')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

print("\nVisualizing Ranked vs. Qualitative (Legendary status count by Generation)")
plt.figure(figsize=(10, 7))
sns.countplot(x='Generation', hue='Legendary', data=df_pokemon)
plt.title('Count of Legendary and Non-Legendary Pokemon by Generation')
plt.xlabel('Generation')
plt.ylabel('Number of Pokemon')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

print("\nVisualizing Qualitative vs. Qualitative (Type 1 vs Type 2 - Heatmap)")
type_crosstab = pd.crosstab(df_pokemon['Type 1'], df_pokemon['Type 2'])

plt.figure(figsize=(12, 10))
sns.heatmap(type_crosstab, annot=True, fmt='d', cmap='Blues')
plt.title('Count of Pokemon by Type 1 and Type 2')
plt.xlabel('Type 2')
plt.ylabel('Type 1')
plt.show()

print("\nVisualizing Combined Relationships (Attack, Defense, Legendary, Speed, Generation)")

generations_to_plot = [1, 2, 3]
df_subset_gen = df_pokemon[df_pokemon['Generation'].isin(generations_to_plot)]

g = sns.relplot(
    data=df_subset_gen,
    x="Attack", y="Defense",
    hue="Legendary", size="Speed",
    col="Generation", col_wrap=2, 
    height=4, aspect=1.1, palette="viridis", sizes=(50, 500) 
)
g.fig.suptitle('Attack vs. Defense by Generation (Colored by Legendary, Sized by Speed)', y=1.03) 
g.set_titles("Generation {col_name}") 
plt.tight_layout()
plt.show()
```

    Using 414 rows after dropping missing values.
    
    Generating visualizations for different variable relationship types...
    
    Visualizing Quantitative vs. Quantitative (Attack vs Defense)
    


    
![png](Markdown9_files/Markdown9_52_1.png)
    


    Visualizing Quantitative vs. Quantitative with Regression Line (Attack vs Defense)
    


    
![png](Markdown9_files/Markdown9_52_3.png)
    


    
    Visualizing Quantitative vs. Ranked (Total stat across Generations)
    


    
![png](Markdown9_files/Markdown9_52_5.png)
    



    
![png](Markdown9_files/Markdown9_52_6.png)
    


    
    Visualizing Quantitative vs. Qualitative (Attack stat by Type 1 - subset)
    


    
![png](Markdown9_files/Markdown9_52_8.png)
    


    
    Visualizing Ranked vs. Qualitative (Legendary status count by Generation)
    


    
![png](Markdown9_files/Markdown9_52_10.png)
    


    
    Visualizing Qualitative vs. Qualitative (Type 1 vs Type 2 - Heatmap)
    


    
![png](Markdown9_files/Markdown9_52_12.png)
    


    
    Visualizing Combined Relationships (Attack, Defense, Legendary, Speed, Generation)
    


    
![png](Markdown9_files/Markdown9_52_14.png)
    


## Heatmaps for qualitative correlations


```python
# git clone https://github.com/ayanatherate/dfcorrs.git
# cd dfcorrs 
# pip install -r requirements.txt

from dfcorrs.cramersvcorr import Cramers
cram=Cramers()
# cram.corr(df_pokemon)
cram.corr(df_pokemon, plot_htmp=True)

```



## Your turn!

Load the "sales" dataset and perform the bivariate analysis together with necessary plots. Remember about to run data preprocessing before the analysis.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2_contingency
import math

df = pd.read_excel("sales.xlsx")
df.head(5)

print("Initial DataFrame created:")
print(df)
print("\nDataFrame Info before preprocessing:")
df.info()

month_replacements = {'paź': 'Oct', 'lis': 'Nov'}
df['Date_Str'] = df['Date'].replace(month_replacements, regex=True)

df['Date'] = pd.to_datetime(df['Date_Str'], format='%d-%b-%y')
df = df.drop('Date_Str', axis=1) 

print("\nAfter Date Conversion:")
print(df[['Date', 'Sales']].head())
df.info()

quantitative_cols = ['Day_Temp', 'No_of_Customers', 'Sales']
categorical_cols = ['Store_Type', 'City_Type', 'Product_Quality']

print("\nMissing values before imputation:")
print(df.isnull().sum())

for col in quantitative_cols:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        print(f"Imputed missing values in '{col}' with mean: {mean_val:.2f}")
        
for col in categorical_cols:
     if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Imputed missing values in '{col}' with mode: {mode_val}")

for col in categorical_cols:
    df[col] = df[col].astype('category')


print("\nDataFrame after Imputation and Type Conversion:")
print(df)
df.info()
print("\nMissing values after imputation:")
print(df.isnull().sum())

print("\n" + "="*50)
print("!!! IMPORTANT NOTE ABOUT THIS ANALYSIS !!!")
print("The dataset provided contains only 12 rows. This is an extremely small sample size.")
print("Any statistical analysis (correlations, associations, significance tests) and visualizations")
print("derived from this data will NOT be reliable or representative of a larger dataset.")
print("The plots are shown for demonstration purposes only.")
print("Correlation/association values and perceived trends in plots are highly likely to be due to random chance.")
print("Interpretation of 'strength' based on common guidelines is not meaningful with N=12.")
print("="*50 + "\n")

print("\n--- Analyzing Quantitative vs. Quantitative ---")

quantitative_pairs = [
    ('Sales', 'No_of_Customers'),
    ('Sales', 'Day_Temp'),
    ('No_of_Customers', 'Day_Temp')
]

for x_var, y_var in quantitative_pairs:
    correlation, p_value = pearsonr(df[x_var], df[y_var])

    print(f"\nPearson Correlation between {x_var} and {y_var}:")
    print(f"  r = {correlation:.3f}")
    print(f"  p-value = {p_value:.4f}")

    plt.figure(figsize=(8, 5))
    sns.regplot(x=x_var, y=y_var, data=df, scatter_kws={'alpha': 0.7})
    plt.title(f'Scatterplot: {x_var} vs. {y_var} (r={correlation:.3f})')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

print("\n--- Analyzing Quantitative vs. Categorical ---")

quantitative_categorical_pairs = [
    ('Sales', 'Store_Type'),
    ('Sales', 'City_Type'),
    ('Sales', 'Product_Quality')
]

for quantitative_var, categorical_var in quantitative_categorical_pairs:
    print(f"\nAnalyzing {quantitative_var} by {categorical_var}:")

    print(f"  Mean {quantitative_var} by {categorical_var}:")
    print(df.groupby(categorical_var)[quantitative_var].mean())
    print(f"  Median {quantitative_var} by {categorical_var}:")
    print(df.groupby(categorical_var)[quantitative_var].median())

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=categorical_var, y=quantitative_var, data=df)
    plt.title(f'{quantitative_var} Distribution by {categorical_var} (Boxplot)')
    plt.xlabel(categorical_var)
    plt.ylabel(quantitative_var)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.violinplot(x=categorical_var, y=quantitative_var, data=df)
    plt.title(f'{quantitative_var} Distribution by {categorical_var} (Violinplot)')
    plt.xlabel(categorical_var)
    plt.ylabel(quantitative_var)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

print("\n--- Analyzing Categorical vs. Categorical ---")

categorical_pairs = [
    ('Store_Type', 'City_Type'),
    ('Store_Type', 'Product_Quality'),
    ('City_Type', 'Product_Quality')
]

def cramers_v(x, y):
    """Calculates Cramer's V statistic for two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    cramer_v = math.sqrt(phi2 / min(r-1, k-1)) if min(r, k) > 1 else 0.0 
    return cramer_v, p

for var1, var2 in categorical_pairs:
    print(f"\nAnalyzing association between {var1} and {var2}:")

    contingency_table = pd.crosstab(df[var1], df[var2])
    print("  Contingency Table:")
    print(contingency_table)

    cramer_v_stat, p_value_chi2 = cramers_v(df[var1], df[var2]) 

    print(f"  Cramer's V = {cramer_v_stat:.3f}")
    print(f"  Chi-squared p-value = {p_value_chi2:.4f}") 

    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Contingency Table Heatmap: {var1} vs. {var2}\nCramers V = {cramer_v_stat:.3f}')
    plt.xlabel(var2)
    plt.ylabel(var1)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(x=var1, hue=var2, data=df, palette='viridis')
    plt.title(f'Count of {var1} by {var2}')
    plt.xlabel(var1)
    plt.ylabel('Count')
    plt.legend(title=var2)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

print("\nAnalysis Complete.")
```

    Initial DataFrame created:
             Date  Store_Type  City_Type  Day_Temp  No_of_Customers   Sales  \
    0  2020-10-01           1          1      30.0            100.0  3112.0   
    1  2020-10-02           2          1      32.0            115.0  3682.0   
    2  2020-10-03           3          3      31.0              NaN  2774.0   
    3  2020-10-04           1          2      29.0            105.0  3182.0   
    4  2020-10-05           1          2      33.0            104.0  1368.0   
    5  2020-10-07           2          2       NaN              NaN     NaN   
    6  2020-11-24           2          3      26.0             90.0  4232.0   
    7  2020-11-25           3          3       NaN             96.0     NaN   
    8  2020-11-26           2          2      27.0            100.0  2356.0   
    9  2020-11-28           3          1       NaN              NaN     NaN   
    10 2020-11-29           1          1      23.0             94.0  1254.0   
    11 2020-11-30           1          1      22.0             91.0  4232.0   
    
       Product_Quality  
    0                A  
    1                A  
    2                A  
    3              NaN  
    4                B  
    5                B  
    6                C  
    7              NaN  
    8                B  
    9                A  
    10               A  
    11               A  
    
    DataFrame Info before preprocessing:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12 entries, 0 to 11
    Data columns (total 7 columns):
     #   Column           Non-Null Count  Dtype         
    ---  ------           --------------  -----         
     0   Date             12 non-null     datetime64[ns]
     1   Store_Type       12 non-null     int64         
     2   City_Type        12 non-null     int64         
     3   Day_Temp         9 non-null      float64       
     4   No_of_Customers  9 non-null      float64       
     5   Sales            9 non-null      float64       
     6   Product_Quality  10 non-null     object        
    dtypes: datetime64[ns](1), float64(3), int64(2), object(1)
    memory usage: 804.0+ bytes
    
    After Date Conversion:
            Date   Sales
    0 2020-10-01  3112.0
    1 2020-10-02  3682.0
    2 2020-10-03  2774.0
    3 2020-10-04  3182.0
    4 2020-10-05  1368.0
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12 entries, 0 to 11
    Data columns (total 7 columns):
     #   Column           Non-Null Count  Dtype         
    ---  ------           --------------  -----         
     0   Date             12 non-null     datetime64[ns]
     1   Store_Type       12 non-null     int64         
     2   City_Type        12 non-null     int64         
     3   Day_Temp         9 non-null      float64       
     4   No_of_Customers  9 non-null      float64       
     5   Sales            9 non-null      float64       
     6   Product_Quality  10 non-null     object        
    dtypes: datetime64[ns](1), float64(3), int64(2), object(1)
    memory usage: 804.0+ bytes
    
    Missing values before imputation:
    Date               0
    Store_Type         0
    City_Type          0
    Day_Temp           3
    No_of_Customers    3
    Sales              3
    Product_Quality    2
    dtype: int64
    Imputed missing values in 'Day_Temp' with mean: 28.11
    Imputed missing values in 'No_of_Customers' with mean: 99.44
    Imputed missing values in 'Sales' with mean: 2910.22
    Imputed missing values in 'Product_Quality' with mode: A
    
    DataFrame after Imputation and Type Conversion:
             Date Store_Type City_Type   Day_Temp  No_of_Customers        Sales  \
    0  2020-10-01          1         1  30.000000       100.000000  3112.000000   
    1  2020-10-02          2         1  32.000000       115.000000  3682.000000   
    2  2020-10-03          3         3  31.000000        99.444444  2774.000000   
    3  2020-10-04          1         2  29.000000       105.000000  3182.000000   
    4  2020-10-05          1         2  33.000000       104.000000  1368.000000   
    5  2020-10-07          2         2  28.111111        99.444444  2910.222222   
    6  2020-11-24          2         3  26.000000        90.000000  4232.000000   
    7  2020-11-25          3         3  28.111111        96.000000  2910.222222   
    8  2020-11-26          2         2  27.000000       100.000000  2356.000000   
    9  2020-11-28          3         1  28.111111        99.444444  2910.222222   
    10 2020-11-29          1         1  23.000000        94.000000  1254.000000   
    11 2020-11-30          1         1  22.000000        91.000000  4232.000000   
    
       Product_Quality  
    0                A  
    1                A  
    2                A  
    3                A  
    4                B  
    5                B  
    6                C  
    7                A  
    8                B  
    9                A  
    10               A  
    11               A  
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12 entries, 0 to 11
    Data columns (total 7 columns):
     #   Column           Non-Null Count  Dtype         
    ---  ------           --------------  -----         
     0   Date             12 non-null     datetime64[ns]
     1   Store_Type       12 non-null     category      
     2   City_Type        12 non-null     category      
     3   Day_Temp         12 non-null     float64       
     4   No_of_Customers  12 non-null     float64       
     5   Sales            12 non-null     float64       
     6   Product_Quality  12 non-null     category      
    dtypes: category(3), datetime64[ns](1), float64(3)
    memory usage: 948.0 bytes
    
    Missing values after imputation:
    Date               0
    Store_Type         0
    City_Type          0
    Day_Temp           0
    No_of_Customers    0
    Sales              0
    Product_Quality    0
    dtype: int64
    
    ==================================================
    !!! IMPORTANT NOTE ABOUT THIS ANALYSIS !!!
    The dataset provided contains only 12 rows. This is an extremely small sample size.
    Any statistical analysis (correlations, associations, significance tests) and visualizations
    derived from this data will NOT be reliable or representative of a larger dataset.
    The plots are shown for demonstration purposes only.
    Correlation/association values and perceived trends in plots are highly likely to be due to random chance.
    Interpretation of 'strength' based on common guidelines is not meaningful with N=12.
    ==================================================
    
    
    --- Analyzing Quantitative vs. Quantitative ---
    
    Pearson Correlation between Sales and No_of_Customers:
      r = -0.120
      p-value = 0.7093
    


    
![png](Markdown9_files/Markdown9_56_1.png)
    


    
    Pearson Correlation between Sales and Day_Temp:
      r = -0.178
      p-value = 0.5789
    


    
![png](Markdown9_files/Markdown9_56_3.png)
    


    
    Pearson Correlation between No_of_Customers and Day_Temp:
      r = 0.766
      p-value = 0.0037
    


    
![png](Markdown9_files/Markdown9_56_5.png)
    


    
    --- Analyzing Quantitative vs. Categorical ---
    
    Analyzing Sales by Store_Type:
      Mean Sales by Store_Type:
    Store_Type
    1    2629.600000
    2    3295.055556
    3    2864.814815
    Name: Sales, dtype: float64
      Median Sales by Store_Type:
    Store_Type
    1    3112.000000
    2    3296.111111
    3    2910.222222
    Name: Sales, dtype: float64
    


    
![png](Markdown9_files/Markdown9_56_7.png)
    



    
![png](Markdown9_files/Markdown9_56_8.png)
    


    
    Analyzing Sales by City_Type:
      Mean Sales by City_Type:
    City_Type
    1    3038.044444
    2    2454.055556
    3    3305.407407
    Name: Sales, dtype: float64
      Median Sales by City_Type:
    City_Type
    1    3112.000000
    2    2633.111111
    3    2910.222222
    Name: Sales, dtype: float64
    


    
![png](Markdown9_files/Markdown9_56_10.png)
    



    
![png](Markdown9_files/Markdown9_56_11.png)
    


    
    Analyzing Sales by Product_Quality:
      Mean Sales by Product_Quality:
    Product_Quality
    A    3007.055556
    B    2211.407407
    C    4232.000000
    Name: Sales, dtype: float64
      Median Sales by Product_Quality:
    Product_Quality
    A    3011.111111
    B    2356.000000
    C    4232.000000
    Name: Sales, dtype: float64
    


    
![png](Markdown9_files/Markdown9_56_13.png)
    



    
![png](Markdown9_files/Markdown9_56_14.png)
    


    
    --- Analyzing Categorical vs. Categorical ---
    
    Analyzing association between Store_Type and City_Type:
      Contingency Table:
    City_Type   1  2  3
    Store_Type         
    1           3  2  0
    2           1  2  1
    3           1  0  2
      Cramer's V = 0.477
      Chi-squared p-value = 0.2439
    


    
![png](Markdown9_files/Markdown9_56_16.png)
    



    
![png](Markdown9_files/Markdown9_56_17.png)
    


    
    Analyzing association between Store_Type and Product_Quality:
      Contingency Table:
    Product_Quality  A  B  C
    Store_Type              
    1                4  1  0
    2                1  2  1
    3                3  0  0
      Cramer's V = 0.478
      Chi-squared p-value = 0.2419
    


    
![png](Markdown9_files/Markdown9_56_19.png)
    



    
![png](Markdown9_files/Markdown9_56_20.png)
    


    
    Analyzing association between City_Type and Product_Quality:
      Contingency Table:
    Product_Quality  A  B  C
    City_Type               
    1                5  0  0
    2                1  3  0
    3                2  0  1
      Cramer's V = 0.673
      Chi-squared p-value = 0.0280
    


    
![png](Markdown9_files/Markdown9_56_22.png)
    



    
![png](Markdown9_files/Markdown9_56_23.png)
    


    
    Analysis Complete.
    

# Summary

There are many ways to *describe* our data:

- Measure **central tendency**.

- Measure its **variability**; **skewness** and **kurtosis**.

- Measure what **correlations** our data have.

All of these are **useful** and all of them are also **exploratory data analysis** (EDA).
