\#Exercise 01 - Data Preprocessing



\## Goal



The goal of this exercise is to practice basic data preprocessing techniques using Pandas and Numpy.



The exercise focuses on Handling missing values and creating a new feature from an existing numerical column.



\#Dataset



The dataset contains customer transaction data with the following columns:



| Column | Description |

| ------ | ----------- |

| 'customer\_id' | Unique customer identifier |

| 'age' | Customer age |

| 'Country' | Customer Country |

| 'purchase' | Purchase amount |

| 'is\_fraud' | Fraud indicator ('0' or '1') |



\## Tasks



The 'preprocess()' function performs the following operations:



1. Replace missing values in 'age' with the median age.
2. Replace missing values in 'country' with '"UNKNOWN"'.
3. Create a new 'purchase\_log' feature using the natural logarithm of 'purchase'.
4. Return the preprocessed DataFrame.



\## Technologies



* Python 3
* Pandas
* Numpy







&#x20;







