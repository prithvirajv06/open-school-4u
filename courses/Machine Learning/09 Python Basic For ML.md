# Python Basics for Machine Learning

## Introduction: Why Python for ML?

**Python is the #1 language for Machine Learning. Here's why:**

✅ **Easy to learn** - Reads like English
✅ **Powerful libraries** - NumPy, Pandas, Scikit-learn, TensorFlow
✅ **Huge community** - Lots of help available
✅ **Industry standard** - Most ML jobs use Python

**Good news:** You don't need to master Python to start ML!

---

## Installation and Setup

### Installing Python

**Step 1: Download Python**
- Go to python.org
- Download latest version (3.8 or higher)
- During installation: ✓ Check "Add Python to PATH"

**Step 2: Verify Installation**
```bash
# Open terminal/command prompt
python --version
# Should show: Python 3.x.x
```

**Step 3: Install Essential Libraries**
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

---

### Your First Python Program

**Create a file: `hello.py`**
```python
print("Hello, Machine Learning!")
```

**Run it:**
```bash
python hello.py
```

**Output:**
```
Hello, Machine Learning!
```

🎉 **Congratulations! You're a Python programmer!**

---

## Python Basics (Everything You Need for ML)

### 1. Variables (Storing Information)

**Think of variables as labeled boxes that hold data**

```python
# Numbers
age = 25
price = 50.99
temperature = -5

# Text (Strings)
name = "John"
city = "Mumbai"

# Boolean (True/False)
is_student = True
has_graduated = False

# Use variables
print(name)  # John
print(age)   # 25
```

**Real ML Example:**
```python
# House price prediction
bedrooms = 3
bathrooms = 2
sqft = 1500
price = 7500000  # ₹75 lakhs

print(f"House with {bedrooms} bedrooms costs ₹{price}")
# Output: House with 3 bedrooms costs ₹7500000
```

---

### 2. Data Types

**Python automatically figures out what type of data you have**

```python
# Integer (whole numbers)
age = 25

# Float (decimal numbers)
price = 99.99

# String (text)
name = "Alice"

# Boolean (True/False)
is_spam = False

# Check type
print(type(age))    # <class 'int'>
print(type(price))  # <class 'float'>
print(type(name))   # <class 'str'>
```

**Type Conversion:**
```python
# Convert string to number
age_text = "25"
age_number = int(age_text)

# Convert number to string
score = 95
score_text = str(score)

# Convert to float
price = float("99.99")
```

---

### 3. Lists (Collections of Data)

**Lists = Ordered collection of items (like arrays)**

```python
# Create a list
prices = [50, 60, 75, 80, 90]
names = ["Alice", "Bob", "Charlie"]
mixed = [1, "hello", 3.14, True]

# Access items (indexing starts at 0!)
print(prices[0])   # 50 (first item)
print(prices[2])   # 75 (third item)
print(prices[-1])  # 90 (last item)

# Modify items
prices[0] = 55
print(prices)  # [55, 60, 75, 80, 90]

# Add items
prices.append(100)
print(prices)  # [55, 60, 75, 80, 90, 100]

# Remove items
prices.remove(55)
print(prices)  # [60, 75, 80, 90, 100]

# Length
print(len(prices))  # 5
```

**Real ML Example:**
```python
# Features for house price prediction
house_features = [3, 2, 1500, 10]  # bedrooms, bathrooms, sqft, age
house_prices = [50, 60, 75, 80, 90]  # lakhs

print(f"Average price: ₹{sum(house_prices)/len(house_prices)} lakhs")
# Output: Average price: ₹71.0 lakhs
```

---

### 4. Loops (Repeating Actions)

**Loops = Do something multiple times**

#### For Loop (when you know how many times)

```python
# Loop through a list
prices = [50, 60, 75, 80]

for price in prices:
    print(f"₹{price} lakhs")

# Output:
# ₹50 lakhs
# ₹60 lakhs
# ₹75 lakhs
# ₹80 lakhs
```

```python
# Loop with range (0 to 4)
for i in range(5):
    print(i)
# Output: 0, 1, 2, 3, 4
```

**Real ML Example:**
```python
# Calculate squared errors
predictions = [50, 60, 75]
actuals = [52, 58, 80]

for i in range(len(predictions)):
    error = predictions[i] - actuals[i]
    squared_error = error ** 2
    print(f"Squared Error: {squared_error}")

# Output:
# Squared Error: 4
# Squared Error: 4
# Squared Error: 25
```

#### While Loop (when you don't know how many times)

```python
count = 0
while count < 5:
    print(count)
    count += 1  # Same as count = count + 1

# Output: 0, 1, 2, 3, 4
```

---

### 5. Conditional Statements (Making Decisions)

**If-Else = Make decisions in code**

```python
age = 25

if age >= 18:
    print("Adult")
else:
    print("Minor")
# Output: Adult
```

```python
score = 85

if score >= 90:
    print("A")
elif score >= 80:
    print("B")
elif score >= 70:
    print("C")
else:
    print("F")
# Output: B
```

**Real ML Example:**
```python
# Classify prediction confidence
confidence = 0.85

if confidence >= 0.9:
    print("Very confident")
elif confidence >= 0.7:
    print("Confident")
elif confidence >= 0.5:
    print("Uncertain")
else:
    print("Very uncertain")
# Output: Confident
```

---

### 6. Functions (Reusable Code)

**Functions = Recipe for code (write once, use many times)**

```python
# Define a function
def greet(name):
    print(f"Hello, {name}!")

# Call the function
greet("Alice")  # Hello, Alice!
greet("Bob")    # Hello, Bob!
```

**With Return Values:**
```python
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average

# Use the function
prices = [50, 60, 75, 80]
avg = calculate_average(prices)
print(f"Average: {avg}")  # Average: 66.25
```

**Real ML Example:**
```python
def calculate_accuracy(predictions, actuals):
    correct = 0
    total = len(predictions)
    
    for i in range(total):
        if predictions[i] == actuals[i]:
            correct += 1
    
    accuracy = correct / total
    return accuracy

# Use function
pred = [1, 0, 1, 1, 0]
true = [1, 0, 1, 0, 0]
acc = calculate_accuracy(pred, true)
print(f"Accuracy: {acc}")  # Accuracy: 0.8 (80%)
```

---

### 7. Dictionaries (Key-Value Pairs)

**Dictionaries = Like a real dictionary (word → definition)**

```python
# Create a dictionary
person = {
    "name": "Alice",
    "age": 25,
    "city": "Mumbai"
}

# Access values
print(person["name"])  # Alice
print(person["age"])   # 25

# Add new key-value
person["salary"] = 500000

# Modify value
person["age"] = 26

# Loop through dictionary
for key, value in person.items():
    print(f"{key}: {value}")

# Output:
# name: Alice
# age: 26
# city: Mumbai
# salary: 500000
```

**Real ML Example:**
```python
# Store model performance
model_scores = {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.88,
    "f1_score": 0.85
}

print(f"Model Accuracy: {model_scores['accuracy']}")
# Output: Model Accuracy: 0.85
```

---

### 8. String Operations (Text Manipulation)

**Strings = Text data (very important for NLP!)**

```python
# Create string
message = "Hello, Machine Learning!"

# Length
print(len(message))  # 25

# Uppercase/Lowercase
print(message.upper())  # HELLO, MACHINE LEARNING!
print(message.lower())  # hello, machine learning!

# Check if contains
print("Learning" in message)  # True
print("Python" in message)    # False

# Split into words
words = message.split()
print(words)  # ['Hello,', 'Machine', 'Learning!']

# Replace
new_message = message.replace("Hello", "Welcome")
print(new_message)  # Welcome, Machine Learning!

# Strip whitespace
text = "  hello  "
print(text.strip())  # "hello"
```

**Real ML Example (Text Processing):**
```python
# Email spam detection
email = "FREE MONEY!!! Click now!!!"

# Convert to lowercase
email_lower = email.lower()

# Check for spam words
spam_words = ["free", "money", "click", "winner"]
spam_score = 0

for word in spam_words:
    if word in email_lower:
        spam_score += 1

print(f"Spam Score: {spam_score}/4")  # Spam Score: 3/4

if spam_score >= 2:
    print("Likely spam!")
# Output: Likely spam!
```

---

### 9. List Comprehension (Pythonic Way)

**List Comprehension = Create lists in one line (elegant!)**

**Traditional way:**
```python
squares = []
for i in range(5):
    squares.append(i ** 2)
print(squares)  # [0, 1, 4, 9, 16]
```

**List comprehension (better!):**
```python
squares = [i ** 2 for i in range(5)]
print(squares)  # [0, 1, 4, 9, 16]
```

**With condition:**
```python
# Only even squares
even_squares = [i ** 2 for i in range(10) if i % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]
```

**Real ML Example:**
```python
# Calculate errors for all predictions
predictions = [50, 60, 75, 80]
actuals = [52, 58, 77, 78]

errors = [pred - actual for pred, actual in zip(predictions, actuals)]
print(errors)  # [-2, 2, -2, 2]

# Absolute errors
abs_errors = [abs(error) for error in errors]
print(abs_errors)  # [2, 2, 2, 2]
```

---

### 10. File Operations (Reading/Writing Data)

**Reading Data from Files:**

```python
# Read entire file
with open('data.txt', 'r') as file:
    content = file.read()
    print(content)

# Read line by line
with open('data.txt', 'r') as file:
    for line in file:
        print(line.strip())
```

**Writing Data to Files:**

```python
# Write to file (overwrites existing)
with open('output.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("Machine Learning is fun!")

# Append to file
with open('output.txt', 'a') as file:
    file.write("\nNew line added!")
```

**Real ML Example (Reading CSV):**
```python
# Read CSV file
with open('houses.csv', 'r') as file:
    header = file.readline().strip().split(',')
    print(header)  # ['bedrooms', 'bathrooms', 'price']
    
    for line in file:
        data = line.strip().split(',')
        bedrooms = int(data[0])
        bathrooms = int(data[1])
        price = float(data[2])
        print(f"{bedrooms} bed, {bathrooms} bath: ₹{price}L")
```

---

## Essential Python Libraries for ML

### 1. NumPy (Numerical Computing)

**Why NumPy?** Fast array operations (much faster than Python lists!)

```python
import numpy as np

# Create array
prices = np.array([50, 60, 75, 80, 90])

# Basic operations
print(prices.mean())  # 71.0 (average)
print(prices.std())   # 14.14 (standard deviation)
print(prices.min())   # 50
print(prices.max())   # 90

# Mathematical operations (vectorized - very fast!)
prices_in_crores = prices / 100
print(prices_in_crores)  # [0.5, 0.6, 0.75, 0.8, 0.9]

# Element-wise operations
doubled = prices * 2
print(doubled)  # [100, 120, 150, 160, 180]

# Boolean indexing
expensive = prices[prices > 70]
print(expensive)  # [75, 80, 90]
```

**2D Arrays (Matrices):**
```python
# Create 2D array (rows x columns)
data = np.array([
    [3, 2, 1500],  # House 1: bedrooms, bathrooms, sqft
    [4, 3, 2000],  # House 2
    [2, 1, 1000]   # House 3
])

print(data.shape)  # (3, 3) - 3 rows, 3 columns

# Access elements
print(data[0, 0])  # 3 (first house, bedrooms)
print(data[:, 0])  # [3, 4, 2] (all bedrooms)

# Statistics
print(data.mean(axis=0))  # Mean of each column
# Output: [3. 2. 1500.] (avg bedrooms, bathrooms, sqft)
```

---

### 2. Pandas (Data Manipulation)

**Why Pandas?** Work with tables (like Excel in Python!)

```python
import pandas as pd

# Create DataFrame (table)
data = {
    'bedrooms': [3, 4, 2, 3],
    'bathrooms': [2, 3, 1, 2],
    'sqft': [1500, 2000, 1000, 1800],
    'price': [75, 95, 50, 80]
}

df = pd.DataFrame(data)
print(df)

# Output:
#    bedrooms  bathrooms  sqft  price
# 0         3          2  1500     75
# 1         4          3  2000     95
# 2         2          1  1000     50
# 3         3          2  1800     80
```

**Common Operations:**

```python
# View first few rows
print(df.head(2))

# Summary statistics
print(df.describe())

# Access column
print(df['price'])

# Filter rows
expensive = df[df['price'] > 70]
print(expensive)

# Sort
sorted_df = df.sort_values('price', ascending=False)
print(sorted_df)

# Add new column
df['price_per_sqft'] = df['price'] / df['sqft'] * 100000
print(df)

# Read CSV
df = pd.read_csv('houses.csv')

# Write CSV
df.to_csv('output.csv', index=False)
```

---

### 3. Matplotlib (Visualization)

**Why Matplotlib?** Create charts and graphs

```python
import matplotlib.pyplot as plt

# Simple line plot
x = [1, 2, 3, 4, 5]
y = [50, 60, 75, 80, 90]

plt.plot(x, y)
plt.xlabel('House Number')
plt.ylabel('Price (₹ Lakhs)')
plt.title('House Prices')
plt.show()
```

**Scatter Plot:**
```python
# House size vs price
sizes = [1000, 1500, 2000, 2500, 3000]
prices = [50, 75, 95, 110, 130]

plt.scatter(sizes, prices)
plt.xlabel('Size (sq ft)')
plt.ylabel('Price (₹ Lakhs)')
plt.title('House Size vs Price')
plt.show()
```

**Histogram:**
```python
# Distribution of prices
prices = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

plt.hist(prices, bins=5)
plt.xlabel('Price (₹ Lakhs)')
plt.ylabel('Frequency')
plt.title('Price Distribution')
plt.show()
```

---

## Python for ML: Putting It All Together

### Complete Example: Simple House Price Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Create sample data
data = {
    'bedrooms': [2, 3, 3, 4, 4, 2, 3, 5],
    'bathrooms': [1, 2, 2, 3, 2, 1, 2, 4],
    'sqft': [1000, 1500, 1400, 2000, 1800, 900, 1600, 2500],
    'age': [10, 5, 8, 2, 6, 15, 7, 1],
    'price': [50, 75, 70, 95, 85, 45, 78, 120]
}

df = pd.DataFrame(data)

# 2. Explore data
print("=== Data Overview ===")
print(df.head())
print("\n=== Statistics ===")
print(df.describe())

# 3. Calculate insights
avg_price = df['price'].mean()
print(f"\nAverage Price: ₹{avg_price} lakhs")

expensive = df[df['price'] > avg_price]
print(f"\nExpensive houses (>{avg_price}L): {len(expensive)}")

# 4. Add calculated column
df['price_per_sqft'] = df['price'] / df['sqft'] * 100000

# 5. Find patterns
correlation = df['sqft'].corr(df['price'])
print(f"\nCorrelation between size and price: {correlation:.2f}")

# 6. Visualize
plt.figure(figsize=(12, 4))

# Plot 1: Size vs Price
plt.subplot(1, 3, 1)
plt.scatter(df['sqft'], df['price'])
plt.xlabel('Size (sq ft)')
plt.ylabel('Price (₹ Lakhs)')
plt.title('Size vs Price')

# Plot 2: Price Distribution
plt.subplot(1, 3, 2)
plt.hist(df['price'], bins=5, edgecolor='black')
plt.xlabel('Price (₹ Lakhs)')
plt.ylabel('Count')
plt.title('Price Distribution')

# Plot 3: Bedrooms vs Price
plt.subplot(1, 3, 3)
bedroom_avg = df.groupby('bedrooms')['price'].mean()
plt.bar(bedroom_avg.index, bedroom_avg.values)
plt.xlabel('Bedrooms')
plt.ylabel('Avg Price (₹ Lakhs)')
plt.title('Bedrooms vs Avg Price')

plt.tight_layout()
plt.show()

# 7. Save results
df.to_csv('analyzed_houses.csv', index=False)
print("\nResults saved to 'analyzed_houses.csv'")
```

---

## Common Python Errors and Solutions

### 1. IndentationError

```python
# ❌ Wrong (inconsistent indentation)
def greet():
print("Hello")  # Error!

# ✅ Correct (use 4 spaces or Tab consistently)
def greet():
    print("Hello")
```

### 2. NameError

```python
# ❌ Wrong (variable not defined)
print(price)  # NameError: name 'price' is not defined

# ✅ Correct (define variable first)
price = 50
print(price)
```

### 3. IndexError

```python
# ❌ Wrong (index out of range)
prices = [50, 60, 75]
print(prices[5])  # IndexError: list index out of range

# ✅ Correct (use valid index)
print(prices[0])  # 50
print(prices[-1])  # 75 (last item)
```

### 4. TypeError

```python
# ❌ Wrong (mixing incompatible types)
age = "25"
next_year = age + 1  # TypeError

# ✅ Correct (convert types)
age = int("25")
next_year = age + 1
print(next_year)  # 26
```

### 5. KeyError

```python
# ❌ Wrong (key doesn't exist)
person = {"name": "Alice", "age": 25}
print(person["salary"])  # KeyError: 'salary'

# ✅ Correct (check if key exists or use get)
if "salary" in person:
    print(person["salary"])
else:
    print("Salary not found")

# Or use get with default
print(person.get("salary", 0))  # 0 (default value)
```

---

## Python Best Practices for ML

### 1. Use Meaningful Variable Names

```python
# ❌ Bad
a = 3
b = 2
c = 1500
d = 75

# ✅ Good
bedrooms = 3
bathrooms = 2
sqft = 1500
price = 75
```

### 2. Add Comments

```python
# Calculate average house price
prices = [50, 60, 75, 80]
average = sum(prices) / len(prices)  # Simple mean calculation
print(f"Average: ₹{average}L")
```

### 3. Use Functions for Reusable Code

```python
# ✅ Good - Reusable
def calculate_rmse(predictions, actuals):
    """Calculate Root Mean Squared Error"""
    errors = [(p - a) ** 2 for p, a in zip(predictions, actuals)]
    mse = sum(errors) / len(errors)
    rmse = mse ** 0.5
    return rmse

# Use multiple times
rmse1 = calculate_rmse([50, 60], [52, 58])
rmse2 = calculate_rmse([75, 80], [77, 78])
```

### 4. Handle Errors Gracefully

```python
# Use try-except for error handling
try:
    price = int(input("Enter price: "))
    print(f"Price: ₹{price}")
except ValueError:
    print("Invalid input! Please enter a number.")
```

---

## Quick Reference Cheat Sheet

### Variables & Data Types
```python
number = 42              # Integer
decimal = 3.14          # Float
text = "hello"          # String
flag = True             # Boolean
```

### Lists
```python
items = [1, 2, 3]
items.append(4)         # Add item
items[0]                # Access first
len(items)              # Length
```

### Loops
```python
for item in items:      # For loop
    print(item)

while condition:        # While loop
    # do something
```

### Conditionals
```python
if condition:
    # do this
elif other_condition:
    # do that
else:
    # do this otherwise
```

### Functions
```python
def function_name(param):
    # do something
    return result
```

### Dictionaries
```python
data = {"key": "value"}
data["key"]             # Access
data["new"] = "value"   # Add
```

### NumPy
```python
import numpy as np
arr = np.array([1, 2, 3])
arr.mean()              # Average
```

### Pandas
```python
import pandas as pd
df = pd.read_csv('file.csv')
df.head()               # First rows
df['column']            # Access column
```

### Matplotlib
```python
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
```

---

## Practice Exercises

### Exercise 1: Basic Calculations
```python
# Calculate and print:
# 1. Sum of [10, 20, 30, 40, 50]
# 2. Average of the same list
# 3. Maximum value
# 4. Minimum value
```

### Exercise 2: Price Analysis
```python
# Given house prices: [50, 60, 75, 80, 90, 95]
# 1. How many houses cost more than ₹70L?
# 2. What's the average price?
# 3. Create a new list with 10% discount on all prices
```

### Exercise 3: Text Processing
```python
# Given email: "FREE MONEY!!! Win now!!!"
# 1. Convert to lowercase
# 2. Count how many times '!' appears
# 3. Check if it contains any spam words: ['free', 'win', 'money']
```

---

## Next Steps

**You now know Python basics for ML!**

**To practice:**
1. Complete exercises above
2. Type code examples yourself (don't just read!)
3. Modify examples to try new things
4. Start with simple ML project

**Remember:**
- Practice daily (even 15 minutes!)
- Google errors (Stack Overflow is your friend)
- Don't memorize - understand the logic
- Learn by doing, not just reading

**You're ready to start using Python for Machine Learning!** 🚀