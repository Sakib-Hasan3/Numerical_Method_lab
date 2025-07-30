# Bisection Method Code Explanation (বাইসেকশন পদ্ধতির কোড ব্যাখ্যা)

## Cell 1: Library Import (লাইব্রেরি ইমপোর্ট)

```python
import numpy as np
import matplotlib.pyplot as plt
```

**বাংলা ব্যাখ্যা:**
- `import numpy as np` - NumPy লাইব্রেরি ইমপোর্ট করা হয়েছে যা গাণিতিক গণনা এবং array operations এর জন্য ব্যবহৃত হয়
- `import matplotlib.pyplot as plt` - Matplotlib এর pyplot মডিউল ইমপোর্ট করা হয়েছে যা গ্রাফ এবং চার্ট তৈরি করার জন্য ব্যবহৃত হয়

## Cell 2: Function Definition (ফাংশন সংজ্ঞা)

```python
def f(x):
    return x**3 - 4*x - 9 
```

**বাংলা ব্যাখ্যা:**
- `def f(x):` - একটি ফাংশন সংজ্ঞায়িত করা হয়েছে যার নাম `f` এবং এটি `x` নামক একটি parameter নেয়
- `return x**3 - 4*x - 9` - এই ফাংশন f(x) = x³ - 4x - 9 গাণিতিক সমীকরণ return করে। এটিই সেই সমীকরণ যার root বা মূল আমরা খুঁজে বের করতে চাই

## Cell 3: Bisection Algorithm (বাইসেকশন অ্যালগরিদম)

```python
def bisection(a, b):
    for i in range(20):   
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c
```

**বাংলা ব্যাখ্যা:**
- `def bisection(a, b):` - বাইসেকশন পদ্ধতির জন্য একটি ফাংশন সংজ্ঞায়িত করা হয়েছে যা দুটি parameter `a` এবং `b` নেয় (interval এর শুরু ও শেষ বিন্দু)
- `for i in range(20):` - ২০ বার লুপ চালানো হবে (সর্বোচ্চ ২০টি iteration)
- `c = (a + b) / 2` - interval এর মধ্যবিন্দু `c` বের করা হয়েছে
- `if f(c) == 0:` - যদি মধ্যবিন্দুতে ফাংশনের মান শূন্য হয়, তাহলে এটিই আমাদের root
- `return c` - root পাওয়া গেলে তা return করা হয়
- `elif f(a) * f(c) < 0:` - যদি `f(a)` এবং `f(c)` এর গুণফল ঋণাত্মক হয়, তাহলে root টি `a` এবং `c` এর মধ্যে রয়েছে
- `b = c` - তাই নতুন interval এর শেষ বিন্দু `c` করা হয়
- `else:` - অন্যথায় root টি `c` এবং `b` এর মধ্যে রয়েছে
- `a = c` - তাই নতুন interval এর শুরু বিন্দু `c` করা হয়
- `return c` - সর্বশেষ পাওয়া মধ্যবিন্দু return করা হয় (approximation হিসেবে)

## Cell 4: Execution and Result (কার্যকরকরণ এবং ফলাফল)

```python
a, b = 2, 3
root = bisection(a, b)
print("Root ≈", root)
```

**বাংলা ব্যাখ্যা:**
- `a, b = 2, 3` - প্রাথমিক interval নির্ধারণ করা হয়েছে যেখানে `a = 2` এবং `b = 3`
- `root = bisection(a, b)` - বাইসেকশন ফাংশন কল করে root বের করা হয়েছে এবং `root` ভেরিয়েবলে সংরক্ষণ করা হয়েছে
- `print("Root ≈", root)` - পাওয়া root এর আনুমানিক মান প্রিন্ট করা হয়েছে

## Cell 5: Graph Visualization (গ্রাফ ভিজুয়ালাইজেশন)

```python
x = np.linspace(1, 4, 400)
y = f(x)
plt.plot(x, y, label="f(x)")
plt.axhline(0, color="black")
plt.axvline(root, color="red", linestyle="--", label=f"Root ≈ {root:.4f}")
plt.scatter(root, f(root), color="red")
plt.legend()
plt.title("Bisection Method (Easy Code)")
plt.show()
```

**বাংলা ব্যাখ্যা:**
- `x = np.linspace(1, 4, 400)` - ১ থেকে ৪ পর্যন্ত ৪০০টি সমান দূরত্বে বিভক্ত x-axis এর মান তৈরি করা হয়েছে
- `y = f(x)` - প্রতিটি x মানের জন্য corresponding y মান (ফাংশনের মান) বের করা হয়েছে
- `plt.plot(x, y, label="f(x)")` - x এবং y মান ব্যবহার করে ফাংশনের গ্রাফ আঁকা হয়েছে
- `plt.axhline(0, color="black")` - x-axis এ একটি কালো রেখা আঁকা হয়েছে (y = 0 line)
- `plt.axvline(root, color="red", linestyle="--", label=f"Root ≈ {root:.4f}")` - root এর স্থানে একটি লাল ডট ডট লাইন আঁকা হয়েছে
- `plt.scatter(root, f(root), color="red")` - root এর স্থানে একটি লাল বিন্দু দেখানো হয়েছে
- `plt.legend()` - গ্রাফে legend (চিহ্নের ব্যাখ্যা) দেখানো হয়েছে
- `plt.title("Bisection Method (Easy Code)")` - গ্রাফের শিরোনাম সেট করা হয়েছে
- `plt.show()` - গ্রাফটি প্রদর্শন করা হয়েছে

## বাইসেকশন পদ্ধতির মূল নীতি:
এই পদ্ধতিতে আমরা একটি interval নিয়ে তার মধ্যবিন্দু বের করি। যদি মধ্যবিন্দুতে ফাংশনের মান শূন্য না হয়, তাহলে আমরা interval এর যে অর্ধেকে root রয়েছে সেটি নিয়ে আবার একই প্রক্রিয়া চালাই। এভাবে interval ক্রমান্বয়ে ছোট হতে থাকে এবং আমরা root এর কাছাকাছি পৌঁছাই।
