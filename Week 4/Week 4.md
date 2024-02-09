There are a number of collection types in Python. While types such as int and str hold a single value, collection
types hold multiple values.
Lists
The list type is probably the most commonly used collection type in Python. Despite its name, a list is more like an
array in other languages, mostly JavaScript. In Python, a list is merely an ordered collection of valid Python values. A
list can be created by enclosing values, separated by commas, in square brackets:
GoalKicker.com – Python® Notes for Professionals 16
int_list = [1, 2, 3]
string_list = ['abc', 'defghi']
A list can be empty:
empty_list = []
The elements of a list are not restricted to a single data type, which makes sense given that Python is a dynamic
language:
mixed_list = [1, 'abc', True, 2.34, None]
A list can contain another list as its element:
nested_list = [['a', 'b', 'c'], [1, 2, 3]]
The elements of a list can be accessed via an index, or numeric representation of their position. Lists in Python are
zero-indexed meaning that the first element in the list is at index 0, the second element is at index 1 and so on:
names = ['Alice', 'Bob', 'Craig', 'Diana', 'Eric']
print(names[0]) # Alice
print(names[2]) # Craig
Indices can also be negative which means counting from the end of the list (-1 being the index of the last element).
So, using the list from the above example:
print(names[-1]) # Eric
print(names[-4]) # Bob
Lists are mutable, so you can change the values in a list:
names[0] = 'Ann'
print(names)
# Outputs ['Ann', 'Bob', 'Craig', 'Diana', 'Eric']
Besides, it is possible to add and/or remove elements from a list:
Append object to end of list with L.append(object), returns None.
names = ['Alice', 'Bob', 'Craig', 'Diana', 'Eric']
names.append("Sia")
print(names)
# Outputs ['Alice', 'Bob', 'Craig', 'Diana', 'Eric', 'Sia']
Add a new element to list at a specific index. L.insert(index, object)
names.insert(1, "Nikki")
print(names)
# Outputs ['Alice', 'Nikki', 'Bob', 'Craig', 'Diana', 'Eric', 'Sia']
Remove the first occurrence of a value with L.remove(value), returns None
names.remove("Bob")
print(names) # Outputs ['Alice', 'Nikki', 'Craig', 'Diana', 'Eric', 'Sia']
GoalKicker.com – Python® Notes for Professionals 17
Get the index in the list of the first item whose value is x. It will show an error if there is no such item.
name.index("Alice")
0
Count length of list
len(names)
6
count occurrence of any item in list
a = [1, 1, 1, 2, 3, 4]
a.count(1)
3
Reverse the list
a.reverse()
[4, 3, 2, 1, 1, 1]
# or
a[::-1]
[4, 3, 2, 1, 1, 1]
Remove and return item at index (defaults to the last item) with L.pop([index]), returns the item
names.pop() # Outputs 'Sia'
You can iterate over the list elements like below:
for element in my_list:
print (element)
Tuples
A tuple is similar to a list except that it is fixed-length and immutable. So the values in the tuple cannot be changed
nor the values be added to or removed from the tuple. Tuples are commonly used for small collections of values
that will not need to change, such as an IP address and port. Tuples are represented with parentheses instead of
square brackets:
ip_address = ('10.20.30.40', 8080)
The same indexing rules for lists also apply to tuples. Tuples can also be nested and the values can be any valid
Python valid.
A tuple with only one member must be defined (note the comma) this way:
one_member_tuple = ('Only member',)
or
one_member_tuple = 'Only member', # No brackets
or just using tuple syntax
GoalKicker.com – Python® Notes for Professionals 18
one_member_tuple = tuple(['Only member'])
Dictionaries
A dictionary in Python is a collection of key-value pairs. The dictionary is surrounded by curly braces. Each pair is
separated by a comma and the key and value are separated by a colon. Here is an example:
state_capitals = {
'Arkansas': 'Little Rock',
'Colorado': 'Denver',
'California': 'Sacramento',
'Georgia': 'Atlanta'
}
To get a value, refer to it by its key:
ca_capital = state_capitals['California']
You can also get all of the keys in a dictionary and then iterate over them:
for k in state_capitals.keys():
print('{} is the capital of {}'.format(state_capitals[k], k))
Dictionaries strongly resemble JSON syntax. The native json module in the Python standard library can be used to
convert between JSON and dictionaries.
set
A set is a collection of elements with no repeats and without insertion order but sorted order. They are used in
situations where it is only important that some things are grouped together, and not what order they were
included. For large groups of data, it is much faster to check whether or not an element is in a set than it is to do
the same for a list.
Defining a set is very similar to defining a dictionary:
first_names = {'Adam', 'Beth', 'Charlie'}
Or you can build a set using an existing list:
my_list = [1,2,3]
my_set = set(my_list)
Check membership of the set using in:
if name in first_names:
print(name)
You can iterate over a set exactly like a list, but remember: the values will be in an arbitrary, implementation-
defined order.
defaultdict
A defaultdict is a dictionary with a default value for keys, so that keys for which no value has been explicitly
defined can be accessed without errors. defaultdict is especially useful when the values in the dictionary are
collections (lists, dicts, etc) in the sense that it does not need to be initialized every time when a new key is used.
GoalKicker.com – Python® Notes for Professionals 19
A defaultdict will never raise a KeyError. Any key that does not exist gets the default value returned.
For example, consider the following dictionary
>>> state_capitals = {
'Arkansas': 'Little Rock',
'Colorado': 'Denver',
'California': 'Sacramento',
'Georgia': 'Atlanta'
}
If we try to access a non-existent key, python returns us an error as follows
>>> state_capitals['Alabama']
Traceback (most recent call last):
File "<ipython-input-61-236329695e6f>", line 1, in <module>
state_capitals['Alabama']
KeyError: 'Alabama'
Let us try with a defaultdict. It can be found in the collections module.
>>> from collections import defaultdict
>>> state_capitals = defaultdict(lambda: 'Boston')
What we did here is to set a default value (Boston) in case the give key does not exist. Now populate the dict as
before:
>>> state_capitals['Arkansas'] = 'Little Rock'
>>> state_capitals['California'] = 'Sacramento'
>>> state_capitals['Colorado'] = 'Denver'
>>> state_capitals['Georgia'] = 'Atlanta'
If we try to access the dict with a non-existent key, python will return us the default value i.e. Boston
>>> state_capitals['Alabama']
'Boston'
and returns the created values for existing key just like a normal dictionary
>>> state_capitals['Arkansas']
'Little Rock