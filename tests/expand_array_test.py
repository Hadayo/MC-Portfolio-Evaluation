from context import portopt as pt

arr = pt.array(3)
arr.append_col([1, 2, 3])
arr.append_col([2, 3, 4])
print(arr)

print(arr.last_col())
